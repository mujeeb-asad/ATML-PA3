from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from common.utils import (append_jsonl, gpu_name, load_json, load_yaml, peak_memory_gb, read_jsonl, repo_path, save_json,
                          wall_timer)
from task3_ttc.answers import NO_ANSWER, last_boxed, math_groups, scorer_versions
from task3_ttc.offline import bootstrap_index, paired_bootstrap, weighted_vote

#step 4 -> parallel sampling vs a longer response vs sequential refinement, live, on the first 50 math evaluation
#questions. the same 3,072-token output allowance per question ->
#
#  single      one response capped at 3,072 new tokens
#  parallel    three independent responses capped at 1,024 each, self-consistency picks the answer (ties -> earliest)
#  refinement  the first parallel response (reused, cost charged to both strategies), then one critique-and-revision
#              response capped at 2,048 that sees it as an assistant message followed by the review instruction. the
#              revised answer is returned, never chosen against the initial one with gold answers
#
#qwen/qwen2.5-3b-instruct at the readme's revision, fp16, no quantization, batch size 1, native chat template, system
#message "You are a helpful assistant.", temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0. every call has
#its own seed, sha256("live|6304|sample_id|strategy|call") -> first 8 bytes (63 bits). time per call is end to end
#(chat template, tokenization, generation, decoding) with the model already loaded. output tokens exclude the stop
#token, as in the supplied export, and truncated = the call ended at its cap without a stop token.
#
#results/task3_ttc/
#    live_generations.jsonl   one record per call -> seed, prompt, response, token counts, truncation, extracted answer,
#                             seconds, peak memory (appended as generated, so an interrupted run resumes)
#    live_questions.csv       per (question, strategy) -> answer, correctness, input/output tokens, seconds
#    live_summary.json        per strategy totals and rates, refinement transitions, paired bootstrap differences


def call_seed(cfg, sample_id, strategy, call):
    digest = hashlib.sha256(f"live|{cfg['seed']}|{sample_id}|{strategy}|{call}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def load_model(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["live_model"], revision=cfg["live_revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["live_model"], revision=cfg["live_revision"], dtype=torch.float16).to(device)
    model.eval()
    stops = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    #the checkpoint's own generation_config carries repetition_penalty 1.05, so it is replaced outright rather than
    #merged, and nothing but these values reaches generate
    model.generation_config = GenerationConfig(
        do_sample=True, temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]),
        repetition_penalty=float(cfg["repetition_penalty"]), eos_token_id=stops, pad_token_id=stops[1], use_cache=True)
    return tokenizer, model, stops


@torch.no_grad()
def generate(model, tokenizer, stops, messages, cap, seed, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    torch.manual_seed(seed)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap)
    new = out[0, ids.shape[1]:].tolist()
    stopped = bool(new) and new[-1] in stops
    kept = new[:-1] if stopped else new
    text = tokenizer.decode(kept, skip_special_tokens=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"prompt": prompt, "response": text, "input_tokens": int(ids.shape[1]), "output_tokens": len(kept),
            "generated_tokens": len(new), "truncated": not stopped, "seconds": elapsed(), "peak_memory_gb": peak_memory_gb()}


def run_generation(cfg, questions, path):
    done = {(r["sample_id"], r["strategy"], r["call"]): r for r in read_jsonl(path)} if path.exists() else {}
    if sum(k[0] in set(questions["sample_id"]) for k in done) == len(questions) * (2 + int(cfg["parallel_responses"])):
        print("[live] generation already done")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, stops = load_model(cfg, device)
    for qi, q in enumerate(questions.itertuples()):
        user = {"role": "user", "content": q.question_text + cfg["question_suffix"]}
        base = [{"role": "system", "content": cfg["system_message"]}, user]
        calls = [("single", 0, base, int(cfg["single_cap"]))]
        calls += [("parallel", c, base, int(cfg["parallel_cap"])) for c in range(int(cfg["parallel_responses"]))]
        calls.append(("refinement", 0, None, int(cfg["refine_cap"]))) #messages filled in once the initial response exists
        for strategy, call, messages, cap in calls:
            if (q.sample_id, strategy, call) in done:
                continue
            if strategy == "refinement":
                initial = done[(q.sample_id, "parallel", 0)]["response"]
                messages = base + [{"role": "assistant", "content": initial}, {"role": "user", "content": cfg["refine_instruction"]}]
            seed = call_seed(cfg, q.sample_id, strategy, call)
            record = {"sample_id": q.sample_id, "strategy": strategy, "call": call, "seed": seed, "cap": cap,
                      **generate(model, tokenizer, stops, messages, cap, seed, device)}
            record["extracted_answer"] = last_boxed(record["response"])
            append_jsonl(path, record)
            done[(q.sample_id, strategy, call)] = record
            print(f"[live] {qi + 1}/{len(questions)} {q.sample_id} {strategy}[{call}] {record['output_tokens']} tokens "
                  f"{record['seconds']:.1f} s truncated {record['truncated']}")


def score(cfg, questions, records):
    #per (question, strategy) -> final answer, correctness and the strategy's total cost. the shared initial response
    #is charged to both parallel and refinement
    by = {(r["sample_id"], r["strategy"], r["call"]): r for r in records}
    rows = []
    for q in questions.itertuples():
        calls = {"single": [by[(q.sample_id, "single", 0)]],
                 "parallel": [by[(q.sample_id, "parallel", c)] for c in range(int(cfg["parallel_responses"]))],
                 "refinement": [by[(q.sample_id, "parallel", 0)], by[(q.sample_id, "refinement", 0)]]}
        for strategy, used in calls.items():
            if strategy == "parallel":
                index, groups = math_groups([r["extracted_answer"] for r in used], q.gold_answer)
                winner = weighted_vote(index, [1] * len(index))[0]
            else: #single -> its only response, refinement -> the revised (last) response
                index, groups = math_groups([used[-1]["extracted_answer"]], q.gold_answer)
                winner = 0
            row = {"sample_id": q.sample_id, "strategy": strategy, "answer": groups[winner]["key"],
                   "correct": bool(groups[winner]["correct"]), "extracted": groups[winner]["key"] != NO_ANSWER, #the final answer itself (parallel -> the voted one, not the last call)
                   "input_tokens": sum(r["input_tokens"] for r in used), "output_tokens": sum(r["output_tokens"] for r in used),
                   "seconds": sum(r["seconds"] for r in used), "calls": len(used),
                   "truncated_calls": sum(r["truncated"] for r in used), "extracted_calls": sum(r["extracted_answer"] is not None for r in used)}
            if strategy == "refinement":
                first = math_groups([used[0]["extracted_answer"]], q.gold_answer)[1][0]
                row.update({"initial_answer": first["key"], "initial_correct": bool(first["correct"]),
                            "revision_input_tokens": used[1]["input_tokens"], "revision_output_tokens": used[1]["output_tokens"],
                            "revision_truncated": used[1]["truncated"]})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(cfg, table, records):
    out = {"strategies": {}}
    for strategy, t in table.groupby("strategy", sort=False):
        calls = [r for r in records if r["strategy"] == strategy]
        out["strategies"][strategy] = {
            "questions": len(t), "accuracy": float(t["correct"].mean()), "final_answer_extracted_rate": float(t["extracted"].mean()),
            "mean_input_tokens": float(t["input_tokens"].mean()), "mean_output_tokens": float(t["output_tokens"].mean()),
            "mean_total_tokens": float((t["input_tokens"] + t["output_tokens"]).mean()),
            "mean_seconds": float(t["seconds"].mean()), "total_seconds": float(t["seconds"].sum()),
            "truncation_rate_of_strategy_calls": float(t["truncated_calls"].sum() / t["calls"].sum()),
            "extraction_rate_of_strategy_calls": float(t["extracted_calls"].sum() / t["calls"].sum()),
            "own_calls": len(calls), "own_call_truncation_rate": float(np.mean([r["truncated"] for r in calls])),
            "peak_memory_gb": max((r["peak_memory_gb"] or 0) for r in calls) if calls else None}
    ref = table[table["strategy"] == "refinement"]
    before, after = ref["initial_correct"].astype(bool), ref["correct"].astype(bool)
    out["refinement_transitions"] = {
        "wrong_to_correct": int((~before & after).sum()), "correct_to_wrong": int((before & ~after).sum()),
        "correct_to_correct": int((before & after).sum()), "wrong_to_wrong": int((~before & ~after).sum()),
        "answer_changed": int((ref["initial_answer"] != ref["answer"]).sum())}
    acc = table.pivot(index="sample_id", columns="strategy", values="correct").astype(float)
    index = bootstrap_index(len(acc), cfg)
    out["bootstrap"] = [{"strategy": a, "baseline": b, **paired_bootstrap(acc[a], acc[b], index)}
                        for a, b in [("parallel", "single"), ("refinement", "single"), ("refinement", "parallel")]]
    return out


def run(config_path, rescore=False):
    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    previous = load_json(results / "live_summary.json") if (results / "live_summary.json").exists() else None
    if previous and not rescore:
        print("[live] already done")
        return
    ids = load_json(results / "splits.json")["live_math"]
    ids = ids[:cfg["live_limit"]] if cfg.get("live_limit") else ids
    rows = pd.read_parquet(repo_path(cfg["data_dir"]) / "rollouts_math.parquet", columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
    questions = rows[rows["rollout_id"] == 0].set_index("sample_id").loc[ids].reset_index()
    path = results / "live_generations.jsonl"
    elapsed = wall_timer()
    run_generation(cfg, questions, path)
    records = [r for r in read_jsonl(path) if r["sample_id"] in set(ids)]
    table = score(cfg, questions, records)
    table.to_csv(results / "live_questions.csv", index=False)
    summary = summarize(cfg, table, records)
    summary.update({"gpu": gpu_name(), "model": cfg["live_model"], "revision": cfg["live_revision"], "dtype": "float16",
                    "batch_size": 1, "scorer": scorer_versions(), "questions": ids, "seconds_this_session": elapsed(),
                    "settings": {k: cfg[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "single_cap",
                                                     "parallel_responses", "parallel_cap", "refine_cap")}})
    if previous and rescore:
        #--rescore -> scores recomputed from the saved generations only (no new generation), so the card and the
        #session time stay those of the generation run
        summary.update({"gpu": previous["gpu"], "seconds_this_session": previous["seconds_this_session"], "rescored": True})
    save_json(results / "live_summary.json", summary)
    print(f"[live] {summary['strategies']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--rescore", action="store_true") #recompute the tables from the saved generations
    args = ap.parse_args()
    run(args.config, args.rescore)


if __name__ == "__main__":
    main()
