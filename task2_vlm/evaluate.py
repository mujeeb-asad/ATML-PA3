from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from common.utils import gpu_name, load_yaml, read_jsonl, repo_path, save_json, set_seed, wall_timer, write_jsonl
from task2_vlm.model import (FeatureBank, PromptEncoder, assemble, build_vlm, example_nll, generate, load_tokenizer,
                             load_vlm_state)
from task2_vlm.vqa_eval import normalize, soft_accuracy

#development metrics (called by train.py at every evaluation point) and the final evaluation of the stage 3 model.
#
#image conditions, the question and the original references always stay fixed ->
#  correct        the question's own image
#  complementary  the partner image of the same pair, whose reference answer differs
#  mismatched     an unrelated image from another pair of the same suite (fixed mapping, made in prepare_data)
#  none           no visual tokens at all, text unchanged, masks and positions rebuilt
#
#answer nll = teacher-forced nll of the reference target, averaged over its tokens (termination excluded) within each
#question, then over questions. the image-swap penalty is dNLL = NLL(y | q, I_partner) - NLL(y | q, I_correct)

CONDITIONS = ["correct", "complementary", "mismatched", "none"]
SUITES = {"grounding_test": "grounding", "vqa_test": "vqa"}


def question_rows(pairs, condition):
    #one row per question (two per pair), with the image this condition shows it
    rows = []
    for p in pairs:
        for slot in (0, 1):
            image = {"correct": p["images"][slot], "complementary": p["images"][1 - slot],
                     "mismatched": p.get("mismatched", [None, None])[slot], "none": None}[condition]
            rows.append({"pair": p, "slot": slot, "image": image})
    return rows


@torch.no_grad()
def batched_nll(vlm, pieces, images, supervise_end, batch_size):
    #example_nll over a whole list in fixed-size batches, eval mode, fp16 autocast -> one value per example
    out = []
    for i in range(0, len(pieces), batch_size):
        with torch.autocast(device_type=vlm.device.type, dtype=torch.float16):
            embeds, mask, labels = assemble(vlm.model, vlm.connector, vlm.bank, pieces[i:i + batch_size],
                                            images[i:i + batch_size], supervise_end=supervise_end)
            out.append(example_nll(vlm.model, embeds, mask, labels).float().cpu())
    return torch.cat(out).numpy()


def answer_pieces(vlm, rows, cfg):
    #the question with its reference target attached, for teacher forcing
    return [vlm.encoder.encode(r["pair"]["question"], r["pair"]["targets"][r["slot"]], int(cfg["answer_cap"])) for r in rows]


def dev_metrics(vlm, dev, cfg):
    #caption reference loss on the 96 development captions (same definition as the training loss, <|im_end|>
    #included) and the answer nll of both development suites under correct, complementary and no-image inputs
    was_training = vlm.model.training
    vlm.model.eval()
    batch = int(cfg["dev_batch_size"])
    caps = dev["captions"]
    pieces = [vlm.encoder.encode(cfg["caption_prompt"], c["caption"], int(cfg["caption_cap"])) for c in caps]
    metrics = {"caption_loss": float(batched_nll(vlm, pieces, [c["image"] for c in caps], True, batch).mean())}
    for suite in ("grounding_dev", "vqa_dev"):
        nll = {}
        for condition in ("correct", "complementary", "none"):
            rows = question_rows(dev[suite], condition)
            nll[condition] = batched_nll(vlm, answer_pieces(vlm, rows, cfg), [r["image"] for r in rows], False, batch)
            metrics[f"{suite}_nll_{condition}"] = float(nll[condition].mean())
        metrics[f"{suite}_swap_penalty"] = float((nll["complementary"] - nll["correct"]).mean())
    vlm.model.train(was_training)
    return metrics


def load_dev(cfg):
    splits = repo_path(cfg["data_dir"]) / "splits"
    return {"captions": read_jsonl(splits / "caption_dev.jsonl"), "grounding_dev": read_jsonl(splits / "grounding_dev.jsonl"),
            "vqa_dev": read_jsonl(splits / "vqa_dev.jsonl")}


#--------------------------------------------------------------------------------------------- final evaluation

def score_rows(rows, predictions, nll, suite_kind, condition, seconds):
    #per-question records. grounding -> normalized exact match against the template answer. vqa -> soft consensus
    #accuracy against all ten human answers. canonical_match (both suites) -> the prediction equals the question's
    #own normalized canonical answer, used for pair accuracy. partner_match -> it equals the partner image's answer
    records = []
    for r, pred, value in zip(rows, predictions, nll):
        p, slot = r["pair"], r["slot"]
        norm = normalize(pred)
        score = soft_accuracy(pred, p["references"][slot]) if suite_kind == "vqa" else float(norm == p["canonical"][slot])
        records.append({"pair_id": p["pair_id"], "type": p["type"], "slot": slot, "condition": condition,
                        "image": r["image"], "question": p["question"], "target": p["targets"][slot],
                        "canonical": p["canonical"][slot], "partner_canonical": p["canonical"][1 - slot],
                        "prediction": pred, "normalized": norm, "score": score,
                        "canonical_match": norm == p["canonical"][slot], "partner_match": norm == p["canonical"][1 - slot],
                        "empty": norm == "", "nll": float(value), "seconds_per_answer": seconds})
    return records


def pair_table(records):
    #one row per pair -> the pair-level values the bootstrap resamples (a complete pair is the resampling unit)
    by_pair = {}
    for r in records:
        by_pair.setdefault(r["pair_id"], [None, None])[r["slot"]] = r
    out = {}
    for pid, (a, b) in by_pair.items():
        out[pid] = {"type": a["type"], "accuracy": (a["score"] + b["score"]) / 2,
                    "pair_accuracy": float(a["canonical_match"] and b["canonical_match"]),
                    "identical": float(a["normalized"] == b["normalized"]), "empty": (a["empty"] + b["empty"]) / 2,
                    "nll": (a["nll"] + b["nll"]) / 2, "partner_match": (a["partner_match"] + b["partner_match"]) / 2}
    return out


def bootstrap(values, index):
    #mean and 95% percentile interval over resamples of complete pairs. index -> (resamples, n_pairs) pair indices,
    #drawn once per suite and shared by every condition, so differences between conditions use matched pairs
    values = np.asarray(values, dtype=float)
    means = values[index].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def summarize(records, cfg):
    #per suite and condition -> accuracy, pair accuracy and reference nll with pair-bootstrap intervals, plus the
    #identical-answer, empty-answer and partner-following rates and generation time per answer. per question type
    #the same accuracy and pair accuracy. correct-image advantages over complementary swaps, on matched pairs ->
    #  accuracy advantage = acc(correct) - acc(complementary), nll advantage = nll(complementary) - nll(correct)
    rng = np.random.default_rng(int(cfg["seed"]))
    summary = {}
    for suite in SUITES:
        tables = {c: pair_table([r for r in records if r["suite"] == suite and r["condition"] == c]) for c in CONDITIONS}
        pids = sorted(tables["correct"])
        index = rng.integers(len(pids), size=(int(cfg["bootstrap_resamples"]), len(pids)))
        col = lambda c, key, ids=pids: [tables[c][p][key] for p in ids]
        out = {"pairs": len(pids), "conditions": {}, "types": {}, "advantages": {}}
        for c in CONDITIONS:
            rows = [r for r in records if r["suite"] == suite and r["condition"] == c]
            out["conditions"][c] = {
                "accuracy": bootstrap(col(c, "accuracy"), index), "pair_accuracy": bootstrap(col(c, "pair_accuracy"), index),
                "nll": bootstrap(col(c, "nll"), index), "identical_rate": float(np.mean(col(c, "identical"))),
                "empty_rate": float(np.mean(col(c, "empty"))), "partner_match_rate": float(np.mean(col(c, "partner_match"))),
                "seconds_per_answer": float(np.mean([r["seconds_per_answer"] for r in rows]))}
        for t in sorted({tables["correct"][p]["type"] for p in pids}):
            ids = [p for p in pids if tables["correct"][p]["type"] == t]
            t_index = rng.integers(len(ids), size=(int(cfg["bootstrap_resamples"]), len(ids)))
            out["types"][t] = {c: {"accuracy": bootstrap(col(c, "accuracy", ids), t_index),
                                   "pair_accuracy": bootstrap(col(c, "pair_accuracy", ids), t_index)} for c in CONDITIONS}
        acc_gap = np.array(col("correct", "accuracy")) - np.array(col("complementary", "accuracy"))
        nll_gap = np.array(col("complementary", "nll")) - np.array(col("correct", "nll"))
        out["advantages"] = {"accuracy": bootstrap(acc_gap, index), "nll": bootstrap(nll_gap, index)}
        summary[suite] = out
    return summary


def final_evaluation(config_path):
    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    if (results / "eval_final.json").exists():
        print("[eval] already evaluated")
        return
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    model, connector, adapters = build_vlm(cfg, device)
    final_stage = cfg["stages"][-1]["name"]
    load_vlm_state(connector, adapters, load_file(str(repo_path(cfg["outputs_dir"]) / f"stage_{final_stage}.safetensors")))
    model.eval()
    vlm = SimpleNamespace(model=model, connector=connector, bank=FeatureBank(cfg, device), encoder=PromptEncoder(tokenizer, cfg),
                          device=device)
    splits = repo_path(cfg["data_dir"]) / "splits"
    limit = cfg.get("eval_limit_pairs")

    records, elapsed = [], wall_timer()
    for suite, kind in SUITES.items():
        pairs = read_jsonl(splits / f"{suite}.jsonl")[:limit] if limit else read_jsonl(splits / f"{suite}.jsonl")
        for condition in CONDITIONS:
            rows = question_rows(pairs, condition)
            prompts = [vlm.encoder.encode(r["pair"]["question"]) for r in rows]
            images = [r["image"] for r in rows]
            predictions, gen_seconds, size = [], 0.0, int(cfg["final_batch_size"])
            for i in range(0, len(rows), size): #greedy, 16 new tokens, batch size 8, thinking disabled
                if device.type == "cuda":
                    torch.cuda.synchronize()
                batch_time = wall_timer()
                predictions += generate(model, connector, vlm.bank, tokenizer, prompts[i:i + size], images[i:i + size],
                                        int(cfg["max_new_tokens"]))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gen_seconds += batch_time()
            nll = batched_nll(vlm, answer_pieces(vlm, rows, cfg), images, False, int(cfg["dev_batch_size"]))
            scored = score_rows(rows, predictions, nll, kind, condition, gen_seconds / len(rows))
            records += [{"suite": suite, **r} for r in scored]
            print(f"[eval] {suite:15s} {condition:13s} accuracy {np.mean([r['score'] for r in scored]):.3f}  "
                  f"empty {np.mean([r['empty'] for r in scored]):.3f}  nll {np.mean(nll):.3f}")

    write_jsonl(results / "eval_final_examples.jsonl", records) #raw references, generations, per-question metrics
    save_json(results / "eval_final.json", {
        "summary": summarize(records, cfg), "seconds": elapsed(), "gpu": gpu_name(), "final_stage": final_stage,
        "settings": {"decoding": "greedy", "max_new_tokens": int(cfg["max_new_tokens"]), "batch_size": int(cfg["final_batch_size"]),
                     "enable_thinking": False, "bootstrap_resamples": int(cfg["bootstrap_resamples"]), "limit_pairs": limit}})
    print(f"[eval] done in {elapsed() / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    final_evaluation(args.config)


if __name__ == "__main__":
    main()
