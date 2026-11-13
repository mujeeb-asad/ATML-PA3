from __future__ import annotations

import argparse
import json

import torch
from safetensors.torch import load_file

from common.utils import (clear_gpu, gpu_name, load_json, load_yaml, repo_path, save_json, set_seed, wall_timer,
                          write_jsonl)
from task1_finetune.adapters import inject_adapters, load_adapter_state
from task1_finetune.data import chat_prompt_ids, load_tokenizer
from task1_finetune.models import build_condition_model, checkpoint_dir, eval_model_dir, load_base, run_dir

#evaluates one of the seven conditions (--condition base | full_ft_mid | full_ft | l2sp | lora4 | lora16 | oplora16)
#with lm-evaluation-harness, identical settings for every condition ->
#  gsm8k_cot  zero-shot generation, non-thinking chat template, greedy, 1,024 new tokens, stock stops and filters
#  hellaswag / boolq / arc_easy  zero-shot candidate likelihood, stock completion prompts, no chat template
#every model is scored in fp16 with dropout off (lm-eval puts the model in eval mode)

TRUNCATION_SLACK = 8 #see gsm8k_records


def merge_check(cfg, condition, tokenizer, merged_fp32):
    #adapter conditions only -> logits of the merged model (W0 + dW in the weights) against the unmerged model
    #(frozen backbone + the trained adapter modules computing B A on the fly, oplora projecting its factors at
    #every forward pass) on a small batch: the first 4 gsm8k and first 4 hellaswag probe prompts
    spec = cfg["conditions"][condition]
    method_spec = cfg["methods"][spec["method"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_json(repo_path(cfg["results_dir"]) / "probe_set.json")
    prompts = [chat_prompt_ids(tokenizer, f"Q: {r['question']}\nA:") for r in probe["gsm8k"][:4]]
    prompts += [tokenizer(r["query"], add_special_tokens=False)["input_ids"] for r in probe["hellaswag"][:4]]

    unmerged = load_base(cfg, torch.float32)
    protect_k = method_spec.get("protect_k") if method_spec["kind"] == "oplora" else None
    adapters = inject_adapters(unmerged, cfg["adapter_targets"], int(method_spec["r"]), float(method_spec["alpha"]),
                               float(cfg["adapter_dropout"]), protect_k)
    load_adapter_state(adapters, load_file(str(checkpoint_dir(cfg, spec["method"], spec["update"]) / "adapter.safetensors")))
    if protect_k is not None:
        bases = load_file(str(run_dir(cfg, spec["method"]) / "bases.safetensors"))
        for name, a in adapters.items():
            a.set_bases(bases[f"{name}.U_k"], bases[f"{name}.V_k"])
    unmerged.to(device).eval()
    merged_fp32.to(device).eval()

    worst, agree, scale = 0.0, 0, 0.0
    with torch.no_grad():
        #one prompt at a time, so no padding enters the comparison. fp32 on both sides isolates the merge itself
        for ids in prompts:
            x = torch.tensor([ids], device=device)
            a, b = unmerged(input_ids=x).logits[0].float(), merged_fp32(input_ids=x).logits[0].float()
            worst = max(worst, float((a - b).abs().max()))
            scale = max(scale, float(a.abs().max()))
            agree += int((a.argmax(-1) == b.argmax(-1)).all())
    del unmerged
    clear_gpu()
    merged_fp32.cpu()
    return {"prompts": len(prompts), "dtype": "float32", "max_abs_logit_diff": worst, "max_abs_logit": scale,
            "prompts_with_identical_argmax": agree}


def build_eval_model(cfg, condition, tokenizer):
    #writes the fp16 copy lm-eval loads. adapter conditions are merged into a fresh copy of the starting model first
    #and checked against their unmerged form. the starting model itself is loaded straight from the hub
    spec = cfg["conditions"][condition]
    if spec["method"] is None:
        return None, None
    path = eval_model_dir(cfg, condition)
    check_path = repo_path(cfg["results_dir"]) / f"merge_check_{condition}.json"
    if (path / "config.json").exists():
        return path, load_json(check_path) if check_path.exists() else None
    check = None
    if cfg["methods"][spec["method"]]["kind"] == "full":
        model = build_condition_model(cfg, condition, torch.float16)
    else:
        merged = build_condition_model(cfg, condition, torch.float32)
        check = merge_check(cfg, condition, tokenizer, merged)
        save_json(check_path, check)
        print(f"[eval {condition}] merge check {check}")
        model = merged.to(torch.float16)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    del model
    clear_gpu()
    return path, check


def gsm8k_records(samples, tokenizer, cap):
    #lm-eval logs every gsm8k doc once per filter (strict-match and flexible-extract). both are kept, the primary
    #metric is flexible-extract (the last number-like match in the response). '[invalid]' is lm-eval's marker for
    #nothing extractable, which can never equal the target, so unextractable answers score as incorrect.
    #truncation -> lm-eval only keeps the decoded text, cut at the first stop string. a response that reached the
    #1,024-token cap had no stop and re-encodes to ~1,024 tokens (re-encoding can merge a few tokens), so it is
    #flagged when it re-encodes to within TRUNCATION_SLACK tokens of the cap
    by_doc = {}
    for s in samples:
        row = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"],
                                              "prompt": s["arguments"][0][0], #the exact chat-templated prompt string
                                              "response": s["resps"][0][0]})
        key = "flexible" if s["filter"] == "flexible-extract" else "strict"
        row[f"extracted_{key}"] = s["filtered_resps"][0]
        row[f"correct_{key}"] = float(s["exact_match"])
    records = []
    for doc_id in sorted(by_doc):
        row = by_doc[doc_id]
        #the stock flexible regex (-?[$0-9.,]{2,}) can also match digit-free text such as a closing '$$' latex
        #delimiter, so an extraction only counts as a number when it holds at least one digit
        row["extractable"] = any(ch.isdigit() for ch in row["extracted_flexible"])
        row["response_tokens"] = len(tokenizer(row["response"], add_special_tokens=False)["input_ids"])
        row["truncated"] = row["response_tokens"] >= cap - TRUNCATION_SLACK
        records.append(row)
    return records


def likelihood_records(samples):
    #one row per doc -> the loglikelihood of every candidate, the gold index and the per-doc metrics
    records = []
    for s in sorted(samples, key=lambda s: s["doc_id"]):
        row = {"doc_id": s["doc_id"], "target": s["target"], "choice_loglik": [float(r[0][0]) for r in s["resps"]],
               "acc": float(s["acc"])}
        if "acc_norm" in s:
            row["acc_norm"] = float(s["acc_norm"])
        records.append(row)
    return records


def evaluate_condition(config_path, condition):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    if (results / f"eval_{condition}.json").exists():
        print(f"[eval {condition}] already evaluated")
        return
    seed = int(cfg["seed"])
    set_seed(seed)
    tokenizer = load_tokenizer(cfg)
    path, check = build_eval_model(cfg, condition, tokenizer)

    #one fp16 lm-eval model for both calls. enable_thinking=False goes into every chat template call, so gsm8k
    #prompts end with the empty think block. it has no effect on the likelihood tasks, which use no chat template
    lm = HFLM(pretrained=str(path) if path else cfg["base_model"], revision=None if path else cfg["model_revision"],
              dtype="float16", batch_size=int(cfg["gsm8k_batch_size"]), enable_thinking=False,
              device="cuda" if torch.cuda.is_available() else "cpu")
    seeds = {"random_seed": seed, "numpy_random_seed": seed, "torch_random_seed": seed, "fewshot_random_seed": seed}
    limit = cfg.get("eval_limit")
    cap = int(cfg["gsm8k_max_new_tokens"])

    elapsed = wall_timer()
    gsm = lm_eval.simple_evaluate(model=lm, tasks=[cfg["gsm8k_task"]], num_fewshot=0, apply_chat_template=True,
                                  gen_kwargs={"max_gen_toks": cap}, limit=limit, log_samples=True, **seeds)
    gsm_seconds = elapsed()
    #the likelihood tasks get their own batch size. hflm reads batch_size_per_gpu at every call, so it is switched
    #in place. the longest boolq prompt is ~1,270 tokens and lm-eval scores the longest prompts first, padded
    #together, so the full-vocabulary log-softmax of that first batch sets the memory ceiling here
    lm.batch_size_per_gpu = int(cfg["likelihood_batch_size"])
    elapsed = wall_timer()
    mc = lm_eval.simple_evaluate(model=lm, tasks=list(cfg["likelihood_tasks"]), num_fewshot=0, apply_chat_template=False,
                                 limit=limit, log_samples=True, **seeds)
    mc_seconds = elapsed()

    task = cfg["gsm8k_task"]
    gsm_rows = gsm8k_records(gsm["samples"][task], tokenizer, cap)
    write_jsonl(results / f"eval_{condition}_gsm8k.jsonl", gsm_rows)
    scores = {
        "gsm8k_flexible": gsm["results"][task]["exact_match,flexible-extract"], #primary
        "gsm8k_flexible_stderr": gsm["results"][task]["exact_match_stderr,flexible-extract"],
        "gsm8k_strict": gsm["results"][task]["exact_match,strict-match"],
        "gsm8k_extractable_rate": sum(r["extractable"] for r in gsm_rows) / len(gsm_rows),
        "gsm8k_truncation_rate": sum(r["truncated"] for r in gsm_rows) / len(gsm_rows),
        "gsm8k_response_tokens_mean": sum(r["response_tokens"] for r in gsm_rows) / len(gsm_rows),
    }
    counts = {"gsm8k": len(gsm_rows)}
    for name in cfg["likelihood_tasks"]:
        rows = likelihood_records(mc["samples"][name])
        write_jsonl(results / f"eval_{condition}_{name}.jsonl", rows)
        counts[name] = len(rows)
        res = mc["results"][name]
        scores[f"{name}_acc"], scores[f"{name}_acc_stderr"] = res["acc,none"], res["acc_stderr,none"]
        if "acc_norm,none" in res: #hellaswag and arc_easy (primary), boolq has plain accuracy only
            scores[f"{name}_acc_norm"], scores[f"{name}_acc_norm_stderr"] = res["acc_norm,none"], res["acc_norm_stderr,none"]

    summary = {
        "condition": condition, "spec": cfg["conditions"][condition], "model_path": str(path) if path else cfg["base_model"],
        "scores": scores, "examples": counts, "merge_check": check,
        "gsm8k_seconds": gsm_seconds, "likelihood_seconds": mc_seconds, "gpu": gpu_name(),
        "settings": {"dtype": "float16", "num_fewshot": 0, "gsm8k_chat_template": True, "enable_thinking": False,
                     "gsm8k_max_gen_toks": cap, "greedy": True, "gsm8k_batch_size": int(cfg["gsm8k_batch_size"]),
                     "likelihood_batch_size": int(cfg["likelihood_batch_size"]), "limit": limit},
        #the exact task definitions and harness version used, kept as the record of the fixed scoring setup
        "lm_eval": json.loads(json.dumps({"version": lm_eval.__version__, "versions": {**gsm["versions"], **mc["versions"]},
                                          "configs": {**gsm["configs"], **mc["configs"]}}, default=str)),
    }
    save_json(results / f"eval_{condition}.json", summary)
    print(f"[eval {condition}] " + "  ".join(f"{k} {v:.4f}" for k, v in scores.items() if "stderr" not in k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--condition", required=True)
    args = ap.parse_args()
    evaluate_condition(args.config, args.condition)


if __name__ == "__main__":
    main()
