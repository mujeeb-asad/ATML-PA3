from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset

from common.utils import load_yaml, repo_path, save_json, write_jsonl
from task1_finetune.data import encode_example, load_tokenizer, normalize_question, supervised_tokens

#builds the two fixed inputs every later step reuses, before any model is trained or evaluated ->
#  1. the training set: the first 40,000 eligible MetaMathQA rows in source order (never shuffled, never truncated)
#  2. the activation probe set: 32 gsm8k test questions + 32 hellaswag validation contexts, sampled with seed 6304


def hellaswag_query(doc):
    #the exact context string lm-eval scores for hellaswag (its utils.process_docs), so the probe prompts match the
    #benchmark prompt format. imported from lm-eval itself rather than copied
    from lm_eval.tasks.hellaswag.utils import preprocess
    return preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())


def build_training_set(cfg, tokenizer):
    gsm_test = load_dataset(cfg["gsm8k_dataset"], "main", split="test")
    contaminated = {normalize_question(q) for q in gsm_test["question"]}

    source = load_dataset(cfg["train_dataset"], revision=cfg["train_dataset_revision"], split="train")
    target = int(cfg["train_examples"])
    kept, dropped = [], {"empty": 0, "too_long": 0, "gsm8k_test_match": 0}
    #rows are taken strictly in source order and the scan stops at the 40,000th eligible one
    for index, row in enumerate(source):
        query, response = row["query"] or "", row["response"] or ""
        if not query.strip() or not response.strip():
            dropped["empty"] += 1
            continue
        #either field matching a gsm8k test question disqualifies the row (query can be a rephrasing, original_question
        #is the question it was derived from)
        if normalize_question(query) in contaminated or normalize_question(row["original_question"] or "") in contaminated:
            dropped["gsm8k_test_match"] += 1
            continue
        example = encode_example(tokenizer, query, response)
        if len(example["input_ids"]) > int(cfg["max_length"]):
            dropped["too_long"] += 1
            continue
        kept.append({"source_index": index, "type": row["type"], "query": query, "response": response,
                     "tokens": len(example["input_ids"]), "response_tokens": supervised_tokens(example)})
        if len(kept) == target:
            break
    assert len(kept) == target, f"only {len(kept)} eligible rows"
    return kept, dropped, index + 1


def build_probe_set(cfg):
    #sampled once with seed 6304 and saved, before any model result exists. ids are row indices of the stock splits
    #(gsm8k main/test and hellaswag validation), the same doc ids lm-eval reports
    rng = np.random.default_rng(int(cfg["seed"]))
    n = int(cfg["probe_per_domain"])
    gsm_test = load_dataset(cfg["gsm8k_dataset"], "main", split="test")
    hellaswag = load_dataset(cfg["hellaswag_dataset"], split="validation")
    gsm_ids = sorted(int(i) for i in rng.choice(len(gsm_test), size=n, replace=False))
    hs_ids = sorted(int(i) for i in rng.choice(len(hellaswag), size=n, replace=False))
    return {
        "seed": int(cfg["seed"]),
        "gsm8k": [{"doc_id": i, "question": gsm_test[i]["question"]} for i in gsm_ids],
        "hellaswag": [{"doc_id": i, "query": hellaswag_query(hellaswag[i])} for i in hs_ids],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg)

    if repo_path(cfg["train_file"]).exists():
        print(f"training set already built -> {cfg['train_file']}")
    else:
        kept, dropped, scanned = build_training_set(cfg, tokenizer)
        write_jsonl(cfg["train_file"], kept)
        tokens = np.array([r["tokens"] for r in kept])
        response_tokens = np.array([r["response_tokens"] for r in kept])
        save_json(results / "train_selection.json", {
            "dataset": cfg["train_dataset"], "revision": cfg["train_dataset_revision"],
            "rows_scanned": scanned, "rows_kept": len(kept), "dropped": dropped,
            "tokens_mean": float(tokens.mean()), "tokens_max": int(tokens.max()),
            "supervised_tokens_total": int(response_tokens.sum()),
            "source_indices": [r["source_index"] for r in kept], #the exact rows used, in training order
        })
        print(f"kept {len(kept)} of {scanned} scanned rows, dropped {dropped}")

    #the probe set is never rebuilt once it exists, so every model is always diagnosed on the same 64 prompts
    if (results / "probe_set.json").exists():
        print("probe set already exists, kept as is")
    else:
        save_json(results / "probe_set.json", build_probe_set(cfg))
        print(f"probe set saved -> {results / 'probe_set.json'}")


if __name__ == "__main__":
    main()
