from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from math_verify import verify

from common.utils import load_yaml, repo_path, save_json, wall_timer
from task3_ttc.answers import NO_ANSWER, math_groups, math_parse, mathvista_groups, mathvista_normalize, scorer_versions

#cpu only. joins and checks the supplied rollout tables, makes the splits, and forms the answer groups and
#correctness labels every later step uses.
#
#  1. ATML-PA3-Data (google drive folder) and the mathvista testmini metadata (pinned github commit), downloaded once
#  2. the files of each model are joined on (dataset, sample_id, rollout_id) and every per-token list is checked to
#     have exactly n_generated_text_tokens entries (attention's per-token preceding counts must be 0 .. T-1)
#  3. splits -> the 1,000 question ids sorted lexicographically, permuted with default_rng(6304), first 200
#     development, remaining 800 evaluation. the first 50 math evaluation questions are the live subset
#  4. answer groups and correctness (answers.py), formed per question without the gold answer, then graded
#  5. per-rollout summaries -> T (retained tokens), L = mean logp, K = mean kl_u_p (self-certainty), and for
#     mathvista V = mean of img / (img + prompt + generated) attention (visual engagement)
#
#data/task3/
#    ATML-PA3-Data/            the supplied parquet files (never committed)
#    testmini.json             mathvista question metadata
#    rollouts_{dataset}.parquet   one row per rollout -> keys, text, answers, group, correctness, T, L, K, V
#    tokens_{dataset}.npz         flat per-token logp, kl (and visual engagement v) + row offsets, rollout row order
#    prepared/                    splits.json, rollouts_{dataset}.csv (no text), answer_groups_{dataset}.json,
#                                 prepare_summary.json -> copied into every results dir that uses this data
#                                 (results/task3_ttc, and the smoke run's), so the preparation runs only once


def fetch_raw(cfg):
    data = repo_path(cfg["data_dir"])
    folder = data / "ATML-PA3-Data"
    if not all((folder / name / "generations.parquet").exists() for name in cfg["datasets"].values()):
        data.mkdir(parents=True, exist_ok=True)
        print("[data] downloading ATML-PA3-Data")
        subprocess.run(["gdown", "--folder", cfg["rollout_drive_folder"], "-O", str(folder)], check=True)
    meta = data / "testmini.json"
    if not meta.exists():
        urllib.request.urlretrieve(cfg["mathvista_metadata_url"], meta)
    return folder, json.loads(meta.read_text())


def flat(table, column):
    #a list column -> one flat numpy array (all rollouts back to back) and the per-row lengths
    col = table.column(column).combine_chunks()
    return col.flatten().to_numpy(), np.diff(col.offsets.to_numpy())


def load_dataset(folder, dataset, name):
    #every file sorted by (sample_id, rollout_id), so rows line up across files and with the flat token arrays
    order = [("sample_id", "ascending"), ("rollout_id", "ascending")]
    read = lambda f, cols: pq.read_table(folder / name / f, columns=["dataset", "sample_id", "rollout_id", *cols]).sort_by(order)
    gen = read("generations.parquet", ["question_text", "generated_text", "generated_token_ids", "n_generated_text_tokens",
                                       "hit_max_new_tokens", "gold_answer", "extracted_answer"])
    prob = read("probability_metrics.parquet", ["logp", "kl_u_p"])
    tables = [gen, prob]
    if dataset == "mathvista":
        att = read("attention_metrics.parquet", ["n_generated_text_tokens", "img_attn_total", "prompt_text_attn_total",
                                                 "generated_text_attn_total"])
        tables.append(att)

    checks = {}
    keys = [t.select(["dataset", "sample_id", "rollout_id"]).to_pandas() for t in tables]
    checks["same_keys_in_every_file"] = all(k.equals(keys[0]) for k in keys[1:])
    rows = keys[0]
    rows["n_tokens"] = gen.column("n_generated_text_tokens").to_numpy()
    checks["single_dataset_name"] = rows["dataset"].nunique() == 1 and rows["dataset"].iloc[0] == dataset
    per_q = rows.groupby("sample_id")["rollout_id"].apply(list)
    checks["rollouts_0_to_31_per_question"] = all(ids == list(range(32)) for ids in per_q)
    _, id_len = flat(gen, "generated_token_ids")
    logp, logp_len = flat(prob, "logp")
    kl, kl_len = flat(prob, "kl_u_p")
    lengths = {"generated_token_ids": id_len, "logp": logp_len, "kl_u_p": kl_len}
    tokens = {"logp": logp, "kl": kl}
    if dataset == "mathvista":
        prev, prev_len = flat(att, "n_generated_text_tokens")
        img, img_len = flat(att, "img_attn_total")
        prompt, prompt_len = flat(att, "prompt_text_attn_total")
        generated, gen_len = flat(att, "generated_text_attn_total")
        lengths.update({"n_generated_text_tokens (attention)": prev_len, "img_attn_total": img_len,
                        "prompt_text_attn_total": prompt_len, "generated_text_attn_total": gen_len})
        starts = np.repeat(np.cumsum(prev_len) - prev_len, prev_len)
        checks["attention_preceding_counts_are_0_to_T-1"] = bool(np.array_equal(prev, np.arange(len(prev)) - starts))
        tokens["v"] = img / (img + prompt + generated) #visual engagement v_it
        checks["first_prediction_has_zero_generated_attention"] = bool((generated[np.cumsum(gen_len) - gen_len] == 0).all())
    for field, n in lengths.items():
        checks[f"{field}_matches_n_generated_text_tokens"] = bool(np.array_equal(n, rows["n_tokens"].to_numpy()))
    checks["finite_token_values"] = all(bool(np.isfinite(v).all()) for v in tokens.values())
    checks["no_empty_rollout"] = bool((rows["n_tokens"] > 0).all())
    assert all(checks.values()), checks

    text = gen.select(["question_text", "generated_text", "hit_max_new_tokens", "gold_answer", "extracted_answer"]).to_pandas()
    rows = pd.concat([rows, text], axis=1)
    offsets = np.concatenate([[0], np.cumsum(rows["n_tokens"].to_numpy())])
    means = lambda v: np.add.reduceat(v.astype(np.float64), offsets[:-1]) / rows["n_tokens"].to_numpy()
    rows["L"], rows["K"] = means(tokens["logp"]), means(tokens["kl"])
    if dataset == "mathvista":
        rows["V"] = means(tokens["v"])
    return rows, tokens, offsets, checks


def make_splits(ids, cfg):
    ordered = np.array(sorted(ids))[np.random.default_rng(int(cfg["seed"])).permutation(len(ids))].tolist()
    dev = int(cfg["dev_questions"])
    return {"order": ordered, "dev": ordered[:dev], "eval": ordered[dev:]}


def label_answers(rows, dataset, meta):
    #group index, group key and correctness for every rollout, plus the per-question group table
    group_col, key_col, correct_col, groups_out = [], [], [], {}
    checks, fallbacks = {"rollouts_graded_differently_alone": 0}, 0
    for sample_id, q in rows.groupby("sample_id", sort=False):
        answers = [None if pd.isna(a) else a for a in q["extracted_answer"]]
        gold = q["gold_answer"].iloc[0]
        if dataset == "math":
            index, groups = math_groups(answers, gold)
            #a group is graded through its representative. grading each distinct string on its own is a sanity check
            gold_parsed = math_parse(gold)
            for a, g in zip(answers, index):
                if a is not None and bool(verify(gold_parsed, math_parse(a.strip()))) != groups[g]["correct"]:
                    checks["rollouts_graded_differently_alone"] += 1
        else:
            index, groups = mathvista_groups(answers, gold, meta[sample_id])
            #multiple-choice extractions that are not an in-range option letter -> mapped by the similarity fallback
            if meta[sample_id]["question_type"] == "multi_choice":
                letters = {chr(ord("A") + i) for i in range(len(meta[sample_id]["choices"]))}
                fallbacks += sum(a is not None and a.strip() not in letters for a in answers)
        group_col += index
        key_col += [groups[g]["key"] for g in index]
        correct_col += [groups[g]["correct"] for g in index]
        groups_out[sample_id] = {"gold": gold, "groups": groups}
    rows["group"], rows["answer_key"], rows["correct"] = group_col, key_col, correct_col
    return groups_out, checks, fallbacks


def prepare(config_path):
    cfg = load_yaml(config_path)
    data, results = repo_path(cfg["data_dir"]), repo_path(cfg["results_dir"])
    prepared = data / "prepared"
    if not (data / "done.json").exists():
        build(cfg, data, prepared)
    else:
        print("[data] already prepared")
    results.mkdir(parents=True, exist_ok=True)
    for f in prepared.iterdir():
        shutil.copy(f, results / f.name)


def build(cfg, data, prepared):
    prepared.mkdir(parents=True, exist_ok=True)
    elapsed = wall_timer()
    folder, meta = fetch_raw(cfg)
    summary, splits = {"scorer": scorer_versions(), "datasets": {}}, {}
    for dataset, name in cfg["datasets"].items():
        rows, tokens, offsets, checks = load_dataset(folder, dataset, name)
        sp = make_splits(rows["sample_id"].unique().tolist(), cfg)
        splits[dataset] = sp
        split_of = {s: "dev" for s in sp["dev"]} | {s: "eval" for s in sp["eval"]}
        rows["split"] = rows["sample_id"].map(split_of)
        if dataset == "mathvista":
            gold = rows.groupby("sample_id")["gold_answer"].first()
            checks["mathvista_gold_answers_normalize_to_themselves"] = all(mathvista_normalize(g, meta[s]) == g for s, g in gold.items())
        groups, label_checks, fallbacks = label_answers(rows, dataset, meta)
        checks.update(label_checks)

        rows.to_parquet(data / f"rollouts_{dataset}.parquet", index=False)
        np.savez(data / f"tokens_{dataset}.npz", offsets=offsets, **tokens)
        cols = ["sample_id", "rollout_id", "split", "n_tokens", "hit_max_new_tokens", "extracted_answer", "answer_key",
                "group", "correct", "L", "K"] + (["V"] if dataset == "mathvista" else [])
        rows[cols].to_csv(prepared / f"rollouts_{dataset}.csv", index=False)
        save_json(prepared / f"answer_groups_{dataset}.json", groups)

        per_q = rows.groupby("sample_id")
        unparsed = sum(1 for q in groups.values() for g in q["groups"] if g["key"] != NO_ANSWER and not g.get("parsed", True))
        summary["datasets"][dataset] = {
            "checks": checks, "questions": int(rows["sample_id"].nunique()), "rollouts": len(rows),
            "retained_tokens": int(rows["n_tokens"].sum()), "capped_rollouts": int(rows["hit_max_new_tokens"].sum()),
            "null_extractions": int(rows["extracted_answer"].isna().sum()),
            "unconvertible_extraction_rollouts": int(sum(g["size"] for q in groups.values() for g in q["groups"]
                                                        if g["key"] != NO_ANSWER and not g.get("parsed", True))),
            "unconvertible_groups": unparsed, "multi_choice_similarity_fallbacks": fallbacks,
            "rollout_accuracy": float(rows["correct"].mean()),
            "mean_groups_per_question": float(per_q["group"].nunique().mean()),
            "questions_with_a_correct_rollout": int(per_q["correct"].any().sum()),
        }
        print(f"[data] {dataset} {summary['datasets'][dataset]}")
    splits["live_math"] = splits["math"]["eval"][:int(cfg["live_questions"])]
    save_json(prepared / "splits.json", splits)
    summary["seconds"] = elapsed()
    save_json(prepared / "prepare_summary.json", summary)
    save_json(data / "done.json", {"seconds": summary["seconds"]})
    print(f"[data] done in {elapsed() / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
