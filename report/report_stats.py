from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
#local helpers instead of common.utils, so the report scripts run without torch
repo_path = lambda p: REPO / p
read_jsonl = lambda p: [json.loads(line) for line in Path(p).open(encoding="utf-8") if line.strip()]
save_json = lambda p, obj: Path(p).write_text(json.dumps(obj, indent=2), encoding="utf-8")

#the few numbers the report quotes that no task script saves, computed from the saved per-example results only
#(nothing is rerun, no model is loaded). everything lands in results/report/, so every number in the report traces
#to a file. seed 6304 for every bootstrap
#
#  task 1  paired tests between conditions (bootstrap ci of the difference + exact mcnemar), an extraction-controlled
#          gsm8k delta (base rescored with its own last \boxed{} answer), boolq yes-rate and ranking auc
#  task 2  constant-answer baselines per question type, how the no-image answers look (sentences, not short answers)
#  task 3  live discordant questions and where refinement changed answers, how often single-score rules pick
#          no_answer / single-rollout answers, adaptive policies against fixed-n self-consistency at equal cost

OUT = repo_path("results/report")
T1 = repo_path("results/task1_finetune")
T2 = repo_path("results/task2_vlm")
T2_49 = repo_path("results/task2_vlm_49")
T3 = repo_path("results/task3_ttc")
SEED, DRAWS = 6304, 10_000

CONDITIONS = ["base", "full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]
METRIC = {"gsm8k": "correct_flexible", "hellaswag": "acc_norm", "boolq": "acc", "arc_easy": "acc_norm"}
#base vs every condition, then the manual's controlled comparisons (a -> b means b minus a)
PAIRS = [("base", c) for c in CONDITIONS[1:]] + [("lora4", "lora16"), ("full_ft", "l2sp"), ("lora16", "oplora16"),
                                                  ("full_ft", "lora16"), ("full_ft_mid", "full_ft")]


def paired(x_a, x_b, rng):
    #x_a, x_b -> 0/1 per example in the same doc order. difference in pp, percentile bootstrap ci over examples,
    #exact mcnemar p on the discordant examples
    d = x_b.astype(float) - x_a.astype(float)
    boots = d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(1) * 100
    gain, loss = int(((x_a == 0) & (x_b == 1)).sum()), int(((x_a == 1) & (x_b == 0)).sum())
    p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
    return {"diff_pp": 100 * d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "b_gains": gain, "b_losses": loss, "mcnemar_p": p}


def number(s):
    #numeric value of an answer string, $ , and backslashes removed -> None if it is not a number
    if not isinstance(s, str):
        return None
    try:
        return float(s.replace(",", "").replace("$", "").replace("\\", "").strip().rstrip("."))
    except ValueError:
        return None


def task1():
    rng = np.random.default_rng(SEED)
    load = lambda c, b: pd.DataFrame(read_jsonl(T1 / f"eval_{c}_{b}.jsonl")).set_index("doc_id").sort_index()
    data = {(c, b): load(c, b) for c in CONDITIONS for b in METRIC}

    rows = []
    for b, col in METRIC.items():
        for a, c in PAIRS:
            rows.append({"benchmark": b, "a": a, "b": c, **paired(data[(a, b)][col].values, data[(c, b)][col].values, rng)})
    pd.DataFrame(rows).to_csv(OUT / "t1_paired_tests.csv", index=False)

    #gsm8k with the extraction artefact removed. the stock flexible regex takes the last number-like match, which for
    #a quarter of the base responses is the closing $$ after a correct \boxed{n}. control -> count a base response
    #correct when its last \boxed{} value equals the target numerically. fine-tuned responses are 100% extractable,
    #so they keep their stock score
    base = data[("base", "gsm8k")]
    boxed = base["response"].map(lambda s: (re.findall(r"\\boxed\{([^{}]*)\}", s) or [None])[-1])
    boxed_ok = np.array([number(x) is not None and number(t) is not None and abs(number(x) - number(t)) < 1e-9
                         for x, t in zip(boxed, base["target"])])
    unextractable = ~base["extractable"].values.astype(bool)
    base_adj = base["correct_flexible"].values.astype(bool) | (boxed_ok & unextractable)
    rows = [{"condition": "base", "stock": 100 * base["correct_flexible"].mean(), "adjusted": 100 * base_adj.mean(),
             "unextractable": int(unextractable.sum()), "unextractable_boxed_correct": int((boxed_ok & unextractable).sum())}]
    for c in CONDITIONS[1:]:
        x = data[(c, "gsm8k")]["correct_flexible"].values
        t = paired(base_adj.astype(int), x, rng)
        rows.append({"condition": c, "stock": 100 * x.mean(), "adjusted": 100 * x.mean(),
                     "stock_delta_pp": 100 * (x.mean() - base["correct_flexible"].mean()),
                     "adjusted_delta_pp": t["diff_pp"], "adjusted_ci_lo": t["ci_lo"], "adjusted_ci_hi": t["ci_hi"],
                     "adjusted_mcnemar_p": t["mcnemar_p"]})
    pd.DataFrame(rows).to_csv(OUT / "t1_gsm8k_adjusted.csv", index=False)

    #boolq -> lm-eval's choices are [no, yes]. share of yes predictions, and a threshold-free ranking quality (auc of
    #loglik(yes) - loglik(no) between gold-yes and gold-no questions), so a shifted yes/no prior and a change in
    #discrimination can be told apart
    rows = []
    for c in CONDITIONS:
        d = data[(c, "boolq")]
        ll = np.array(d["choice_loglik"].tolist())
        margin, gold = ll[:, 1] - ll[:, 0], d["target"].values
        auc = mannwhitneyu(margin[gold == 1], margin[gold == 0]).statistic / ((gold == 1).sum() * (gold == 0).sum())
        rows.append({"condition": c, "accuracy": 100 * d["acc"].mean(), "predicted_yes_pct": 100 * (margin > 0).mean(),
                     "gold_yes_pct": 100 * gold.mean(), "auc": auc})
    pd.DataFrame(rows).to_csv(OUT / "t1_boolq.csv", index=False)

    #answer markers of the math training data in the 168 qualitative replies (none of the 24 prompts is maths)
    rows = []
    for c in CONDITIONS:
        q = pd.DataFrame(read_jsonl(T1 / f"qualitative_{c}.jsonl"))
        marker = q["response"].str.contains(r"(?:The answer is:\s*-?\d|####\s*-?\d)", regex=True)
        rows.append({"condition": c, "replies": len(q), "numbered_marker": int(marker.sum()),
                     "truncated": int(q["truncated"].sum()), "mean_tokens": q["response_tokens"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "t1_qualitative_counts.csv", index=False)

    #training logs -> mean sft loss over the last 250 updates (the same batches in every run, so means compare
    #directly), pre-clip gradient norm, and how many updates were clipped (global norm above 1.0)
    rows = []
    for m in ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]:
        log = pd.DataFrame(read_jsonl(T1 / f"train_log_{m}.jsonl"))
        g = log["grad_norm"][np.isfinite(log["grad_norm"])]
        rows.append({"method": m, "sft_loss_last250": log[log["update"] > 1000]["sft_loss"].mean(),
                     "grad_norm_median": g.median(), "clipped_updates": int((g > 1.0).sum()), "finite_updates": len(g),
                     "clipped_after_warmup": int((log[(log["update"] > 38) & np.isfinite(log["grad_norm"])]["grad_norm"] > 1.0).sum()),
                     #share of the summed learning rate spent by the midpoint checkpoint (update 625)
                     "lr_share_by_update_625": log[log["update"] <= 625]["lr"].sum() / log["lr"].sum()})
    pd.DataFrame(rows).to_csv(OUT / "t1_training_summary.csv", index=False)


def task2():
    rows = []
    for run, folder in [("196", T2), ("49", T2_49)]:
        e = pd.DataFrame(read_jsonl(folder / "eval_final_examples.jsonl"))
        #constant-answer baselines from the canonical answers -> the best single answer for each question type
        for (suite, t), g in e[e["condition"] == "correct"].groupby(["suite", "type"]):
            counts = g["canonical"].value_counts()
            rows.append({"run": run, "suite": suite, "type": t, "questions": len(g), "best_constant": counts.index[0],
                         "best_constant_pct": 100 * counts.iloc[0] / len(g)})
    pd.DataFrame(rows).to_csv(OUT / "t2_constant_baselines.csv", index=False)

    #what the model says without an image -> share of answers longer than three words (a full sentence instead of
    #the short answer the questions trained it to give), and the identical-answer rate between a pair's two
    #questions (identical inputs, so greedy decoding must give 100%)
    rows = []
    for run, folder in [("196", T2), ("49", T2_49)]:
        e = pd.DataFrame(read_jsonl(folder / "eval_final_examples.jsonl"))
        for suite, g in e.groupby("suite"):
            for cond, h in g.groupby("condition"):
                words = h["prediction"].str.split().str.len()
                rows.append({"run": run, "suite": suite, "condition": cond, "sentence_pct": 100 * (words > 3).mean(),
                             "mean_words": words.mean()})
    pd.DataFrame(rows).to_csv(OUT / "t2_answer_form.csv", index=False)

    #where swapped answers go -> the original image's canonical answer, the partner image's, or neither
    rows = []
    for run, folder in [("196", T2), ("49", T2_49)]:
        e = pd.DataFrame(read_jsonl(folder / "eval_final_examples.jsonl"))
        for suite, g in e[e["condition"] == "complementary"].groupby("suite"):
            rows.append({"run": run, "suite": suite, "original_pct": 100 * g["canonical_match"].mean(),
                         "partner_pct": 100 * g["partner_match"].mean(),
                         "neither_pct": 100 * (~g["canonical_match"] & ~g["partner_match"]).mean()})
    pd.DataFrame(rows).to_csv(OUT / "t2_swap_breakdown.csv", index=False)


def task3():
    out = {}
    #live -> per-question discordance between strategies, and where the revision changed the shared initial answer
    lq = pd.read_csv(T3 / "live_questions.csv")
    correct = lq.pivot(index="sample_id", columns="strategy", values="correct").astype(bool)
    answer = lq.pivot(index="sample_id", columns="strategy", values="answer")
    ref = lq[lq["strategy"] == "refinement"].set_index("sample_id")
    changed = ref["initial_answer"] != ref["answer"]
    extracted_initial = ref["initial_answer"] != "NO_ANSWER"
    out["live"] = {
        "discordant": {f"{a}_right_{b}_wrong": int((correct[a] & ~correct[b]).sum())
                       for a in ["single", "parallel", "refinement"] for b in ["single", "parallel", "refinement"] if a != b},
        "parallel_no_answer_questions": int((answer["parallel"] == "NO_ANSWER").sum()),
        "refinement_changed": int(changed.sum()),
        "refinement_changed_from_no_answer": int((changed & ~extracted_initial).sum()),
        "questions_with_extracted_initial": int(extracted_initial.sum()),
        "accuracy_initial_on_extracted": float(ref.loc[extracted_initial, "initial_correct"].mean()),
        "accuracy_revised_on_extracted": float(ref.loc[extracted_initial, "correct"].mean()),
        "single_output_share_of_allowance": float(lq[lq["strategy"] == "single"]["output_tokens"].sum() / (50 * 3072)),
    }

    #single-score rules as the pool grows -> how often the picked answer is no_answer or has a single supporting
    #rollout in the full pool (evaluation split)
    rows = []
    for d, rules in [("math", ["self_consistency", "mean_logp", "self_certainty", "self_certainty_rwc", "shortest"]),
                     ("mathvista", ["self_consistency", "mean_logp", "self_certainty", "vision", "visconf", "visconf_rwc"])]:
        groups = json.loads((T3 / f"answer_groups_{d}.json").read_text())
        s = pd.read_parquet(T3 / f"selections_{d}.parquet")
        s = s[s["split"] == "eval"]
        for n, g in s.groupby("N"):
            for r in rules:
                picked = [groups[q]["groups"][int(x)] for q, x in zip(g["sample_id"], g[r])]
                rows.append({"dataset": d, "N": n, "rule": r,
                             "no_answer_pct": 100 * np.mean([p["key"] == "NO_ANSWER" for p in picked]),
                             "singleton_pct": 100 * np.mean([p["size"] == 1 for p in picked])})
    pd.DataFrame(rows).to_csv(OUT / "t3_pick_support.csv", index=False)

    #where the oracle-coverage gain from n = 1 to 32 lands, by the share of correct rollouts in the full pool of 32
    #(retrospective, uses gold labels), and what self-consistency does in the same questions
    rows = []
    for d in ["math", "mathvista"]:
        pq = pd.read_csv(T3 / f"per_question_{d}.csv")
        pq = pq[(pq["split"] == "eval") & (pq["method"] == "self_consistency")]
        frac = pd.read_csv(T3 / f"rollouts_{d}.csv").groupby("sample_id")["correct"].mean()
        wide = pq.pivot(index="sample_id", columns="N", values=["oracle", "accuracy"])
        gain = wide[("oracle", 32)] - wide[("oracle", 1)]
        bins = pd.cut(frac.reindex(wide.index), [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0],
                      labels=["0", "(0, .25]", "(.25, .5]", "(.5, .75]", "(.75, 1)", "1"])
        for b, idx in wide.groupby(bins, observed=True).groups.items():
            rows.append({"dataset": d, "correct_fraction": b, "questions": len(idx),
                         "share_of_coverage_gain": gain[idx].sum() / gain.sum(),
                         "sc_accuracy_n1": wide.loc[idx, ("accuracy", 1)].mean(),
                         "sc_accuracy_n32": wide.loc[idx, ("accuracy", 32)].mean()})
    pd.DataFrame(rows).to_csv(OUT / "t3_coverage_gain_by_difficulty.csv", index=False)

    #adaptive policies against fixed-n self-consistency interpolated (linearly in cost) at the same mean cost
    ev = pd.read_csv(T3 / "adaptive_eval.csv")
    rows = []
    for d, g in ev.groupby("dataset"):
        fixed = g[g["policy"].str.startswith("fixed")].sort_values("cost")
        for _, r in g[~g["policy"].str.startswith("fixed")].iterrows():
            interp = np.interp(r["cost"], fixed["cost"], fixed["accuracy"])
            f32 = fixed[fixed["policy"] == "fixed_32"].iloc[0]
            rows.append({"dataset": d, "policy": r["policy"], "accuracy": r["accuracy"], "cost": r["cost"],
                         "cost_share_of_fixed_32": r["cost"] / f32["cost"], "fixed_interp_accuracy": interp,
                         "minus_interp_pp": 100 * (r["accuracy"] - interp)})
    pd.DataFrame(rows).to_csv(OUT / "t3_adaptive_vs_fixed.csv", index=False)
    save_json(OUT / "t3_live_extra.json", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task1()
    task2()
    task3()
    print("[report] statistics written to", OUT)


if __name__ == "__main__":
    main()
