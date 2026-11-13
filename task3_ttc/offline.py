from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common.utils import load_json, load_yaml, repo_path, save_json, wall_timer
from task3_ttc.answers import NO_ANSWER

#steps 1 to 3 on the supplied pools (cpu only) -> repeated sampling, selection rules and what the signals capture.
#
#candidate orders -> per question, ten seeded random permutations of the 32 rollout ids (default_rng(6304), questions
#in sorted id order). budget N uses the first N ids of an order, so budgets are nested and every method sees the
#same candidates. outcomes are averaged over the ten orders within a question, then over questions.
#
#quality scores of rollout i (T_i retained tokens) ->
#  L_i  mean logp                                   (mean log-probability)
#  K_i  mean kl_u_p = mean KL(U || p_it)            (self-certainty)
#  V_i  mean img / (img + prompt + generated)       (visual engagement, mathvista)
#  Z_i  V_i * C_i, C_i = mean_t c_it                (visconf)
#  Z_i^token = mean_t c_it v_it                     (tokenwise fusion ablation, Z^token - Z = Cov_t(c, v))
#  where c_it = clip[0,1]((k_it - Q05(G)) / max(Q95(G) - Q05(G), 1e-8)), Q over the pooled token kl values of the
#  current candidate set G only (linear interpolation), recomputed for every prefix
#  shortest -> score -T_i, the length-only control
#
#selection ->
#  first             the first candidate (consumes only that candidate's tokens)
#  vote              self-consistency, one vote per rollout
#  highest           highest score, ties -> the earliest rollout in the order
#  rwc               rank-weighted consensus. descending average ranks r_i (1 = best, ties share the mean rank),
#                    w_i = (N - r_i) / (N - 1), W(a) = sum of w_i over the rollouts answering a
#  vote ties (plain or weighted) -> the answer whose earliest supporting rollout comes first in the order. at N = 1
#  every rule returns the only candidate. NO_ANSWER is an ordinary group in every vote and is always incorrect
#
#results/task3_ttc/
#    orders_{dataset}.json              the ten orders of every question (candidate ids of every prefix)
#    selections_{dataset}.parquet       one row per (question, order, N) -> cost, coverage, groups, the group each
#                                       method selects and whether it is correct, vote counts and rwc margins
#    per_question_{dataset}.csv         per (question, N, method) -> accuracy, token cost, coverage, averaged over orders
#    offline_summary.csv                per (dataset, split, N, method) -> means over questions + failure decomposition
#    offline_bootstrap.csv              paired question-level bootstrap intervals of the main accuracy differences
#    signals_per_question_{dataset}.csv, signals_summary.json, confidence_vision_rollouts.csv, trajectories.json

METHODS = {"first": ("first", None), "self_consistency": ("vote", None), "mean_logp": ("highest", "L"),
           "self_certainty": ("highest", "K"), "self_certainty_rwc": ("rwc", "K"), "shortest": ("highest", "neg_length")}
VISION_METHODS = {"vision": ("highest", "V"), "vision_rwc": ("rwc", "V"), "visconf": ("highest", "Z"),
                  "visconf_rwc": ("rwc", "Z"), "visconf_token": ("highest", "Z_token"), "visconf_token_rwc": ("rwc", "Z_token")}


def methods_for(dataset):
    return {**METHODS, **VISION_METHODS} if dataset == "mathvista" else dict(METHODS)


#--------------------------------------------------------------------------------------------- loading

class Pool:
    #the 32 supplied rollouts of every question of one dataset -> scalar arrays (question, rollout) and, for the
    #prefix-calibrated visconf scores and the trajectories, the per-token values
    def __init__(self, cfg, dataset):
        data = repo_path(cfg["data_dir"])
        cols = ["sample_id", "rollout_id", "split", "n_tokens", "group", "correct", "answer_key", "L", "K"]
        rows = pd.read_parquet(data / f"rollouts_{dataset}.parquet", columns=cols + (["V"] if dataset == "mathvista" else []))
        self.dataset, self.ids = dataset, rows["sample_id"].unique().tolist() #sorted ids (the files were sorted)
        shape = (len(self.ids), int(cfg["rollouts_per_question"]))
        grid = lambda c: rows[c].to_numpy().reshape(shape)
        self.split = dict(zip(self.ids, grid("split")[:, 0]))
        self.T, self.group, self.correct = grid("n_tokens"), grid("group"), grid("correct")
        self.key = grid("answer_key")
        self.scores = {"L": grid("L"), "K": grid("K"), "neg_length": -grid("n_tokens").astype(float)}
        if dataset == "mathvista":
            self.scores["V"] = grid("V")
        tokens = np.load(data / f"tokens_{dataset}.npz")
        self.offsets = tokens["offsets"]
        self.tokens = {name: tokens[name] for name in tokens.files if name != "offsets"}
        self.row = {s: i for i, s in enumerate(self.ids)}

    def token(self, name, q, i):
        start = q * self.T.shape[1] + i
        return self.tokens[name][self.offsets[start]:self.offsets[start + 1]]


def make_orders(pool, cfg):
    rng = np.random.default_rng(int(cfg["seed"]))
    n = pool.T.shape[1]
    return {s: [rng.permutation(n).tolist() for _ in range(int(cfg["orders"]))] for s in pool.ids}


def selected_ids(pool, cfg, splits):
    #every question, or for the smoke test the first question_limit questions of each split (permutation order)
    limit = cfg.get("question_limit")
    sp = splits[pool.dataset]
    return sp["dev"][:limit] + sp["eval"][:limit] if limit else sp["dev"] + sp["eval"]


#--------------------------------------------------------------------------------------------- scores and rules

def visconf_scores(pool, q, idx, cfg):
    #Z and Z^token for the candidates idx (in order), calibrated on the pooled token kl of idx only
    kl = [pool.token("kl", q, i) for i in idx]
    v = [pool.token("v", q, i) for i in idx]
    lo, hi = np.quantile(np.concatenate(kl), cfg["visconf_quantiles"]) #numpy's default is linear interpolation
    scale = max(hi - lo, float(cfg["visconf_eps"]))
    c = [np.clip((k - lo) / scale, 0.0, 1.0) for k in kl]
    z = np.array([vi.mean() * ci.mean() for vi, ci in zip(v, c)])
    z_token = np.array([(ci * vi).mean() for vi, ci in zip(v, c)])
    return z, z_token, c


def weighted_vote(groups, weights):
    #total weight per answer group in order of first support, so max() resolves ties to the earliest supporter.
    #returns the winner, the winning total, the runner-up total (0 when only one group) and the sum of weights
    totals = {}
    for g, w in zip(groups, weights):
        totals[g] = totals.get(g, 0) + w
    winner = max(totals, key=totals.get)
    ranked = sorted(totals.values(), reverse=True)
    return winner, ranked[0], ranked[1] if len(ranked) > 1 else 0, sum(ranked)


def rwc(groups, score):
    #integer weights 2 (N - r_i), the paper's (N - r_i) / (N - 1) times the constant 2 (N - 1). average ranks are
    #multiples of 0.5, so the vote totals are exact integers and ties are real ties. the margin is scale-free
    ranks = rankdata(-score, method="average")
    winner, top, second, total = weighted_vote(groups, (2 * (len(groups) - ranks)).astype(int).tolist())
    return winner, (top - second) / total


def select_prefix(pool, q, idx, methods, cfg):
    #one (question, order, N) -> the group each method selects, plus the vote statistics the adaptive controllers use
    groups = pool.group[q, idx].tolist()
    scores = {name: values[q, idx] for name, values in pool.scores.items()}
    if pool.dataset == "mathvista":
        scores["Z"], scores["Z_token"], _ = visconf_scores(pool, q, idx, cfg)
    out = {}
    winner, top, second, _ = weighted_vote(groups, [1] * len(groups))
    out["sc_votes"], out["sc_runner_up"] = top, second
    for method, (rule, score) in methods.items():
        if len(idx) == 1 or rule == "first":
            out[method] = groups[0]
        elif rule == "vote":
            out[method] = winner
        elif rule == "highest":
            out[method] = groups[int(np.argmax(scores[score]))] #argmax -> first maximum = earliest in the order
        else:
            out[method], out[f"{method}_margin"] = rwc(groups, scores[score])
    return out


#--------------------------------------------------------------------------------------------- steps 1 and 2

def run_selections(pool, cfg, orders, ids):
    methods = methods_for(pool.dataset)
    records = []
    for s in ids:
        q = pool.row[s]
        correct_of = dict(zip(pool.group[q].tolist(), pool.correct[q].tolist()))
        key_of = dict(zip(pool.group[q].tolist(), pool.key[q].tolist()))
        for o, order in enumerate(orders[s]):
            for n in cfg["budgets"]:
                idx = order[:n]
                rec = {"sample_id": s, "split": pool.split[s], "order": o, "N": n,
                       "cost": int(pool.T[q, idx].sum()), "first_cost": int(pool.T[q, idx[0]]),
                       "oracle": bool(pool.correct[q, idx].any()), "n_groups": len(set(pool.group[q, idx].tolist()))}
                picked = select_prefix(pool, q, idx, methods, cfg) #group indices, see answer_groups_{dataset}.json
                picked["sc_no_answer"] = key_of[picked["self_consistency"]] == NO_ANSWER
                for method in methods:
                    picked[f"{method}_correct"] = correct_of[picked[method]]
                records.append({**rec, **picked})
    return pd.DataFrame(records)


def per_question(sel, dataset):
    #average over orders within a question. firstans is charged only its first candidate, every other method all N
    rows = []
    grouped = sel.groupby(["sample_id", "split", "N"], sort=False)
    means = grouped[[c for c in sel.columns if c.endswith("_correct")] + ["cost", "first_cost", "oracle", "n_groups"]].mean()
    for (s, split, n), r in means.iterrows():
        for method in methods_for(dataset):
            rows.append({"dataset": dataset, "sample_id": s, "split": split, "N": n, "method": method,
                         "accuracy": r[f"{method}_correct"], "cost": r["first_cost"] if method == "first" else r["cost"],
                         "oracle": r["oracle"], "n_groups": r["n_groups"]})
    return pd.DataFrame(rows)


def summarize(table):
    #means over questions. failure decomposition (per question, averaged over orders) ->
    #  not_generated = 1 - coverage          no correct candidate among the N
    #  missed        = coverage - accuracy   a correct candidate was there, the rule picked another answer
    out = table.groupby(["dataset", "split", "N", "method"], sort=False)[["accuracy", "cost", "oracle", "n_groups"]].mean().reset_index()
    out["not_generated"] = 1 - out["oracle"]
    out["missed"] = out["oracle"] - out["accuracy"]
    return out


def contrasts(dataset):
    #(a, b) -> accuracy(a) - accuracy(b). every method against self-consistency, then one factor changed at a time
    methods = methods_for(dataset)
    pairs = [(m, "self_consistency") for m in methods if m != "self_consistency"] + [("self_consistency", "first")]
    pairs += [("self_certainty_rwc", "self_certainty"), ("self_certainty", "mean_logp")]
    if dataset == "mathvista":
        pairs += [("vision_rwc", "vision"), ("visconf_rwc", "visconf"), ("visconf_token_rwc", "visconf_token"),
                  ("visconf", "self_certainty"), ("visconf", "vision"), ("visconf_rwc", "self_certainty_rwc"),
                  ("visconf_rwc", "vision_rwc"), ("visconf_token", "visconf"), ("visconf_token_rwc", "visconf_rwc")]
    return pairs


def bootstrap_index(n, cfg):
    return np.random.default_rng(int(cfg["seed"])).integers(n, size=(int(cfg["bootstrap_resamples"]), n))


def paired_bootstrap(a, b, index):
    #a, b -> per-question values (already averaged over orders) on the same questions. 95% percentile interval
    diff = np.asarray(a, float) - np.asarray(b, float)
    means = diff[index].mean(axis=1)
    return {"difference": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def run_bootstrap(table, dataset, cfg):
    rows = []
    for split in ("eval", "dev"):
        t = table[table["split"] == split]
        ids = sorted(t["sample_id"].unique())
        index = bootstrap_index(len(ids), cfg) #one resample matrix shared by every contrast and budget (paired)
        for n in cfg["budgets"][1:]:
            acc = t[t["N"] == n].pivot(index="sample_id", columns="method", values="accuracy").loc[ids]
            for a, b in contrasts(dataset):
                rows.append({"dataset": dataset, "split": split, "N": n, "method": a, "baseline": b,
                             **paired_bootstrap(acc[a], acc[b], index)})
    return pd.DataFrame(rows)


#--------------------------------------------------------------------------------------------- step 3

def quartile_gap(score, correct, rollout_ids):
    #correctness of the top quarter minus the bottom quarter by score within the question (8 of 32). ties in the
    #score are broken by rollout id, so the split is deterministic
    order = np.lexsort((rollout_ids, -score))
    k = len(score) // 4
    return float(correct[order[:k]].mean() - correct[order[-k:]].mean())


def pearson(a, b):
    #within one rollout, over its tokens. undefined (nan) for a single token or a constant series
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else np.nan


def signals(pool, cfg, ids):
    #at N = 32, within each question -> top-minus-bottom quartile correctness for every score, spearman of every score
    #with length, and mean score of correct minus incorrect rollouts. mathvista also gets per-rollout
    #confidence-vision correlations and the Cov_t(c, v) = Z^token - Z gap, with calibration over all 32 rollouts
    names = ["L", "K", "length"] + (["V", "Z", "Z_token"] if pool.dataset == "mathvista" else [])
    rows, cv_rows = [], []
    for s in ids:
        q = pool.row[s]
        idx = list(range(pool.T.shape[1]))
        correct = pool.correct[q].astype(float)
        scores = {"L": pool.scores["L"][q], "K": pool.scores["K"][q], "length": pool.T[q].astype(float)}
        if pool.dataset == "mathvista":
            scores["V"] = pool.scores["V"][q]
            scores["Z"], scores["Z_token"], c = visconf_scores(pool, q, idx, cfg)
            for i in idx:
                k, v = pool.token("kl", q, i), pool.token("v", q, i)
                cv_rows.append({"sample_id": s, "split": pool.split[s], "rollout_id": i, "correct": bool(correct[i]),
                                "n_tokens": int(pool.T[q, i]), "corr_kl_v": pearson(k, v), "corr_c_v": pearson(c[i], v),
                                "cov_c_v": float(scores["Z_token"][i] - scores["Z"][i])})
        mixed = 0 < correct.sum() < len(correct)
        row = {"sample_id": s, "split": pool.split[s], "mixed": bool(mixed), "correct_fraction": float(correct.mean()),
               "mean_length_correct": float(pool.T[q][correct == 1].mean()) if correct.any() else np.nan,
               "mean_length_incorrect": float(pool.T[q][correct == 0].mean()) if (correct == 0).any() else np.nan}
        for name in names:
            sc = scores[name]
            row[f"{name}_quartile_gap"] = quartile_gap(sc, correct, np.arange(len(sc)))
            row[f"{name}_correct_minus_incorrect"] = float(sc[correct == 1].mean() - sc[correct == 0].mean()) if mixed else np.nan
            if name != "length":
                row[f"{name}_spearman_length"] = float(spearmanr(sc, pool.T[q]).statistic) if np.ptp(sc) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cv_rows), names


def summarize_signals(table, cv, names):
    out = {}
    for split in ("eval", "dev"):
        t = table[table["split"] == split]
        mixed = t[t["mixed"]]
        out[split] = {"questions": len(t), "mixed_questions": len(mixed), "scores": {
            name: {"quartile_gap_all": float(t[f"{name}_quartile_gap"].mean()),
                   "quartile_gap_mixed": float(mixed[f"{name}_quartile_gap"].mean()),
                   "correct_minus_incorrect_mixed": float(mixed[f"{name}_correct_minus_incorrect"].mean()),
                   **({"spearman_length": float(t[f"{name}_spearman_length"].mean())} if f"{name}_spearman_length" in t else {})}
            for name in names},
            "length_correct_minus_incorrect_mixed": float((mixed["mean_length_correct"] - mixed["mean_length_incorrect"]).mean())}
        if len(cv):
            c = cv[cv["split"] == split]
            per_q = c.groupby("sample_id")[["corr_kl_v", "corr_c_v", "cov_c_v"]].mean()
            by_correct = c.groupby(["sample_id", "correct"])["corr_kl_v"].mean().unstack()
            out[split]["confidence_vision"] = {
                "per_question_mean_corr_kl_v": per_q["corr_kl_v"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
                "per_question_mean_corr_c_v": per_q["corr_c_v"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
                "per_question_mean_cov_c_v": per_q["cov_c_v"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
                "rollouts_with_undefined_corr": int(c["corr_kl_v"].isna().sum()),
                "corr_kl_v_correct_minus_incorrect_mixed": float((by_correct[True] - by_correct[False]).mean())}
    return out


def trajectories(pool, cfg, splits):
    #fixed illustrative set -> the first trajectory_questions evaluation questions (saved permutation order) whose 32
    #rollouts include both correct and incorrect answers, and in each the lowest-id correct and lowest-id incorrect
    #rollout. token signals averaged within trajectory_bins equal slices of relative position (calibration over all 32)
    bins, picked = int(cfg["trajectory_bins"]), []
    for s in splits[pool.dataset]["eval"]:
        q = pool.row[s]
        if 0 < pool.correct[q].sum() < pool.T.shape[1]:
            picked.append(s)
        if len(picked) == int(cfg["trajectory_questions"]):
            break
    out = []
    for s in picked:
        q = pool.row[s]
        c = visconf_scores(pool, q, list(range(pool.T.shape[1])), cfg)[2] if pool.dataset == "mathvista" else None
        for want in (True, False):
            i = int(np.flatnonzero(pool.correct[q] == want)[0])
            series = {"logp": pool.token("logp", q, i), "kl": pool.token("kl", q, i)}
            if c is not None:
                series.update({"c": c[i], "v": pool.token("v", q, i)})
            edges = np.linspace(0, len(series["kl"]), bins + 1).astype(int)
            binned = {k: [float(v[a:b].mean()) if b > a else None for a, b in zip(edges[:-1], edges[1:])] for k, v in series.items()}
            out.append({"dataset": pool.dataset, "sample_id": s, "rollout_id": i, "correct": want,
                        "answer": pool.key[q, i], "n_tokens": int(pool.T[q, i]), "binned": binned})
    return out


def run(config_path):
    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    if (results / "offline_summary.csv").exists():
        print("[offline] already done")
        return
    elapsed = wall_timer()
    splits = load_json(results / "splits.json")
    summaries, boots, sig_summary, traj = [], [], {}, []
    for dataset in cfg["datasets"]:
        pool = Pool(cfg, dataset)
        ids = selected_ids(pool, cfg, splits)
        orders = make_orders(pool, cfg)
        (results / f"orders_{dataset}.json").write_text(json.dumps(orders)) #compact, ~1 mb
        sel = run_selections(pool, cfg, orders, ids)
        sel.to_parquet(results / f"selections_{dataset}.parquet", index=False)
        table = per_question(sel, dataset)
        table.to_csv(results / f"per_question_{dataset}.csv", index=False)
        summaries.append(summarize(table))
        boots.append(run_bootstrap(table, dataset, cfg))
        print(f"[offline] {dataset} selections done ({elapsed() / 60:.1f} min)")

        sig, cv, names = signals(pool, cfg, ids)
        sig.to_csv(results / f"signals_per_question_{dataset}.csv", index=False)
        if len(cv):
            cv.to_csv(results / "confidence_vision_rollouts.csv", index=False)
        sig_summary[dataset] = summarize_signals(sig, cv, names)
        traj += trajectories(pool, cfg, splits)
        print(f"[offline] {dataset} signals done ({elapsed() / 60:.1f} min)")
    save_json(results / "signals_summary.json", sig_summary)
    save_json(results / "trajectories.json", traj)
    pd.concat(boots).to_csv(results / "offline_bootstrap.csv", index=False)
    save_json(results / "offline_run.json", {"seconds": elapsed(), "question_limit": cfg.get("question_limit")})
    pd.concat(summaries).to_csv(results / "offline_summary.csv", index=False) #written last -> marks the step as done
    print(f"[offline] done in {elapsed() / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
