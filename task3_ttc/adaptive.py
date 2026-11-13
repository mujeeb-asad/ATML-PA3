from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common.utils import load_yaml, repo_path, save_json, wall_timer
from task3_ttc.offline import bootstrap_index, paired_bootstrap

#step 5 -> adaptive sampling, simulated on the supplied pools (cpu only). rollouts are revealed in the saved orders
#(offline.py), stopping is checked after n in {4, 8, 16} candidates, otherwise the controller stops at 32. both
#controllers return the current self-consistency answer, so they differ only in when they stop ->
#
#  agreement      stop when the leading answer is unique, is not NO_ANSWER, and has at least tau * n of the n votes
#  quality-aware  the same condition, plus the self-consistency and rwc winners agree and the rwc margin is at least
#                 gamma. rwc uses self-certainty for math and visconf for mathvista. margin = (largest answer vote -
#                 second largest) / total vote weight, 0 for the runner-up when there is one group. scores, the visconf
#                 calibration and the ranks use only the revealed candidates (they are the prefix values offline.py
#                 already computed for budgets 4, 8, 16)
#
#setting selection, per dataset and controller family, on the development split only -> among every setting plus the
#policy that always consumes 32, the lowest mean token cost whose accuracy is at most adaptive_tolerance below
#fixed-32 self-consistency (all averaged over the ten orders), ties -> higher accuracy, then smaller tau, then smaller
#gamma. the chosen settings are saved before the evaluation split is touched.
#
#results/task3_ttc/
#    adaptive_dev_grid.csv          every setting on the development split
#    adaptive_selected.json         the frozen settings (written before held-out evaluation)
#    adaptive_eval.csv              held-out accuracy, cost, rollouts, stopping distribution, next to fixed-N self-consistency
#    adaptive_per_question.csv      per (question, policy) on the evaluation split, averaged over orders
#    adaptive_early_stops.csv       every incorrect early stop on the evaluation split, flagged when fixed-32 recovers it
#    adaptive_bootstrap.csv         paired bootstrap of adaptive minus fixed self-consistency accuracy


def load_checkpoints(cfg, dataset):
    #per (question, order) -> the self-consistency state at every check and at 32 as arrays (questions x orders x 4)
    sel = pd.read_parquet(repo_path(cfg["results_dir"]) / f"selections_{dataset}.parquet")
    checks = [*cfg["adaptive_checks"], int(cfg["rollouts_per_question"])]
    sel = sel[sel["N"].isin(checks)].sort_values(["sample_id", "order", "N"])
    quality = f"{cfg['adaptive_quality_score'][dataset]}_rwc"
    ids = sel["sample_id"].unique().tolist()
    shape = (len(ids), int(cfg["orders"]), len(checks))
    arr = lambda c: sel[c].to_numpy().reshape(shape)
    state = {"n": np.array(checks), "votes": arr("sc_votes"), "runner_up": arr("sc_runner_up"),
             "no_answer": arr("sc_no_answer"), "agree": arr("self_consistency") == arr(quality),
             "margin": arr(f"{quality}_margin"), "correct": arr("self_consistency_correct"), "cost": arr("cost"),
             "answer": arr("self_consistency")}
    split = sel.groupby("sample_id", sort=False)["split"].first()
    return ids, split.to_numpy(), state


def simulate(state, tau, gamma):
    #stop index per (question, order) -> the first check whose condition holds, else the last (32). gamma None ->
    #the agreement controller. tau None -> always 32 (the fallback policy)
    n_checks = len(state["n"]) - 1
    if tau is None:
        stop = np.full(state["votes"].shape[:2], n_checks)
    else:
        ok = (state["votes"] > state["runner_up"]) & ~state["no_answer"] & (state["votes"] >= tau * state["n"])
        if gamma is not None:
            ok &= state["agree"] & (state["margin"] >= gamma)
        ok = ok[..., :n_checks]
        stop = np.where(ok.any(-1), ok.argmax(-1), n_checks)
    pick = lambda key: np.take_along_axis(state[key], stop[..., None], -1)[..., 0]
    return {"stop": stop, "n": state["n"][stop], "correct": pick("correct"), "cost": pick("cost")}


def policy_table(state, mask, settings):
    #mean over orders within a question, then over the questions in mask
    rows = []
    for family, tau, gamma in settings:
        out = simulate(state, tau, gamma)
        rows.append({"family": family, "tau": tau, "gamma": gamma,
                     "accuracy": out["correct"][mask].mean(-1).mean(), "cost": out["cost"][mask].mean(-1).mean(),
                     "rollouts": out["n"][mask].mean(-1).mean()})
    return pd.DataFrame(rows)


def choose(grid, family, reference, tolerance):
    #lowest cost within tolerance of fixed-32 self-consistency, ties -> higher accuracy, smaller tau, smaller gamma.
    #the fallback (always 32) is a candidate of every family and always qualifies
    cand = grid[(grid["family"].isin([family, "fixed_32"])) & (grid["accuracy"] >= reference - tolerance - 1e-12)].copy()
    cand["tau_key"], cand["gamma_key"] = cand["tau"].fillna(np.inf), cand["gamma"].fillna(np.inf)
    best = cand.sort_values(["cost", "accuracy", "tau_key", "gamma_key"], ascending=[True, False, True, True]).iloc[0]
    return {"family": family, "chosen": best["family"], "tau": None if pd.isna(best["tau"]) else float(best["tau"]),
            "gamma": None if pd.isna(best["gamma"]) else float(best["gamma"]), "dev_accuracy": float(best["accuracy"]),
            "dev_cost": float(best["cost"]), "dev_reference_accuracy": float(reference)}


def run(config_path):
    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    if (results / "adaptive_eval.csv").exists():
        print("[adaptive] already done")
        return
    elapsed = wall_timer()
    taus, gammas = [float(t) for t in cfg["adaptive_taus"]], [float(g) for g in cfg["adaptive_gammas"]]
    settings = ([("fixed_32", None, None)] + [("agreement", t, None) for t in taus]
                + [("quality", t, g) for t, g in itertools.product(taus, gammas)])
    grids, selected, evals, per_q, early, boots = [], {}, [], [], [], []
    sel_cache = {}
    for dataset in cfg["datasets"]:
        ids, split, state = load_checkpoints(cfg, dataset)
        sel_cache[dataset] = (ids, split, state)
        dev = split == "dev"
        grid = policy_table(state, dev, settings)
        grid.insert(0, "dataset", dataset)
        grids.append(grid)
        reference = float(grid[grid["family"] == "fixed_32"]["accuracy"].iloc[0])
        selected[dataset] = {f: choose(grid, f, reference, float(cfg["adaptive_tolerance"])) for f in ("agreement", "quality")}
    pd.concat(grids).to_csv(results / "adaptive_dev_grid.csv", index=False)
    save_json(results / "adaptive_selected.json", selected) #frozen before anything on the evaluation split is computed
    print(f"[adaptive] selected on development -> {selected}")

    for dataset, (ids, split, state) in sel_cache.items():
        mask = split == "eval"
        ev_ids = [s for s, m in zip(ids, mask) if m]
        index = bootstrap_index(len(ev_ids), cfg)
        sel = pd.read_parquet(results / f"selections_{dataset}.parquet")
        sel = sel[sel["split"] == "eval"]
        fixed = sel.groupby(["sample_id", "N"])[["self_consistency_correct", "cost"]].mean()
        for n in cfg["budgets"]: #fixed-N self-consistency next to the adaptive policies
            f = fixed.xs(n, level="N").loc[ev_ids]
            evals.append({"dataset": dataset, "policy": f"fixed_{n}", "tau": None, "gamma": None,
                          "accuracy": f["self_consistency_correct"].mean(), "cost": f["cost"].mean(), "rollouts": n,
                          #a fixed budget has a stopping distribution only when it is one of the checkpoints
                          **{f"stop_at_{c}": float(c == n) if n in state["n"] else np.nan for c in state["n"]}})
        full32 = simulate(state, None, None)
        for family, choice in selected[dataset].items():
            out = simulate(state, choice["tau"], None if family == "agreement" else choice["gamma"]) #tau None -> fixed 32
            stops = state["n"][out["stop"][mask]]
            evals.append({"dataset": dataset, "policy": family, "tau": choice["tau"], "gamma": choice["gamma"],
                          "accuracy": out["correct"][mask].mean(-1).mean(), "cost": out["cost"][mask].mean(-1).mean(),
                          "rollouts": out["n"][mask].mean(-1).mean(),
                          **{f"stop_at_{c}": float((stops == c).mean()) for c in state["n"]}})
            acc_q = out["correct"][mask].mean(-1)
            for s, a, c, r in zip(ev_ids, acc_q, out["cost"][mask].mean(-1), out["n"][mask].mean(-1)):
                per_q.append({"dataset": dataset, "policy": family, "sample_id": s, "accuracy": a, "cost": c, "rollouts": r})
            for n in cfg["budgets"]:
                f = fixed.xs(n, level="N").loc[ev_ids]["self_consistency_correct"].to_numpy()
                boots.append({"dataset": dataset, "policy": family, "baseline": f"fixed_{n}", **paired_bootstrap(acc_q, f, index)})
            #incorrect early stops -> stopped before 32 with a wrong answer. recovered = fixed-32 is right on that order
            qi, oi = np.nonzero((out["n"] < state["n"][-1]) & ~out["correct"] & mask[:, None])
            for q, o in zip(qi, oi):
                early.append({"dataset": dataset, "policy": family, "sample_id": ids[q], "order": int(o),
                              "stop_n": int(out["n"][q, o]), "stopped_answer_group": int(state["answer"][q, o, out["stop"][q, o]]),
                              "fixed_32_answer_group": int(state["answer"][q, o, -1]),
                              "recovered_by_fixed_32": bool(full32["correct"][q, o])})
    save_json(results / "adaptive_run.json", {"seconds": elapsed()})
    pd.DataFrame(per_q).to_csv(results / "adaptive_per_question.csv", index=False)
    pd.DataFrame(early, columns=["dataset", "policy", "sample_id", "order", "stop_n", "stopped_answer_group",
                                 "fixed_32_answer_group", "recovered_by_fixed_32"]).to_csv(results / "adaptive_early_stops.csv", index=False)
    pd.DataFrame(boots).to_csv(results / "adaptive_bootstrap.csv", index=False)
    pd.DataFrame(evals).to_csv(results / "adaptive_eval.csv", index=False) #written last -> marks the step as done
    print(f"[adaptive] done in {elapsed():.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
