from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#the report's figures, drawn only from saved results (results/task*/ and results/report/ from report_stats.py).
#nothing is rerun. one pdf per figure in results/report/, sized for the neurips text width (5.5 in)
#
#colour -> one fixed hue per entity (method, image condition, selection rule), in the same order everywhere, so a
#colour means the same thing in every figure. line style is a second encoding, so no line relies on colour alone

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = RES / "report"
read_jsonl = lambda p: [json.loads(line) for line in Path(p).open(encoding="utf-8") if line.strip()]

#categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red), recessive ink for text
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 5.5

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

METHODS = {"full_ft": ("Full FT", SLOT[0], "-"), "l2sp": ("Full FT + L2-SP", SLOT[1], "--"),
           "lora4": ("LoRA-4", SLOT[2], "-"), "lora16": ("LoRA-16", SLOT[3], "-"), "oplora16": ("OPLoRA-16", SLOT[4], "--")}
CONDITIONS = {"correct": ("correct", SLOT[0], "o"), "complementary": ("complementary swap", SLOT[1], "s"),
              "mismatched": ("mismatched", SLOT[2], "^"), "none": ("no image", SLOT[3], "D")}


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=8)


def fig_task1():
    #(a) sft loss for the five runs, rolling mean over 50 updates. (b) l2-sp's distance from the start over training,
    #with full ft's two measured points. (c) the gain / retention plane -> gsm8k change (stock score, and with the
    #base model's extraction failures removed) against the hellaswag change, one point per evaluated condition
    t1 = RES / "task1_finetune"
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.85), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
    ax = axes[0]
    for m, (name, color, style) in METHODS.items():
        log = pd.DataFrame(read_jsonl(t1 / f"train_log_{m}.jsonl"))
        ax.plot(log["update"], log["sft_loss"].rolling(50, min_periods=10).mean(), color=color, ls=style, lw=1.2, label=name)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (50-update mean)")
    ax.set_ylim(0.17, 0.36)
    ax.legend(loc="upper right", handlelength=2.2)
    panel_label(ax, "(a) training loss")

    ax = axes[1]
    log = pd.DataFrame(read_jsonl(t1 / "train_log_l2sp.jsonl"))
    disp = json.loads((t1 / "displacement.json").read_text())
    ax.plot(log["update"], log["displacement_l2"], color=METHODS["l2sp"][1], ls="--", label="L2-SP")
    ax.plot([625, 1250], [disp["full_ft_mid"], disp["full_ft"]], color=METHODS["full_ft"][1], marker="o", ms=4,
            label="Full FT")
    ax.set_xlabel("optimizer update")
    ax.set_ylabel(r"$\|\theta-\theta_0\|_2$")
    ax.set_ylim(0, 11.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), handlelength=1.6)
    panel_label(ax, "(b) distance from start")

    ax = axes[2]
    bench = pd.read_csv(t1 / "benchmark_table.csv").set_index("condition")
    adj = pd.read_csv(OUT / "t1_gsm8k_adjusted.csv").set_index("condition")
    points = {"full_ft_mid": ("Full FT (mid)", SLOT[0], 0.45), **{k: (v[0], v[1], 1.0) for k, v in METHODS.items()}}
    for c, (name, color, alpha) in points.items():
        y = bench.loc[c, "hellaswag_delta_pp"]
        x_stock, x_adj = bench.loc[c, "gsm8k_delta_pp"], adj.loc[c, "adjusted_delta_pp"]
        ax.plot([x_adj, x_stock], [y, y], color=color, lw=0.8, alpha=alpha, zorder=1)
        ax.scatter([x_stock], [y], s=22, color=color, alpha=alpha, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([x_adj], [y], s=22, facecolor="white", edgecolor=color, linewidth=1.1, alpha=alpha, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.annotate("Full FT mid", (bench.loc["full_ft_mid", "gsm8k_delta_pp"], bench.loc["full_ft_mid", "hellaswag_delta_pp"]),
                xytext=(-2, -9), textcoords="offset points", ha="right", fontsize=6, color=INK2)
    ax.set_xlabel("GSM8K change vs base (pp)")
    ax.set_ylabel("HellaSwag change (pp)")
    ax.set_xlim(-1.5, 23.5)
    ax.set_ylim(-2.2, 0.5)
    panel_label(ax, "(c) gain against retention")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_task1.pdf")
    plt.close(fig)


def fig_task2():
    #(a) training loss and development caption loss with the stage boundaries. (b) the image-swap penalty on both
    #development suites. (c, d) reference-answer nll under the correct image, the complementary image and no image
    t2 = RES / "task2_vlm"
    log = pd.DataFrame(read_jsonl(t2 / "train_log.jsonl"))
    dev = pd.read_csv(t2 / "dev_metrics.csv")
    ends = [2250, 3750]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 2.75))
    axes = axes.ravel()

    def stages(ax, labels=False):
        for e in ends:
            ax.axvline(e, color=INK2, lw=0.6, ls=":")
        if labels:
            for x, t in [(1125, "caption"), (3000, "instruct"), (4125, "VQA")]:
                ax.text(x, 0.97, t, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=INK2,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

    ax = axes[0]
    ax.plot(log["update"], log["loss"].rolling(50, min_periods=10).mean(), color=INK2, lw=0.9, label="training (50-update mean)")
    ax.plot(dev["update"], dev["caption_loss"], color=SLOT[0], marker="o", ms=3, label="dev caption loss")
    stages(ax, True)
    ax.set_ylim(0, 6.8)
    ax.set_xlabel("update")
    ax.set_ylabel("NLL per token (nats)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    panel_label(ax, "(a) training and caption loss")

    ax = axes[1]
    ax.plot(dev["update"], dev["grounding_dev_swap_penalty"], color=SLOT[6], marker="o", ms=3, label="grounding dev")
    ax.plot(dev["update"], dev["vqa_dev_swap_penalty"], color=SLOT[5], marker="s", ms=3, ls="--", label="VQA dev")
    ax.axhline(0, color=INK2, lw=0.6)
    stages(ax, True)
    ax.set_xlabel("update")
    ax.set_ylabel(r"$\Delta$NLL (nats)")
    ax.set_ylim(-0.4, 5.0)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    panel_label(ax, "(b) image-swap penalty (complementary minus correct)")

    for ax, suite, title in [(axes[2], "grounding_dev", "(c) grounding dev NLL"), (axes[3], "vqa_dev", "(d) VQA dev NLL")]:
        for cond in ["correct", "complementary", "none"]:
            name, color, marker = CONDITIONS[cond]
            ax.plot(dev["update"], dev[f"{suite}_nll_{cond}"], color=color, marker=marker, ms=3, label=name)
        stages(ax, True)
        ax.set_ylim(0, 15.5)
        ax.set_xlabel("update")
        panel_label(ax, title)
    axes[2].set_ylabel("answer NLL (nats)")
    handles, labels = axes[3].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, h_pad=0.6, rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "fig_task2_curves.pdf")
    plt.close(fig)

    #accuracy per question type under the three image conditions that produce short answers, 95% pair-bootstrap
    #intervals. filled -> 196 visual tokens, hollow -> the optional 49-token run
    summary = {run: json.loads((RES / folder / "eval_final.json").read_text())["summary"]
               for run, folder in [("196", "task2_vlm"), ("49", "task2_vlm_49")]}
    types = [("grounding_test", "presence", "grounding\npresence"), ("grounding_test", "counting", "grounding\ncounting"),
             ("vqa_test", "yes/no", "VQA\nyes/no"), ("vqa_test", "number", "VQA\nnumber"), ("vqa_test", "other", "VQA\nother")]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.05))
    offsets = {"correct": -0.22, "complementary": 0.0, "mismatched": 0.22}
    for i, (suite, t, label) in enumerate(types):
        for cond, off in offsets.items():
            name, color, marker = CONDITIONS[cond]
            for run, shift, filled in [("196", -0.05, True), ("49", 0.05, False)]:
                a = summary[run][suite]["types"][t][cond]["accuracy"]
                x = i + off + shift
                ax.plot([x, x], [100 * a["ci_low"], 100 * a["ci_high"]], color=color, lw=0.8, alpha=0.8 if filled else 0.5)
                ax.scatter([x], [100 * a["mean"]], s=14, marker=marker, color=color if filled else "white",
                           edgecolor=color, linewidth=0.9, zorder=3, label=name if (i == 0 and run == "196") else None)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t[2] for t in types])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 102)
    ax.grid(axis="x", visible=False)
    ax.scatter([], [], s=14, color=INK2, label="196 tokens")
    ax.scatter([], [], s=14, facecolor="white", edgecolor=INK2, label="49 tokens")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_task2_types.pdf")
    plt.close(fig)


def fig_task3():
    #accuracy against the candidate budget n on the 800 evaluation questions, with oracle coverage as the ceiling
    t3 = RES / "task3_ttc"
    s = pd.read_csv(t3 / "offline_summary.csv")
    s = s[s["split"] == "eval"]
    rules = {"math": [("self_consistency", "self-consistency", SLOT[0], "-", "o"),
                      ("self_certainty_rwc", "self-certainty + RWC", SLOT[1], "-", "s"),
                      ("self_certainty", "highest self-certainty", SLOT[1], "--", "s"),
                      ("mean_logp", "highest mean log-prob", SLOT[2], "--", "^")],
             "mathvista": [("self_consistency", "self-consistency", SLOT[0], "-", "o"),
                           ("visconf_rwc", "VisConf + RWC", SLOT[6], "-", "D"),
                           ("visconf", "highest VisConf", SLOT[6], "--", "D"),
                           ("self_certainty", "highest self-certainty", SLOT[1], "--", "s"),
                           ("mean_logp", "highest mean log-prob", SLOT[2], "--", "^")]}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharey=False)
    for ax, (d, title) in zip(axes, [("math", "(a) MATH, Qwen2.5-3B"), ("mathvista", "(b) MathVista, Qwen2.5-VL-3B")]):
        g = s[s["dataset"] == d]
        first = g[g["method"] == "first"].sort_values("N")
        sc = g[g["method"] == "self_consistency"].sort_values("N")
        ax.plot(sc["N"], sc["oracle"], color=INK2, ls=(0, (4, 2)), marker="x", ms=3.5, label="oracle coverage")
        ax.fill_between(sc["N"], sc["accuracy"], sc["oracle"], color=GRID, alpha=0.55, lw=0)
        ax.plot(first["N"], first["accuracy"], color=INK2, ls=":", lw=1.0, label="first answer")
        for m, name, color, style, marker in rules[d]:
            r = g[g["method"] == m].sort_values("N")
            ax.plot(r["N"], r["accuracy"], color=color, ls=style, marker=marker, ms=3, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("candidates N")
        ax.set_ylim(0.45 if d == "mathvista" else 0.6, 0.94)
        panel_label(ax, title)
    axes[0].set_ylabel("accuracy (eval split)")
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    fig.savefig(OUT / "fig_task3_scaling.pdf")
    plt.close(fig)

    #adaptive stopping against fixed-n self-consistency on the accuracy / mean generated-token plane
    ev = pd.read_csv(t3 / "adaptive_eval.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.62, 1.95))
    for ax, (d, title) in zip(axes, [("math", "(a) MATH"), ("mathvista", "(b) MathVista")]):
        g = ev[ev["dataset"] == d]
        fixed = g[g["policy"].str.startswith("fixed")].sort_values("cost")
        ax.plot(fixed["cost"], fixed["accuracy"], color=SLOT[0], marker="o", ms=3, label="fixed N, self-consistency")
        for _, r in fixed[fixed["policy"].isin(["fixed_4", "fixed_8", "fixed_16", "fixed_32"])].iterrows():
            ax.annotate(r["policy"].split("_")[1], (r["cost"], r["accuracy"]), xytext=(2, -7), textcoords="offset points",
                        fontsize=5.5, color=INK2)
        for pol, color, marker, name in [("agreement", SLOT[1], "s", "agreement stop"), ("quality", SLOT[6], "D", "quality-aware stop")]:
            r = g[g["policy"] == pol].iloc[0]
            ax.scatter([r["cost"]], [r["accuracy"]], s=20, marker=marker, color=color, edgecolor="white", linewidth=0.6,
                       zorder=3, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("mean generated tokens")
        panel_label(ax, title)
    axes[0].set_ylabel("accuracy (eval split)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=0.8, fontsize=5.8)
    fig.savefig(OUT / "fig_task3_adaptive.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_task1()
    fig_task2()
    fig_task3()
    print("[report] figures written to", OUT)


if __name__ == "__main__":
    main()
