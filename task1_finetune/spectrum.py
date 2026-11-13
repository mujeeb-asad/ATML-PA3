from __future__ import annotations

import argparse
import platform
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from common.utils import load_yaml, repo_path, save_json, wall_timer

#supplementary measurement, not one the manual asks for. how much of each starting weight the oplora protection
#rank covers, from the unchanged base checkpoint only (no trained model, no gpu, nothing else is rerun) ->
#
#  W0 = U S V^T, singular values s_1 >= s_2 >= ... >= s_n, n = min(d_out, d_in)
#  energy kept by the top k = (s_1^2 + ... + s_k^2) / (s_1^2 + ... + s_n^2)
#
#the s_i^2 are the eigenvalues of W0 W0^T, and their sum is ||W0||_F^2, so this is the share of the matrix's squared
#frobenius norm that lies in the k protected directions (the same U_k / V_k pair spans it on both sides).
#every module oplora adapted is measured (7 projection types x 28 layers = 196), with k = protect_k = 128. the
#weights are rounded to fp16 first and then decomposed in fp32, exactly as compute_bases saw them (the frozen fp16
#backbone, .float(), one svd). the diagnostics' rho_k used the fp32 checkpoint values, the same matrices to fp16
#rounding.
#
#results/task1_finetune/
#    oplora_spectrum.csv           one row per adapted module -> shape, energy kept by the top k, singular values
#    oplora_spectrum_curves.json   cumulative energy at every k for the diagnostic modules (down_proj 0, 13, 27)
#    oplora_spectrum_summary.json  per projection type over the 28 layers, plus settings and library versions


def checkpoint_file(cfg):
    #the base model's single safetensors file at the pinned revision (from the local hugging face cache if present)
    return hf_hub_download(cfg["base_model"], "model.safetensors", revision=cfg["model_revision"])


def run(config_path):
    cfg = load_yaml(config_path)
    results = repo_path(cfg["results_dir"])
    k = int(cfg["methods"]["oplora16"]["protect_k"])
    targets = cfg["adapter_targets"]
    diagnostic = {f"model.layers.{layer}.mlp.{cfg['diagnostic_module']}" for layer in cfg["diagnostic_layers"]}
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight$")
    elapsed = wall_timer()

    rows, curves = [], {}
    with safe_open(checkpoint_file(cfg), "pt") as f:
        names = [n for n in f.keys() if (m := pattern.match(n)) and m.group(3) in targets]
        for name in sorted(names, key=lambda n: (int(pattern.match(n).group(1)), targets.index(pattern.match(n).group(3)))):
            layer, module = int(pattern.match(name).group(1)), pattern.match(name).group(3)
            W = f.get_tensor(name).to(torch.float16).float() #one matrix at a time, so memory stays small
            s = torch.linalg.svdvals(W).double().numpy() #descending
            energy = np.cumsum(s ** 2) / np.sum(s ** 2)
            d_out, d_in = W.shape
            rows.append({"module": name[:-len(".weight")], "layer": layer, "type": module, "d_out": d_out, "d_in": d_in,
                         "n_singular": len(s), "k": k, "k_share_of_out_dims": k / d_out, "k_share_of_in_dims": k / d_in,
                         "k_share_of_rank": k / len(s), "energy_top_k": float(energy[k - 1]),
                         "sigma_1": float(s[0]), "sigma_k": float(s[k - 1]), "sigma_k_plus_1": float(s[k]),
                         "sigma_min": float(s[-1]), "frobenius_sq": float(np.sum(s ** 2))})
            if name[:-len(".weight")] in diagnostic:
                curves[name[:-len(".weight")]] = energy.tolist()

    table = pd.DataFrame(rows)
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "oplora_spectrum.csv", index=False)
    save_json(results / "oplora_spectrum_curves.json", curves)
    by_type = table.groupby("type", sort=False)["energy_top_k"].describe()[["mean", "min", "max"]]
    save_json(results / "oplora_spectrum_summary.json", {
        "k": k, "modules": len(table), "by_type_over_layers": by_type.to_dict(orient="index"),
        "diagnostic_modules": table[table["module"].isin(diagnostic)].set_index("module")["energy_top_k"].to_dict(),
        "weights": "base checkpoint rounded to fp16, decomposed in fp32 (torch.linalg.svdvals)",
        "model": cfg["base_model"], "revision": cfg["model_revision"], "seconds": elapsed(),
        "versions": {"torch": torch.__version__, "python": platform.python_version()}})
    print(f"[spectrum] {len(table)} modules, k = {k}, done in {elapsed():.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
