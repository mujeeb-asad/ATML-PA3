from __future__ import annotations

import argparse
import math

import pandas as pd
import torch
from safetensors import safe_open

from common.utils import load_json, load_yaml, repo_path, save_json
from task1_finetune.data import chat_prompt_ids, load_tokenizer
from task1_finetune.models import checkpoint_dir, condition_delta_weight, load_base

#weight-space and activation diagnostics of the local down_proj corrections, all in fp32 ->
#  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   share of the update's energy inside W0's top-k output subspace
#  eta   = ||dW||_F / ||W0||_F                  size of the update relative to the starting weight
#  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)     how strongly the update responds to a real input z, relative to W0
#z is the starting model's down_proj input at the last prompt token of each probe prompt, cached once and reused
#for every checkpoint, so differences between checkpoints come from the local dW alone. plus the full-model
#displacement ||theta - theta0||_2 for the full ft checkpoints


def module_name(cfg, layer):
    return f"model.layers.{layer}.mlp.{cfg['diagnostic_module']}"


def probe_prompts(cfg, tokenizer):
    #the benchmark prompt formats -> gsm8k as lm-eval's zero-shot gsm8k_cot chat prompt ("Q: ...\nA:" in the user
    #turn, opened assistant turn with the empty think block), hellaswag as its plain completion context. the prompt
    #ends before any answer continuation, so the last token is where the model starts producing the answer
    probe = load_json(repo_path(cfg["results_dir"]) / "probe_set.json")
    prompts = [("math", r["doc_id"], chat_prompt_ids(tokenizer, f"Q: {r['question']}\nA:")) for r in probe["gsm8k"]]
    prompts += [("non_math", r["doc_id"], tokenizer(r["query"], add_special_tokens=False)["input_ids"]) for r in probe["hellaswag"]]
    return prompts


@torch.no_grad()
def cache_inputs(model, cfg, prompts, device):
    #forward hooks on the starting model's down_proj -> its input at the last prompt position. prompts run one at a
    #time (no padding), so the last position is the last non-padding prompt token
    layers, cache, current = cfg["diagnostic_layers"], {}, {}
    hooks = [model.get_submodule(module_name(cfg, layer)).register_forward_hook(
        lambda module, inputs, output, layer=layer: current.__setitem__(layer, inputs[0][0, -1].float().cpu()))
        for layer in layers]
    for domain, doc_id, ids in prompts:
        model(input_ids=torch.tensor([ids], device=device))
        for layer in layers:
            cache.setdefault(layer, []).append(current[layer])
    for h in hooks:
        h.remove()
    return {layer: torch.stack(z) for layer, z in cache.items()} #layer -> [n_prompts, d_in]


@torch.no_grad()
def displacement(cfg, condition, base_params):
    #||theta - theta0||_2 over every unique parameter (tied embedding once), streamed tensor by tensor from the
    #saved fp32 master weights so the full model is never duplicated in memory
    spec = cfg["conditions"][condition]
    total = 0.0
    with safe_open(str(checkpoint_dir(cfg, spec["method"], spec["update"]) / "model" / "model.safetensors"), "pt") as f:
        for name, p0 in base_params.items():
            total += float((f.get_tensor(name).float() - p0).pow(2).sum(dtype=torch.float64))
    return math.sqrt(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = int(cfg["diagnostic_protect_k"])

    tokenizer = load_tokenizer(cfg)
    model = load_base(cfg, torch.float32).to(device).eval()
    prompts = probe_prompts(cfg, tokenizer)
    Z = cache_inputs(model, cfg, prompts, device)

    #the starting weights and their top-k left singular vectors, the same bases for every checkpoint
    W0, U_k = {}, {}
    for layer in cfg["diagnostic_layers"]:
        W0[layer] = model.get_submodule(module_name(cfg, layer)).weight.detach().float().cpu()
        U, _, _ = torch.linalg.svd(W0[layer], full_matrices=False)
        U_k[layer] = U[:, :k]
    base_params = {name: p.detach().float().cpu() for name, p in model.named_parameters()}
    del model

    weight_rows, act_rows = [], []
    for condition in cfg["diagnostic_conditions"]:
        for layer in cfg["diagnostic_layers"]:
            dW = condition_delta_weight(cfg, condition, module_name(cfg, layer), W0[layer])
            dW_sq = float(dW.pow(2).sum())
            #U_k has orthonormal columns -> ||U_k U_k^T dW||_F = ||U_k^T dW||_F, no d_out x d_out projector needed
            in_subspace = float((U_k[layer].T @ dW).pow(2).sum())
            weight_rows.append({"condition": condition, "layer": layer,
                                "rho_k": in_subspace / dW_sq if dW_sq > 0 else float("nan"), #undefined for a zero update
                                "eta": math.sqrt(dW_sq) / float(W0[layer].norm()),
                                "dW_fro": math.sqrt(dW_sq), "W0_fro": float(W0[layer].norm())})
            z = Z[layer] #[n_prompts, d_in]
            ratio = (z @ dW.T).norm(dim=1) / ((z @ W0[layer].T).norm(dim=1) + 1e-8)
            for (domain, doc_id, _), r in zip(prompts, ratio.tolist()):
                act_rows.append({"condition": condition, "layer": layer, "domain": domain, "doc_id": doc_id, "R": r})
        print(f"[diagnostics] {condition} done")

    weights = pd.DataFrame(weight_rows)
    acts = pd.DataFrame(act_rows)
    weights.to_csv(results / "diagnostics_weight_space.csv", index=False)
    acts.to_csv(results / "diagnostics_activation_per_prompt.csv", index=False)
    summary = acts.groupby(["condition", "layer", "domain"])["R"].agg(["mean", "median", "std"]).reset_index()
    summary.to_csv(results / "diagnostics_activation_summary.csv", index=False)

    disp = {c: displacement(cfg, c, base_params) for c in cfg["diagnostic_conditions"]
            if cfg["conditions"][c]["method"] and cfg["methods"][cfg["conditions"][c]["method"]]["kind"] == "full"}
    save_json(results / "displacement.json", disp)
    print(weights.round(5).to_string(index=False))
    print(summary.round(5).to_string(index=False))
    print("full-model displacement", disp)


if __name__ == "__main__":
    main()
