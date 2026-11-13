from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from common.utils import repo_path

#where every run keeps its states, and how each of the seven evaluation conditions is rebuilt as a plain hf model.
#
#outputs/task1_finetune/
#    <method>/checkpoints/update_0312 ... update_1250/   recovery checkpoints (model or adapter + optimizer + trainer state)
#    <method>/bases.safetensors                          oplora only, the frozen U_k / V_k, computed once
#    <method>/exported_factors.safetensors               adapter runs, the final effective factors used for inference
#    eval_models/<condition>/                            fp16 merged copy of each condition, what lm-eval loads


def run_dir(cfg, method):
    return repo_path(cfg["outputs_dir"]) / method


def checkpoint_dir(cfg, method, update):
    return run_dir(cfg, method) / "checkpoints" / f"update_{update:04d}"


def eval_model_dir(cfg, condition):
    return repo_path(cfg["outputs_dir"]) / "eval_models" / condition


def load_base(cfg, dtype):
    #the unchanged post-trained checkpoint at the pinned revision
    return AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["model_revision"], dtype=dtype)


def exported_deltas(cfg, method):
    #dense fp32 corrections dW = scale * B A per adapted module, from the exported (projected, for oplora) factors
    factors = load_file(str(run_dir(cfg, method) / "exported_factors.safetensors"))
    scale = float(factors.pop("scale"))
    names = sorted({key.rsplit(".", 1)[0] for key in factors})
    return {name: scale * (factors[f"{name}.B"].float() @ factors[f"{name}.A"].float()) for name in names}


@torch.no_grad()
def build_condition_model(cfg, condition, dtype=torch.float16):
    #full ft conditions -> the saved fp32 master weights of that checkpoint, cast to dtype on load.
    #adapter conditions -> merged into a fresh fp32 copy of the starting model (W0 + dW), then cast to dtype once
    spec = cfg["conditions"][condition]
    if spec["method"] is None:
        return load_base(cfg, dtype)
    kind = cfg["methods"][spec["method"]]["kind"]
    if kind == "full":
        return AutoModelForCausalLM.from_pretrained(checkpoint_dir(cfg, spec["method"], spec["update"]) / "model", dtype=dtype)
    model = load_base(cfg, torch.float32)
    for name, delta in exported_deltas(cfg, spec["method"]).items():
        model.get_submodule(name).weight.add_(delta)
    return model.to(dtype)


def condition_delta_weight(cfg, condition, module_name, base_weight):
    #dW of one module for the diagnostics, in fp32 -> trained minus starting weight for full ft, scale * B A for adapters
    spec = cfg["conditions"][condition]
    kind = cfg["methods"][spec["method"]]["kind"]
    #safe_open reads just the tensors asked for, not the whole 2.4 gb fp32 checkpoint (0.6b params fit one shard)
    if kind == "full":
        with safe_open(str(checkpoint_dir(cfg, spec["method"], spec["update"]) / "model" / "model.safetensors"), "pt") as f:
            return f.get_tensor(f"{module_name}.weight").float() - base_weight.float()
    with safe_open(str(run_dir(cfg, spec["method"]) / "exported_factors.safetensors"), "pt") as f:
        return float(f.get_tensor("scale")) * (f.get_tensor(f"{module_name}.B").float() @ f.get_tensor(f"{module_name}.A").float())
