from __future__ import annotations

import argparse
import math
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from common.utils import (append_jsonl, gpu_name, load_json, load_yaml, peak_memory_gb, read_jsonl, repo_path,
                          save_json, set_seed, size_bytes, wall_timer)
from task1_finetune.adapters import (adapter_state, compute_bases, export_factors, inject_adapters, load_adapter_state,
                                     projection_residuals)
from task1_finetune.data import IGNORE, collate, encode_example, load_tokenizer, supervised_tokens
from task1_finetune.models import checkpoint_dir, load_base, run_dir

#one script for all five variants (--method full_ft | l2sp | lora4 | lora16 | oplora16). everything except the method
#block of the config is shared -> same 40,000 examples in the same order, same tokenization, same batches, same
#optimizer, schedule, clipping and precision recipe. each run starts independently from the original checkpoint
#
#objective per optimizer update ->  L_SFT(theta) + lambda * Omega(theta, theta0)
#  L_SFT  = summed next-token cross entropy over the response + <|im_end|> tokens of the 32 examples, divided by the
#           number of those tokens in the whole effective batch (not a mean of 16 microbatch means)
#  Omega  = 1/2 sum_j (theta_j - theta0_j)^2 over every unique parameter (l2-sp only, lambda = 0 for the others)


def build_model(cfg, method, device):
    #full ft -> every original parameter trains, held in fp32 (the master copy). fp16 only appears inside autocast.
    #adapter runs -> frozen fp16 backbone, fp32 A/B factors injected into every target projection of every layer
    spec = cfg["methods"][method]
    bundle = {"adapters": None, "bases_s": None, "theta0": None}
    if spec["kind"] == "full":
        model = load_base(cfg, torch.float32).to(device)
    else:
        model = load_base(cfg, torch.float16).to(device)
        #reseeded right before injection -> lora-16 and oplora-16 get the very same initial (unprojected) A factors
        torch.manual_seed(int(cfg["seed"]))
        protect_k = spec.get("protect_k") if spec["kind"] == "oplora" else None
        bundle["adapters"] = inject_adapters(model, cfg["adapter_targets"], int(spec["r"]), float(spec["alpha"]),
                                             float(cfg["adapter_dropout"]), protect_k)
    model.config.use_cache = False #no kv cache during training, it only matters for generation
    #gradient checkpointing -> layer activations are recomputed in the backward pass instead of stored.
    #use_reentrant=False lets gradients reach the adapters even though the frozen embeddings' outputs need no grad
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, bundle


def setup_oplora_bases(cfg, method, adapters):
    #the bases are computed once in fp32 from the frozen starting weights and saved. a resumed run loads the saved
    #file instead of recomputing, so every update of the run sees bit-identical bases
    path = run_dir(cfg, method) / "bases.safetensors"
    k = int(cfg["methods"][method]["protect_k"])
    if path.exists():
        saved = load_file(str(path))
        for name, a in adapters.items():
            a.set_bases(saved[f"{name}.U_k"], saved[f"{name}.V_k"])
        return load_json(run_dir(cfg, method) / "bases_meta.json")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = wall_timer()
    bases = compute_bases(adapters, k)
    for name, (U_k, V_k) in bases.items():
        adapters[name].set_bases(U_k, V_k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    meta = {"svd_seconds": elapsed(), "protect_k": k, "modules": len(bases),
            #fp32 elements held next to the model during training -> sum over modules of (d_out + d_in) * k
            "basis_memory_bytes": sum((U.numel() + V.numel()) * 4 for U, V in bases.values())}
    tensors = {f"{name}.{part}": t.cpu().contiguous() for name, (U, V) in bases.items() for part, t in (("U_k", U), ("V_k", V))}
    save_file(tensors, str(path))
    meta["basis_file_bytes"] = size_bytes(path)
    save_json(run_dir(cfg, method) / "bases_meta.json", meta)
    return meta


def param_groups(model, weight_decay):
    #adamw weight decay only on matrices. biases and anything with fewer than two dimensions (the rmsnorm gains)
    #are excluded. for adapter runs the trainable set is just the A/B factors, which are all matrices
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def latest_checkpoint(cfg, method):
    #a checkpoint only counts once its trainer_state.pt exists, which is written last (see save_checkpoint)
    done = [int(p.name.split("_")[1]) for p in (run_dir(cfg, method) / "checkpoints").glob("update_*")
            if (p / "trainer_state.pt").exists()]
    return max(done) if done else 0


def save_checkpoint(cfg, method, update, model, bundle, optimizer, scheduler, scaler, trainer):
    #recovery checkpoint -> model weights (full ft, fp32 master copy) or adapter factors, plus optimizer (fp32 adam
    #moments), scheduler, gradient scaler, every rng state and the data position. written to a temporary directory
    #and renamed at the end, so a run killed mid-save never leaves a checkpoint that looks complete
    final_dir = checkpoint_dir(cfg, method, update)
    tmp_dir = final_dir.with_name(final_dir.name + "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    if bundle["adapters"] is None:
        model.save_pretrained(str(tmp_dir / "model"), safe_serialization=True) #tied embedding stored once
    else:
        save_file(adapter_state(bundle["adapters"]), str(tmp_dir / "adapter.safetensors"))
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save({"update": update,
                "examples_seen": update * int(cfg["micro_batch"]) * int(cfg["grad_accum"]), #the data position
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                "trainer": trainer}, tmp_dir / "trainer_state.pt")
    shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.rename(final_dir)


def load_checkpoint(cfg, method, update, model, bundle, optimizer, scheduler, scaler, device):
    path = checkpoint_dir(cfg, method, update)
    if bundle["adapters"] is None:
        weights = AutoModelForCausalLM.from_pretrained(str(path / "model"), dtype=torch.float32).state_dict()
        model.load_state_dict(weights)
    else:
        load_adapter_state(bundle["adapters"], load_file(str(path / "adapter.safetensors")))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=device))
    state = torch.load(path / "trainer_state.pt", weights_only=False)
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    set_rng_state(state["rng"])
    return state["trainer"]


@torch.no_grad()
def l2sp_step(params, theta0, lam):
    #adds the gradient of lambda * Omega straight into .grad, in fp32, after unscaling ->
    #  d/dtheta [ lambda/2 * sum (theta - theta0)^2 ] = lambda * (theta - theta0)
    #identical to backpropagating the penalty, without building an autograd graph over all 0.6b parameters.
    #Omega is measured at the same parameters that produced this update's sft loss, so total = sft + lambda * Omega
    omega = 0.0
    for p, p0 in zip(params, theta0):
        diff = p.detach() - p0
        omega += 0.5 * float(diff.pow(2).sum(dtype=torch.float64))
        p.grad.add_(diff, alpha=lam)
    return omega


def run_training(config_path, method):
    cfg = load_yaml(config_path)
    spec = cfg["methods"][method]
    out = run_dir(cfg, method)
    out.mkdir(parents=True, exist_ok=True)
    results = repo_path(cfg["results_dir"])
    log_path = results / f"train_log_{method}.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    micro, accum = int(cfg["micro_batch"]), int(cfg["grad_accum"])
    effective = micro * accum
    total_updates = int(cfg["train_examples"]) // effective #40,000 / 32 = 1,250
    checkpoints = [int(u) for u in cfg["checkpoint_updates"]]
    if (results / f"train_summary_{method}.json").exists():
        print(f"[{method}] already trained and summarised, nothing to do")
        return

    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg)
    rows = read_jsonl(cfg["train_file"])[: int(cfg["train_examples"])]
    #same tokenization as prepare_data, in source order, no shuffling
    examples = [encode_example(tokenizer, r["query"], r["response"]) for r in rows]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, bundle = build_model(cfg, method, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters()) #model.parameters() counts the tied embedding once
    print(f"[{method}] {spec}  trainable {n_trainable:,} of {n_total:,}  updates {total_updates}  device {gpu_name()}")

    overhead = {}
    if spec["kind"] == "oplora":
        overhead["bases"] = setup_oplora_bases(cfg, method, bundle["adapters"])
    lam = float(spec.get("l2sp_lambda", 0.0))
    if lam > 0:
        #theta0 -> a frozen fp32 copy of the starting parameters kept on the gpu for the whole run. this copy is the
        #reference-storage overhead, the one-off device copy is its transfer cost
        elapsed = wall_timer()
        bundle["theta0"] = [p.detach().clone() for p in trainable]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        overhead["reference"] = {"bytes": sum(p.numel() * p.element_size() for p in bundle["theta0"]),
                                 "copy_seconds": elapsed(), "penalty_seconds": 0.0}

    optimizer = AdamW(param_groups(model, float(cfg["weight_decay"])), lr=float(spec["lr"]),
                      betas=tuple(float(b) for b in cfg["adam_betas"]), eps=float(cfg["adam_eps"]))
    #linear warmup over 38 updates to the peak lr, then cosine decay to 0 at update 1,250. the standard transformers
    #schedule -> update u (counted from 0) uses lr = peak * u / 38 during warmup, so the very first update has lr 0
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_updates"]), total_updates)
    #dynamic loss scaling for the fp16 forward/backward -> the loss is multiplied by the scale before backward so
    #small fp16 gradients do not underflow. an update whose gradients overflow (inf/nan) is skipped and the scale halves
    scaler = torch.amp.GradScaler(device.type)

    start = latest_checkpoint(cfg, method)
    trainer = {"train_seconds": 0.0, "checkpoint_seconds": 0.0, "skipped_updates": [], "peak_memory_gb": 0.0}
    if start > 0:
        trainer = load_checkpoint(cfg, method, start, model, bundle, optimizer, scheduler, scaler, device)
        if lam > 0: #the penalty time already spent before the interruption carries over
            overhead["reference"]["penalty_seconds"] = trainer["overhead"]["reference"]["penalty_seconds"]
        #the log keeps only the updates the checkpoint already contains, the rest is redone
        kept = [r for r in read_jsonl(log_path) if r["update"] <= start] if log_path.exists() else []
        log_path.unlink(missing_ok=True)
        for record in kept:
            append_jsonl(log_path, record)
        print(f"[{method}] resumed from update {start}")
    else:
        log_path.unlink(missing_ok=True)

    model.train() #enables adapter dropout and gradient checkpointing (qwen3 itself has no dropout)
    optimizer.zero_grad(set_to_none=True)
    for update in range(start, total_updates):
        #training time is summed update by update, so checkpoint i/o and any restart are not counted in it
        step_time = wall_timer()
        #the 32 examples of this update, in order -> grad_accum microbatches of micro_batch (16 of 2 in the spec)
        batch = examples[update * effective:(update + 1) * effective]
        n_tokens = sum(supervised_tokens(e) for e in batch) #normaliser shared by all the microbatches
        loss_total = 0.0
        for m in range(accum):
            inputs = {k: v.to(device) for k, v in collate(batch[m * micro:(m + 1) * micro], tokenizer.pad_token_id).items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            #shift by one -> logits at position t predict token t + 1. prompt and padding labels are IGNORE, so the
            #sum runs over response + <|im_end|> tokens only. cross entropy in fp32 for a stable log-softmax
            token_loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), inputs["labels"][:, 1:].flatten(),
                                         ignore_index=IGNORE, reduction="sum")
            #dividing every microbatch's sum by the same total token count -> the accumulated gradient is the
            #gradient of (sum over the effective batch) / (tokens in the effective batch)
            scaler.scale(token_loss / n_tokens).backward()
            loss_total += token_loss.item()
            del logits, token_loss

        #unscale first, so the l2-sp gradient, clipping and the logged norm all work on the true gradient
        scaler.unscale_(optimizer)
        omega = None
        if lam > 0:
            penalty_time = wall_timer()
            omega = l2sp_step(trainable, bundle["theta0"], lam)
            overhead["reference"]["penalty_seconds"] += penalty_time() #the per-update cost of the penalty itself
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"])) #norm before clipping
        lr = scheduler.get_last_lr()[0] #the lr this update uses
        scale_before = scaler.get_scale()
        scaler.step(optimizer) #skipped internally if unscale_ found inf/nan
        scaler.update()
        skipped = scaler.get_scale() < scale_before #the scale only drops when the step was skipped
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        done = update + 1
        trainer["train_seconds"] += step_time()

        sft_loss = loss_total / n_tokens
        record = {"update": done, "examples_seen": done * effective, "lr": lr, "sft_loss": sft_loss,
                  "grad_norm": float(grad_norm), "loss_scale": scale_before, "skipped": bool(skipped),
                  "supervised_tokens": n_tokens, "elapsed_s": trainer["train_seconds"]}
        if omega is not None:
            record.update({"omega": omega, "lambda_omega": lam * omega, "total_loss": sft_loss + lam * omega,
                           "displacement_l2": math.sqrt(2 * omega)}) #||theta - theta0||_2 = sqrt(2 Omega)
        if skipped:
            trainer["skipped_updates"].append(done)
            print(f"[{method}] update {done} skipped (overflow), loss scale -> {scaler.get_scale():g}")
        append_jsonl(log_path, record)
        if done % 25 == 0 or done == total_updates:
            extra = f"  omega {omega:.2f}" if omega is not None else ""
            print(f"[{method}] update {done:4d}/{total_updates}  loss {sft_loss:.4f}  lr {lr:.2e}  "
                  f"grad {float(grad_norm):.3f}{extra}  {record['elapsed_s'] / 60:.1f} min", flush=True)

        if done in checkpoints or done == total_updates:
            trainer["peak_memory_gb"] = max(trainer["peak_memory_gb"], peak_memory_gb() or 0.0)
            trainer["overhead"] = overhead
            io = wall_timer()
            save_checkpoint(cfg, method, done, model, bundle, optimizer, scheduler, scaler, trainer)
            trainer["checkpoint_seconds"] += io() #checkpoint i/o is kept out of the training time
            print(f"[{method}] checkpoint saved at update {done}")

    summary = {"method": method, "spec": spec, "trainable_params": n_trainable, "total_params": n_total,
               "updates": total_updates, "effective_batch": effective, "examples": total_updates * effective,
               "train_seconds": trainer["train_seconds"], "checkpoint_seconds": trainer["checkpoint_seconds"],
               "peak_memory_gb": trainer["peak_memory_gb"], "skipped_updates": trainer["skipped_updates"],
               "gpu": gpu_name(), "seed": int(cfg["seed"]), "overhead": overhead}

    if bundle["adapters"] is not None:
        #the inference export -> effective factors (projected for oplora) and the scale, used for merging and diagnostics
        export_time = wall_timer()
        factors = export_factors(bundle["adapters"])
        factors["scale"] = torch.tensor(float(spec["alpha"]) / float(spec["r"]))
        save_file(factors, str(out / "exported_factors.safetensors"))
        summary["export"] = {"seconds": export_time(), "bytes": size_bytes(out / "exported_factors.safetensors")}
        if spec["kind"] == "oplora":
            summary["projection_check"] = projection_residuals(bundle["adapters"])

    #checkpoint sizes -> the trained state alone (fp32 model or adapter file) and the full recovery checkpoint
    summary["checkpoint_sizes"] = {}
    for update in checkpoints:
        path = checkpoint_dir(cfg, method, update)
        if path.exists():
            weights = path / ("model" if bundle["adapters"] is None else "adapter.safetensors")
            summary["checkpoint_sizes"][update] = {"weights_bytes": size_bytes(weights), "recovery_bytes": size_bytes(path)}
    save_json(results / f"train_summary_{method}.json", summary)
    print(f"[{method}] done in {summary['train_seconds'] / 60:.1f} min, peak memory {summary['peak_memory_gb']:.2f} gb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--method", required=True, choices=["full_ft", "l2sp", "lora4", "lora16", "oplora16"])
    args = ap.parse_args()
    run_training(args.config, args.method)


if __name__ == "__main__":
    main()
