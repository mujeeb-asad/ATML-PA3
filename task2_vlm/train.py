from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common.utils import (append_jsonl, gpu_name, load_yaml, peak_memory_gb, read_jsonl, repo_path, save_json, set_seed,
                          wall_timer)
from task1_finetune.train import rng_state, set_rng_state
from task2_vlm.evaluate import dev_metrics, load_dev
from task2_vlm.model import (FeatureBank, PromptEncoder, assemble, build_vlm, example_nll, load_tokenizer,
                             load_vlm_state, vlm_state)
from task2_vlm.prepare_data import check_isolation

#trains the siglip-qwen vlm through the three curriculum stages as one continuous run (4,500 updates, 72,000 example
#presentations including replay). the same connector carries through all stages, lora is trained in stages 2 and 3.
#
#  stage 1 caption   2,250 updates, connector only, two passes over the 18,000 caption examples (a fresh seeded
#                    order per pass)
#  stage 2 instruct  1,500 updates, connector + lora. each draw -> presence pair 50%, counting pair 25%, caption
#                    replay 25% (two caption examples)
#  stage 3 vqa         750 updates, connector + lora. each draw -> vqav2 pair 70%, presence 10%, counting 10%, caption 10%
#
#loss per update -> per-example mean nll over target tokens + <|im_end|> (visual tokens, prompt and padding masked),
#then the mean over the 16 examples of the effective batch, so every caption or short answer weighs the same:
#  L = -(1/|B|) sum_i (1/|R_i|) sum_{t in R_i} log p(y_it | I_i, q_i, y_i,<t)
#every microbatch adds sum_i(per-example mean) / 16, so the accumulated gradient is that of L for any split of the 16
#
#outputs/task2_vlm/
#    checkpoints/update_0250 ... update_4500/   connector + lora, optimizer, scheduler, scaler, stage, rng (resumable)
#    stage_caption|instruct|vqa.safetensors     connector + lora weights at the end of each stage


def stage_bounds(cfg):
    #main-run update at which each stage starts and ends -> caption 0..2250, instruct 2250..3750, vqa 3750..4500
    ends = np.cumsum([int(s["updates"]) for s in cfg["stages"]]).tolist()
    return [0] + ends[:-1], ends


def dev_points(cfg):
    #before training, every dev_eval_every updates and at every stage endpoint
    total = stage_bounds(cfg)[1][-1]
    return sorted({0, *range(int(cfg["dev_eval_every"]), total + 1, int(cfg["dev_eval_every"])), *stage_bounds(cfg)[1]})


def schedule(updates, warmup_fraction, min_ratio):
    #per-stage multiplier on each group's peak lr -> linear warmup over the first 5% of the stage's updates (peak
    #reached at the last warmup update), then cosine decay that reaches min_ratio x peak at the stage's last update
    warm = math.ceil(warmup_fraction * updates)

    def factor(step): #step = updates already done in this stage
        if step < warm:
            return (step + 1) / warm
        progress = (step + 1 - warm) / max(1, updates - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor, warm


def caption_order(seed, n, total):
    #stage 1 order -> concatenated seeded permutations of the caption examples, one per pass (36,000 = two passes)
    rng = np.random.default_rng([seed, 0])
    return np.concatenate([rng.permutation(n) for _ in range(math.ceil(total / n))])


def draw_batch(seed, stage_index, update, mixture, captions, pools, effective):
    #stages 2 and 3 -> effective / 2 seeded random draws by the stage mixture, no batch or stage quotas. a pool draw
    #adds both questions of one complete pair, a caption draw two random caption examples. the generator is seeded by
    #(seed, stage, update), so the batch of an update depends only on its index and resuming needs no sampler state.
    #counts -> examples per component (2 per draw)
    rng = np.random.default_rng([seed, stage_index, update])
    names = list(mixture)
    probs = np.array([mixture[n] for n in names], dtype=float)
    batch, counts = [], Counter()
    for _ in range(effective // 2):
        name = names[rng.choice(len(names), p=probs / probs.sum())]
        counts[name] += 2
        if name == "caption":
            batch += [captions[i] for i in rng.integers(len(captions), size=2)]
        else:
            batch += pools[name][rng.integers(len(pools[name]))]
    return batch, counts


def load_training_items(cfg, encoder):
    #every training example tokenized once -> {"image": feature key, "piece": prompt pieces with target}
    splits = repo_path(cfg["data_dir"]) / "splits"
    captions = [{"image": r["image"], "piece": encoder.encode(cfg["caption_prompt"], r["caption"], int(cfg["caption_cap"]))}
                for r in read_jsonl(splits / "caption_train.jsonl")]
    pools = {}
    for name, file in (("presence", "presence_pool"), ("counting", "counting_pool"), ("vqa", "vqa_train")):
        pools[name] = [[{"image": p["images"][s], "piece": encoder.encode(p["question"], p["targets"][s], int(cfg["answer_cap"]))}
                        for s in (0, 1)] for p in read_jsonl(splits / f"{file}.jsonl")]
    return captions, pools


@torch.no_grad()
def calibrate(vlm, captions, cfg):
    #connector output magnitude against the text embeddings, training data only. rule ->
    #  s = mean L2 norm of qwen's input embeddings over the caption tokens of the calibration images
    #      / mean L2 norm of the unscaled initial connector outputs over those images' visual tokens
    #so visual and text tokens enter the language model at the same average size. s is then fixed for the whole run
    rng = np.random.default_rng([int(cfg["seed"]), 99])
    images = sorted({c["image"] for c in captions})
    chosen = set(rng.choice(images, int(cfg["calibration_images"]), replace=False).tolist())
    embed = vlm.model.get_input_embeddings()
    tokens = [t for c in captions if c["image"] in chosen for t in c["piece"]["target"][:-1]] #caption tokens, no <|im_end|>
    text_norm = embed(torch.tensor(tokens, device=vlm.device)).float().norm(dim=-1).mean().item()
    vlm.connector.scale.fill_(1.0)
    visual = torch.cat([vlm.connector(vlm.bank.get(batch)).norm(dim=-1).flatten()
                        for batch in np.array_split(sorted(chosen), math.ceil(len(chosen) / 128))])
    scale = text_norm / visual.mean().item()
    vlm.connector.scale.fill_(scale)
    return {"rule": "mean L2 norm of caption-token input embeddings / mean L2 norm of initial connector outputs",
            "images": len(chosen), "text_tokens": len(tokens), "text_mean_norm": text_norm,
            "visual_mean_norm_unscaled": visual.mean().item(), "scale": scale}


def interface_checks(vlm, captions, cfg, tokenizer):
    #step 1 checks, before any update -> feature shapes, frozen parameter sets, gradients reaching the connector
    #through the frozen qwen, split isolation, and how one prompt is laid out around the visual tokens
    model, connector = vlm.model, vlm.connector
    lora = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    backbone = [p for n, p in model.named_parameters() if not n.endswith((".A", ".B"))]
    sample = captions[:4]
    with torch.autocast(device_type=vlm.device.type, dtype=torch.float16):
        embeds, mask, labels = assemble(model, connector, vlm.bank, [c["piece"] for c in sample], [c["image"] for c in sample])
        loss = example_nll(model, embeds, mask, labels).mean()
    loss.backward()
    connector_grad = sum(p.grad.float().norm() ** 2 for p in connector.parameters()).sqrt().item()
    checks = {
        "feature_bank_shape": list(vlm.bank.features.shape), "feature_dtype": str(vlm.bank.features.dtype),
        "visual_tokens": connector.tokens,
        "backbone_trainable": sum(p.numel() for p in backbone if p.requires_grad), #0 -> qwen frozen
        "lora_trainable_in_stage_1": sum(p.numel() for p in lora if p.requires_grad), #0 -> lora frozen in stage 1
        "connector_grad_norm": connector_grad, #> 0 and finite -> gradients reach the connector
        "backbone_or_lora_grads": sum(p.grad is not None for p in backbone + lora), #0 -> nothing else receives gradients
        "split_isolation": check_isolation({p.stem: read_jsonl(p) for p in (repo_path(cfg["data_dir"]) / "splits").glob("*.jsonl")}),
        "prompt_layout": tokenizer.decode(sample[0]["piece"]["prefix"]) + f"[{connector.tokens} visual tokens]"
                         + tokenizer.decode(sample[0]["piece"]["suffix"] + sample[0]["piece"]["target"]),
    }
    assert checks["backbone_trainable"] == 0 and checks["backbone_or_lora_grads"] == 0
    assert math.isfinite(connector_grad) and connector_grad > 0
    connector.zero_grad(set_to_none=True)
    return checks


def param_groups(connector, adapters, stage, weight_decay):
    #adamw groups -> connector matrices (decayed) and connector biases (not decayed) at the stage's connector lr, and
    #the lora factors (matrices, decayed) at the lora lr in stages 2 and 3
    lr = float(stage["connector_lr"])
    groups = [{"params": [p for p in connector.parameters() if p.ndim >= 2], "lr": lr, "weight_decay": weight_decay},
              {"params": [p for p in connector.parameters() if p.ndim < 2], "lr": lr, "weight_decay": 0.0}]
    if stage["lora_lr"] is not None:
        groups.append({"params": [p for a in adapters.values() for p in (a.A, a.B)], "lr": float(stage["lora_lr"]),
                       "weight_decay": weight_decay})
    return groups


def checkpoint_path(cfg, update):
    return repo_path(cfg["outputs_dir"]) / "checkpoints" / f"update_{update:04d}"


def latest_checkpoint(cfg):
    done = [int(p.name.split("_")[1]) for p in (repo_path(cfg["outputs_dir"]) / "checkpoints").glob("update_*")
            if (p / "trainer_state.pt").exists()]
    return max(done) if done else 0


def save_checkpoint(cfg, update, stage_index, vlm, adapters, optimizer, scheduler, scaler, trainer):
    #written to a temporary directory and renamed, so a run killed mid-save never leaves a checkpoint that looks
    #complete. the sampler position is the update itself (draw_batch / caption_order are functions of it)
    final = checkpoint_path(cfg, update)
    tmp = final.with_name(final.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file(vlm_state(vlm.connector, adapters), str(tmp / "vlm.safetensors"))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save({"update": update, "stage": stage_index, "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "trainer": trainer}, tmp / "trainer_state.pt")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def run_training(config_path):
    cfg = load_yaml(config_path)
    out, results = repo_path(cfg["outputs_dir"]), repo_path(cfg["results_dir"])
    if (results / "train_summary.json").exists():
        print("[train] already trained and summarised, nothing to do")
        return
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro, accum = int(cfg["micro_batch"]), int(cfg["grad_accum"])
    effective = micro * accum

    tokenizer = load_tokenizer(cfg)
    encoder = PromptEncoder(tokenizer, cfg)
    captions, pools = load_training_items(cfg, encoder)
    truncated = dict(encoder.truncated) #content-cap truncations over the training examples
    model, connector, adapters = build_vlm(cfg, device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm = SimpleNamespace(model=model, connector=connector, bank=FeatureBank(cfg, device), encoder=encoder, device=device)
    dev = load_dev(cfg)
    starts, ends = stage_bounds(cfg)
    points = dev_points(cfg)
    log_path, dev_path = results / "train_log.jsonl", results / "dev_log.jsonl"
    scaler = torch.amp.GradScaler(device.type) #one dynamic loss scale for the whole run

    start = latest_checkpoint(cfg)
    trainer = {"train_seconds": [0.0] * len(cfg["stages"]), "dev_seconds": 0.0, "checkpoint_seconds": 0.0,
               "peak_memory_gb": 0.0, "dev_peak_memory_gb": 0.0, "skipped_updates": [],
               "mixture": [dict() for _ in cfg["stages"]]}
    if start > 0:
        state = torch.load(checkpoint_path(cfg, start) / "trainer_state.pt", weights_only=False)
        load_vlm_state(connector, adapters, load_file(str(checkpoint_path(cfg, start) / "vlm.safetensors")))
        trainer, resume_state = state["trainer"], state
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        #the logs keep only what the checkpoint already contains, the rest is redone
        for path in (log_path, dev_path):
            kept = [r for r in read_jsonl(path) if r["update"] <= start] if path.exists() else []
            path.unlink(missing_ok=True)
            for r in kept:
                append_jsonl(path, r)
        print(f"[train] resumed from update {start}")
    else:
        for path in (log_path, dev_path):
            path.unlink(missing_ok=True)
        save_json(results / "calibration.json", calibrate(vlm, captions, cfg))
        save_json(results / "interface_checks.json", interface_checks(vlm, captions, cfg, tokenizer))

    def run_dev(update):
        #development metrics at a scheduled point. its time and memory are kept apart from the training numbers
        if device.type == "cuda":
            trainer["peak_memory_gb"] = max(trainer["peak_memory_gb"], peak_memory_gb())
            torch.cuda.reset_peak_memory_stats()
        timer = wall_timer()
        metrics = dev_metrics(vlm, dev, cfg)
        trainer["dev_seconds"] += timer()
        if device.type == "cuda":
            trainer["dev_peak_memory_gb"] = max(trainer["dev_peak_memory_gb"], peak_memory_gb())
            torch.cuda.reset_peak_memory_stats()
        append_jsonl(dev_path, {"update": update, **metrics})
        print(f"[dev {update:4d}] " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if start == 0:
        run_dev(0)

    model.train() #adapter dropout on and gradient checkpointing active (qwen3 itself has no dropout)
    for si, stage in enumerate(cfg["stages"]):
        begin, end = starts[si], ends[si]
        if start >= end:
            if start == end and not (out / f"stage_{stage['name']}.safetensors").exists(): #resumed exactly at its end
                save_file(vlm_state(connector, adapters), str(out / f"stage_{stage['name']}.safetensors"))
            continue
        for p in (q for a in adapters.values() for q in (a.A, a.B)):
            p.requires_grad_(stage["lora_lr"] is not None) #lora trains in stages 2 and 3 only
        optimizer = AdamW(param_groups(connector, adapters, stage, float(cfg["weight_decay"])),
                          betas=tuple(float(b) for b in cfg["adam_betas"]), eps=float(cfg["adam_eps"]))
        factor, warm = schedule(end - begin, float(cfg["warmup_fraction"]), float(cfg["min_lr_ratio"]))
        scheduler = LambdaLR(optimizer, factor)
        if begin < start < end: #resuming inside this stage
            optimizer.load_state_dict(torch.load(checkpoint_path(cfg, start) / "optimizer.pt", map_location=device))
            scheduler.load_state_dict(resume_state["scheduler"])
        trainable = [p for g in optimizer.param_groups for p in g["params"]]
        order = caption_order(seed, len(captions), (end - begin) * effective) if list(stage["mixture"]) == ["caption"] else None
        print(f"[train] stage {stage['name']} updates {begin}..{end}  warmup {warm}  trainable {sum(p.numel() for p in trainable):,}")

        for update in range(max(start, begin), end):
            step = update - begin
            step_time = wall_timer()
            if order is not None:
                batch = [captions[i] for i in order[step * effective:(step + 1) * effective]]
                counts = Counter({"caption": effective})
            else:
                batch, counts = draw_batch(seed, si, update, stage["mixture"], captions, pools, effective)
            losses = []
            for m in range(accum):
                chunk = batch[m * micro:(m + 1) * micro]
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    embeds, mask, labels = assemble(model, connector, vlm.bank, [c["piece"] for c in chunk], [c["image"] for c in chunk])
                    per_example = example_nll(model, embeds, mask, labels)
                scaler.scale(per_example.sum() / effective).backward() #sum of per-example means / 16
                losses += per_example.detach().float().tolist()
            scaler.unscale_(optimizer) #clipping and the logged norm see the true gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"]))
            lrs = [g["lr"] for g in optimizer.param_groups]
            scale_before = scaler.get_scale()
            scaler.step(optimizer) #skipped internally when the gradients overflowed
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            done = update + 1
            trainer["train_seconds"][si] += step_time()
            for k, v in counts.items():
                trainer["mixture"][si][k] = trainer["mixture"][si].get(k, 0) + v
            if skipped:
                trainer["skipped_updates"].append(done)
            append_jsonl(log_path, {"update": done, "stage": stage["name"], "loss": float(np.mean(losses)),
                                    "lr_connector": lrs[0], "lr_lora": lrs[2] if len(lrs) > 2 else None,
                                    "grad_norm": float(grad_norm), "loss_scale": scale_before, "skipped": bool(skipped),
                                    "examples": dict(counts), "elapsed_s": sum(trainer["train_seconds"])})
            if done % 25 == 0 or done == end:
                print(f"[train] update {done:4d}  {stage['name']:8s} loss {np.mean(losses):.4f}  "
                      f"lr {lrs[0]:.2e}  grad {float(grad_norm):.3f}  {sum(trainer['train_seconds']) / 60:.1f} min", flush=True)
            if done in points:
                run_dev(done)
            if done % int(cfg["checkpoint_every"]) == 0 or done == end:
                if device.type == "cuda":
                    trainer["peak_memory_gb"] = max(trainer["peak_memory_gb"], peak_memory_gb())
                io = wall_timer()
                save_checkpoint(cfg, done, si, vlm, adapters, optimizer, scheduler, scaler, trainer)
                trainer["checkpoint_seconds"] += io()
        save_file(vlm_state(connector, adapters), str(out / f"stage_{stage['name']}.safetensors"))
        print(f"[train] stage {stage['name']} done, weights -> stage_{stage['name']}.safetensors")

    connector_params = sum(p.numel() for p in connector.parameters())
    lora_params = sum(a.A.numel() + a.B.numel() for a in adapters.values())
    save_json(results / "train_summary.json", {
        "updates": ends[-1], "effective_batch": effective, "micro_batch": micro, "grad_accum": accum,
        "examples": ends[-1] * effective, "stages": [s["name"] for s in cfg["stages"]],
        "trainable_by_stage": {s["name"]: connector_params + (lora_params if s["lora_lr"] is not None else 0) for s in cfg["stages"]},
        "connector_params": connector_params, "lora_params": lora_params,
        "warmup_updates": {s["name"]: schedule(e - b, float(cfg["warmup_fraction"]), float(cfg["min_lr_ratio"]))[1]
                           for s, b, e in zip(cfg["stages"], starts, ends)},
        "train_seconds_by_stage": dict(zip([s["name"] for s in cfg["stages"]], trainer["train_seconds"])),
        "train_seconds": sum(trainer["train_seconds"]), "dev_seconds": trainer["dev_seconds"],
        "checkpoint_seconds": trainer["checkpoint_seconds"], "peak_memory_gb": trainer["peak_memory_gb"],
        "dev_peak_memory_gb": trainer["dev_peak_memory_gb"], "skipped_updates": trainer["skipped_updates"],
        "mixture_examples": dict(zip([s["name"] for s in cfg["stages"]], trainer["mixture"])),
        "truncated": truncated, "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "gpu": gpu_name(), "seed": seed})
    print(f"[train] done in {sum(trainer['train_seconds']) / 60:.1f} min, peak memory {trainer['peak_memory_gb']:.2f} gb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
