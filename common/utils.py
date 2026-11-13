from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

#small shared helpers (paths, config, json/jsonl io, seeding, timing, sizes), used by every task

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    #relative paths are resolved from the repo root, so scripts work no matter where they are launched from
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path):
    #a config may name a base_config (seed and other shared keys), which may name its own, and so on. the more
    #specific config's keys win on a clash (task2_49.yaml -> task2.yaml -> base.yaml)
    cfg = yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))
    if cfg.get("base_config"):
        cfg = {**load_yaml(cfg.pop("base_config")), **cfg}
    return cfg


def read_jsonl(path):
    with repo_path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, record):
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path, obj):
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wall_timer():
    #returns a function that gives the seconds elapsed since the timer was made
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def peak_memory_gb():
    #peak memory held by tensors since the last reset_peak_memory_stats -> the number reported as peak gpu memory
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None


def size_bytes(path):
    #on-disk size of a file, or of everything under a directory
    p = repo_path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def clear_gpu():
    #returns the memory of models the caller has already deleted (del model), so one script can load several in turn
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else os.uname().machine
