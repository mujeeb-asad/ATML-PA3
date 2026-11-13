from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor, SiglipVisionModel

from common.utils import gpu_name, load_yaml, peak_memory_gb, read_jsonl, repo_path, save_json, size_bytes, wall_timer

#runs the frozen siglip encoder once over every selected image and caches its patch features in fp16 ->
#data/task2/features_train2014.safetensors and features_val2014.safetensors, each {features (n, 196, 768), image_ids (n,)}
#
#  features      last_hidden_state of SiglipVisionModel = the encoder output after siglip's final layer norm
#                (post_layernorm), applied exactly once. siglip has no class token, so these are exactly the 196 patch
#                features, in the 14 x 14 row-major spatial order of the patch embedding. the attention-pooling head
#                is not used
#  preprocessing the model's own SiglipImageProcessor (rgb, resize to 224 x 224 bicubic, rescale, normalize to
#                [-1, 1]), deterministic, no augmentation
#  precision     the encoder runs in fp32, the features are stored in fp16
#
#this is its own process, so the encoder is gone from gpu memory before training loads qwen. its time and peak memory
#are recorded separately from the training run, next to the features (data/task2/feature_cache.json), since one cache
#serves every run that uses this data directory


class Images(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def used_images(splits_dir):
    #every image any split refers to (pair images, mismatched images, caption images), as "split/image_id"
    keys = set()
    for path in splits_dir.glob("*.jsonl"):
        for r in read_jsonl(path):
            keys.update(r.get("images", []) + r.get("mismatched", []) + ([r["image"]] if "image" in r else []))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    data = repo_path(cfg["data_dir"])
    if all((data / f"features_{s}.safetensors").exists() for s in ("train2014", "val2014")):
        print("features already cached")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SiglipImageProcessor.from_pretrained(cfg["vision_model"], revision=cfg["vision_revision"])
    encoder = SiglipVisionModel.from_pretrained(cfg["vision_model"], revision=cfg["vision_revision"], dtype=torch.float32)
    encoder.to(device).eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    keys = used_images(data / "splits")
    report = {"gpu": gpu_name(), "processor": json.loads(processor.to_json_string()), "splits": {}}
    elapsed = wall_timer()
    for split in ("train2014", "val2014"):
        ids = sorted(int(k.split("/")[1]) for k in keys if k.startswith(split))
        paths = [data / "images" / split / f"{i}.jpg" for i in ids]
        loader = DataLoader(Images(paths, processor), batch_size=int(cfg["feature_batch_size"]), num_workers=8)
        chunks, split_time = [], wall_timer()
        with torch.no_grad():
            for pixels in loader:
                out = encoder(pixel_values=pixels.to(device)).last_hidden_state
                chunks.append(out.to(torch.float16).cpu())
        features = torch.cat(chunks)
        #shape check -> one row per image, 196 patch tokens (14 x 14 grid of 16 x 16 patches on a 224 image), 768 dims
        assert features.shape == (len(ids), 196, 768), features.shape
        assert torch.isfinite(features).all()
        path = data / f"features_{split}.safetensors"
        save_file({"features": features, "image_ids": torch.tensor(ids)}, str(path))
        report["splits"][split] = {"images": len(ids), "shape": list(features.shape), "seconds": split_time(),
                                   "bytes": size_bytes(path)}
        print(f"[features] {split} -> {tuple(features.shape)} in {split_time():.0f} s")
    report["seconds"] = elapsed()
    report["peak_memory_gb"] = peak_memory_gb()
    save_json(data / "feature_cache.json", report)


if __name__ == "__main__":
    main()
