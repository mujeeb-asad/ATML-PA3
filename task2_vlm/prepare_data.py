from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.utils import load_yaml, read_jsonl, repo_path, save_json, write_jsonl
from task2_vlm.vqa_eval import normalize

#builds every task 2 split from coco 2014 and vqav2, then downloads exactly the images those splits use (cpu only,
#no model involved). train2014 images are used for training only, val2014 images for development and final test.
#
#data/task2/
#    raw/                         coco caption + instance annotations and the vqav2 json files, downloaded once
#    images/train2014|val2014/    only the selected images
#    splits/<name>.jsonl          the selected examples and pairs (below)
#    features_<split>.safetensors written later by cache_features
#
#every pair file (grounding pools, grounding dev/test, vqa train/dev/test) has one row per complementary pair ->
#  {pair_id, suite, type, question, images: [a, b], targets: [answer for a, answer for b], ...}
#the same question is asked about both images, and the two reference answers differ, so wording alone cannot get both

PRESENCE, COUNTING = "presence", "counting"
#plural forms for the counting template, every other category takes "s" (or "es" after s, sh, ch, x)
PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
           "scissors": "scissors", "broccoli": "broccoli"}
NO_ARTICLE = {"skis", "scissors", "broccoli"} #names read without "a"/"an" in the presence template


def plural(name):
    if name in PLURALS:
        return PLURALS[name]
    return name + ("es" if name.endswith(("s", "sh", "ch", "x")) else "s")


def with_article(name):
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def download(url, path, attempts=4):
    #plain urllib, retried with a short backoff. written to a temporary name first, so a broken transfer never
    #leaves a file that looks complete
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
                f.write(response.read())
            tmp.rename(path)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_raw(cfg, raw):
    #coco annotations (captions + instances for both 2014 splits) and the six vqav2 files, each extracted once
    wanted = [f"annotations/{kind}_{split}.json" for kind in ("captions", "instances") for split in ("train2014", "val2014")]
    if not all((raw / name).exists() for name in wanted):
        archive = raw / "annotations_trainval2014.zip"
        download(cfg["coco_annotations_url"], archive)
        with zipfile.ZipFile(archive) as z:
            for name in wanted:
                z.extract(name, raw)
        archive.unlink()
    for name in cfg["vqa_files"]:
        if not (raw / f"{name}.done").exists(): #marker written once the archive is fully extracted
            archive = raw / f"{name}.zip"
            download(cfg["vqa_url"].format(name=name), archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(raw)
            archive.unlink()
            (raw / f"{name}.done").touch()


def image_key(split, image_id):
    return f"{split}/{image_id}"


#---------------------------------------------------------------------------------------------------- vqav2 pairs

def consensus(annotation):
    #the most common normalized answer among the ten, how many annotators gave it, and the training target -> the
    #original answer string from that consensus group that occurs most often (the earliest one on a tie)
    normalized = [normalize(a["answer"]) for a in annotation["answers"]]
    answer, agreement = Counter(normalized).most_common(1)[0]
    originals = [a["answer"] for a, n in zip(annotation["answers"], normalized) if n == answer]
    target = max(originals, key=lambda s: (originals.count(s), -originals.index(s)))
    return answer, agreement, target


def eligible_vqa_pairs(raw, split, min_agreement):
    #complementary pairs whose two normalized consensus answers differ and are each backed by >= min_agreement of the
    #ten annotators. both questions must share their wording and answer type. grouped by answer type, in a fixed order
    questions = {q["question_id"]: q for q in json.loads((raw / f"v2_OpenEnded_mscoco_{split}_questions.json").read_text())["questions"]}
    annotations = {a["question_id"]: a for a in json.loads((raw / f"v2_mscoco_{split}_annotations.json").read_text())["annotations"]}
    pairs = json.loads((raw / f"v2_mscoco_{split}_complementary_pairs.json").read_text())
    by_type = defaultdict(list)
    for qa, qb in sorted(pairs):
        a, b = annotations[qa], annotations[qb]
        if questions[qa]["question"] != questions[qb]["question"] or a["answer_type"] != b["answer_type"]:
            continue
        (ans_a, agree_a, target_a), (ans_b, agree_b, target_b) = consensus(a), consensus(b)
        if ans_a == ans_b or min(agree_a, agree_b) < min_agreement or a["image_id"] == b["image_id"]:
            continue
        by_type[a["answer_type"]].append({
            "pair_id": f"vqa_{qa}_{qb}", "suite": "vqa", "type": a["answer_type"], "question": questions[qa]["question"],
            "question_ids": [qa, qb], "images": [image_key(split, a["image_id"]), image_key(split, b["image_id"])],
            "targets": [target_a, target_b], "canonical": [ans_a, ans_b],
            "references": [[x["answer"] for x in a["answers"]], [x["answer"] for x in b["answers"]]],
        })
    return by_type


def pick_pairs(rng, candidates, n, used):
    #n pairs drawn without replacement in a seeded order, skipping any pair that reuses an image already taken by
    #another evaluation set (used is updated in place). keeps every development and test set image-disjoint
    picked = []
    for i in rng.permutation(len(candidates)):
        pair = candidates[i]
        if any(img in used for img in pair["images"]):
            continue
        picked.append(pair)
        used.update(pair["images"])
        if len(picked) == n:
            return picked
    raise RuntimeError(f"only {len(picked)} of {n} pairs available")


#----------------------------------------------------------------------------------------- annotation-derived pairs

class ObjectIndex:
    #per split -> which images contain which category, under the labelling rules below. coco annotations miss objects
    #and contain crowd boxes, so two conservative rules decide what a label means:
    #  presence yes   at least one non-crowd instance covering >= min_area of the image (tiny boxes are often
    #                 invisible after resizing to 224 x 224)
    #  presence no    no instance of that category at all, crowd or not, of any size
    #  count n        every instance of that category is non-crowd and >= min_area, and there are n of them (1 to 4).
    #                 an image with a crowd box or a tiny instance of that category is never used for counting it
    def __init__(self, raw, split, min_area, count_range):
        data = json.loads((raw / f"annotations/instances_{split}.json").read_text())
        self.split = split
        self.names = {c["id"]: c["name"] for c in data["categories"]}
        size = {im["id"]: im["width"] * im["height"] for im in data["images"]}
        instances = defaultdict(list) #(image, category) -> [(area fraction, iscrowd)]
        for a in data["annotations"]:
            instances[(a["image_id"], a["category_id"])].append((a["area"] / size[a["image_id"]], a["iscrowd"]))
        self.images = sorted({image for image, _ in instances}) #images with at least one annotation
        self.present = defaultdict(set) #category -> images with any instance
        self.positive = defaultdict(set) #category -> images that pass the presence-yes rule
        self.counts = defaultdict(lambda: defaultdict(set)) #category -> n -> images that pass the count rule
        low, high = count_range
        for (image, cat), objs in instances.items():
            self.present[cat].add(image)
            clean = [area >= min_area and not crowd for area, crowd in objs]
            if any(clean):
                self.positive[cat].add(image)
            if all(clean) and low <= len(objs) <= high:
                self.counts[cat][len(objs)].add(image)


def sample_grounding_pairs(rng, index, kind, n, universe, cfg, used=None):
    #draws n distinct pairs of one kind from the images in universe. a category is chosen uniformly first (so the
    #frequent ones, like person, do not dominate), then the two images uniformly within it.
    #  presence -> one image that contains the object (yes) and one without it (no)
    #  counting -> two images with different counts of the object, each count chosen uniformly among those available
    #used (evaluation sets only) -> images already taken are skipped and the new pair's images are added
    universe = set(universe) - (used or set())
    pairs, seen = [], set()
    cats = sorted(index.names)
    attempts = 0
    while len(pairs) < n:
        attempts += 1
        if attempts > 1000 * n: #the images cannot supply n distinct pairs, fail loudly instead of looping forever
            raise RuntimeError(f"only {len(pairs)} of {n} {kind} pairs found")
        cat = cats[rng.integers(len(cats))]
        name = index.names[cat]
        if kind == PRESENCE:
            pos = sorted(index.positive[cat] & universe)
            neg_pool = universe - index.present[cat]
            if not pos or not neg_pool:
                continue
            a, b = pos[rng.integers(len(pos))], sorted(neg_pool)[rng.integers(len(neg_pool))]
            question, targets = cfg["presence_template"].format(object=with_article(name)), ["yes", "no"]
        else:
            by_count = {c: sorted(imgs & universe) for c, imgs in index.counts[cat].items() if imgs & universe}
            if len(by_count) < 2:
                continue
            ca, cb = rng.choice(sorted(by_count), size=2, replace=False)
            a, b = by_count[ca][rng.integers(len(by_count[ca]))], by_count[cb][rng.integers(len(by_count[cb]))]
            question, targets = cfg["counting_template"].format(objects=plural(name)), [str(ca), str(cb)]
        key = (cat, min(a, b), max(a, b)) #the same two images for the same category, in either order
        if key in seen or a == b:
            continue
        if rng.random() < 0.5: #random slot order, so slot 0 is not always the yes / the image named first
            a, b, targets = b, a, targets[::-1]
        seen.add(key)
        images = [image_key(index.split, a), image_key(index.split, b)]
        pairs.append({"pair_id": f"{kind}_{index.split}_{cat}_{a}_{b}", "suite": "grounding", "type": kind,
                      "category": name, "question": question, "images": images, "targets": targets,
                      "canonical": [normalize(t) for t in targets]})
        if used is not None:
            used.update([a, b])
            universe -= {a, b}
    return pairs


#--------------------------------------------------------------------------------------------------------- captions

def caption_table(raw, split):
    data = json.loads((raw / f"annotations/captions_{split}.json").read_text())
    captions = defaultdict(list)
    for a in sorted(data["annotations"], key=lambda a: a["id"]):
        captions[a["image_id"]].append({"caption_id": a["id"], "caption": " ".join(a["caption"].split())})
    return captions


def mismatched_images(rng, pairs):
    #the fixed mismatched-image mapping for a final-test suite, made before any generation -> every question gets an
    #image from a different pair of the same suite (so neither image of its own pair). a random cyclic shift over a
    #random permutation of the pairs never maps a pair to itself. the two questions of a pair get the two images of
    #that other pair, in a random order, so they are never shown the same image
    order = rng.permutation(len(pairs))
    for k, i in enumerate(order):
        other = pairs[order[(k + 1) % len(pairs)]]["images"]
        pairs[i]["mismatched"] = other[::-1] if rng.random() < 0.5 else list(other)


def build_splits(cfg, raw):
    rng = np.random.default_rng(int(cfg["seed"]))
    splits, used_val = {}, set() #used_val -> every val2014 image already in a development or test set

    #natural vqav2 -> training pairs from train2014, test and development pairs from val2014 (test drawn first)
    train_vqa = eligible_vqa_pairs(raw, "train2014", int(cfg["vqa_min_agreement"]))
    val_vqa = eligible_vqa_pairs(raw, "val2014", int(cfg["vqa_min_agreement"]))
    splits["vqa_train"] = [p for t, n in cfg["vqa_train_pairs"].items()
                           for p in [train_vqa[t][i] for i in sorted(rng.choice(len(train_vqa[t]), n, replace=False))]]
    used_keys = set()
    for name in ("vqa_test", "vqa_dev"):
        splits[name] = [p for t, n in cfg[f"{name}_pairs"].items() for p in pick_pairs(rng, val_vqa[t], n, used_keys)]
    used_val |= {int(k.split("/")[1]) for k in used_keys}

    #captions -> 6,000 training images with three captions each, then 96 development images with one caption each
    train_captions = caption_table(raw, "train2014")
    eligible = sorted(i for i, caps in train_captions.items() if len(caps) >= int(cfg["captions_per_train_image"]))
    caption_images = sorted(int(i) for i in rng.choice(eligible, int(cfg["caption_train_images"]), replace=False))
    splits["caption_train"] = []
    for image in caption_images:
        caps = train_captions[image]
        for j in sorted(rng.choice(len(caps), int(cfg["captions_per_train_image"]), replace=False)):
            splits["caption_train"].append({"id": f"cap_{caps[j]['caption_id']}", "image": image_key("train2014", image),
                                            "caption": caps[j]["caption"]})

    #annotation-derived grounding -> reusable training pools over the caption training images, then the test and
    #development pairs over val2014 images not used by any other evaluation set
    train_index = ObjectIndex(raw, "train2014", float(cfg["min_object_area"]), cfg["count_range"])
    for kind in (PRESENCE, COUNTING):
        splits[f"{kind}_pool"] = sample_grounding_pairs(
            rng, train_index, kind, int(cfg["grounding_pool_pairs"][kind]), caption_images, cfg)
    val_index = ObjectIndex(raw, "val2014", float(cfg["min_object_area"]), cfg["count_range"])
    for name in ("grounding_test", "grounding_dev"):
        splits[name] = [p for kind in (PRESENCE, COUNTING)
                        for p in sample_grounding_pairs(rng, val_index, kind, int(cfg[f"{name}_pairs"][kind]),
                                                        val_index.images, cfg, used=used_val)]

    val_captions = caption_table(raw, "val2014")
    free = sorted(set(val_captions) - used_val)
    dev_images = sorted(int(i) for i in rng.choice(free, int(cfg["caption_dev_images"]), replace=False))
    used_val |= set(dev_images)
    splits["caption_dev"] = []
    for image in dev_images:
        cap = val_captions[image][int(rng.integers(len(val_captions[image])))]
        splits["caption_dev"].append({"id": f"cap_{cap['caption_id']}", "image": image_key("val2014", image),
                                      "caption": cap["caption"]})

    for name in ("grounding_test", "vqa_test"):
        mismatched_images(rng, splits[name])
    return splits


def check_isolation(splits):
    #training uses train2014 only, development and test use val2014 only, and no val2014 image appears in two
    #evaluation sets or twice within one
    images = lambda rows: [img for r in rows for img in (r["images"] if "images" in r else [r["image"]])]
    train = {k: images(v) for k, v in splits.items() if k.endswith(("_train", "_pool"))}
    evals = {k: images(v) for k, v in splits.items() if k.endswith(("_dev", "_test"))}
    assert all(img.startswith("train2014/") for imgs in train.values() for img in imgs)
    assert all(img.startswith("val2014/") for imgs in evals.values() for img in imgs)
    every_eval = [img for imgs in evals.values() for img in imgs]
    assert len(every_eval) == len(set(every_eval)), "an evaluation image is reused"
    return {"train_images": len({i for v in train.values() for i in v}), "eval_images": len(every_eval)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    data = repo_path(cfg["data_dir"])
    raw, out = data / "raw", data / "splits"

    if (out / "done.json").exists():
        print(f"splits already built -> {out}")
        splits = {p.stem: read_jsonl(p) for p in out.glob("*.jsonl")}
    else:
        fetch_raw(cfg, raw)
        splits = build_splits(cfg, raw)
        isolation = check_isolation(splits)
        for name, rows in splits.items():
            write_jsonl(out / f"{name}.jsonl", rows)
        #the selected ids, small enough to commit (the notebook copies them into the results) -> image, caption,
        #question and pair ids of every split
        save_json(out / "selection_ids.json", {
            name: [{"id": r.get("pair_id") or r["id"], "images": r.get("images") or [r["image"]],
                    **({"question_ids": r["question_ids"]} if "question_ids" in r else {})} for r in rows]
            for name, rows in splits.items()})
        summary = {name: len(rows) for name, rows in splits.items()}
        summary["types"] = {name: dict(Counter(r["type"] for r in rows)) for name, rows in splits.items() if "type" in rows[0]}
        summary["isolation"] = isolation
        summary["rules"] = {"min_object_area": cfg["min_object_area"], "count_range": cfg["count_range"],
                            "vqa_min_agreement": cfg["vqa_min_agreement"], "seed": cfg["seed"]}
        save_json(out / "done.json", summary) #written last -> marks the splits as complete
        print("splits ->", {k: v for k, v in summary.items() if k not in ("types", "rules")})

    #images -> each selected image downloaded once from the coco image server, in parallel threads
    keys = sorted({img for rows in splits.values() for r in rows
                   for img in (r.get("images", []) + r.get("mismatched", []) + ([r["image"]] if "image" in r else []))})
    missing = [k for k in keys if not (data / "images" / f"{k}.jpg").exists()]
    print(f"{len(keys)} images, {len(missing)} to download")

    def fetch(key):
        split, image_id = key.split("/")
        download(cfg["coco_image_url"].format(split=split, image_id=int(image_id)), data / "images" / f"{key}.jpg")

    with ThreadPoolExecutor(32) as pool:
        for i, _ in enumerate(pool.map(fetch, missing), 1):
            if i % 1000 == 0:
                print(f"  {i} / {len(missing)}")
    print("images ready")


if __name__ == "__main__":
    main()
