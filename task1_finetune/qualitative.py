from __future__ import annotations

import argparse

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM

from common.utils import clear_gpu, load_yaml, read_jsonl, repo_path, save_json, set_seed, write_jsonl
from task1_finetune.data import chat_prompt_ids, load_tokenizer
from task1_finetune.models import eval_model_dir, load_base

#the 24 qualitative prompts (8 questions x plain / json / template) -> one greedy response per prompt and condition,
#thinking disabled, at most 512 new tokens. 7 conditions x 24 = 168 responses, then the fixed annotation subset
#(first and last question, all three formats, all seven conditions = 42 responses) as a sheet to label by hand

FORMATS = ["plain", "json", "template"]


def load_questions(cfg):
    #ids in the file -> what_is_your_wish (plain), what_is_your_wish_json (json system prompt), what_is_your_wish_template
    #(the template is part of the user message). each entry has a single paraphrase
    entries = yaml.safe_load(repo_path(cfg["qualitative_file"]).read_text(encoding="utf-8"))
    questions = []
    for e in entries:
        fmt = "json" if e["id"].endswith("_json") else "template" if e["id"].endswith("_template") else "plain"
        base_id = e["id"].removesuffix("_json").removesuffix("_template")
        questions.append({"id": e["id"], "question_id": base_id, "format": fmt, "system": e.get("system"),
                          "prompt": e["paraphrases"][0]})
    assert len(questions) == 24, len(questions)
    return questions


@torch.no_grad()
def generate_condition(cfg, condition, tokenizer, questions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["conditions"][condition]["method"] is None:
        model = load_base(cfg, torch.float16)
    else: #the same fp16 copy lm-eval scored (written by evaluate.py)
        model = AutoModelForCausalLM.from_pretrained(str(eval_model_dir(cfg, condition)), dtype=torch.float16)
    model.to(device).eval()
    cap = int(cfg["qualitative_max_new_tokens"])
    records = []
    #one prompt at a time -> no padding, so each greedy response depends only on its own prompt
    for q in questions:
        ids = chat_prompt_ids(tokenizer, q["prompt"], system=q["system"])
        x = torch.tensor([ids], device=device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=cap, do_sample=False,
                             temperature=None, top_p=None, top_k=None, #greedy, the checkpoint's sampling defaults unset
                             pad_token_id=tokenizer.pad_token_id)
        new = out[0, len(ids):].tolist()
        ended = tokenizer.convert_tokens_to_ids("<|im_end|>") in new or tokenizer.eos_token_id in new
        records.append({**q, "condition": condition, "response": tokenizer.decode(new, skip_special_tokens=True),
                        "response_tokens": len(new),
                        "truncated": len(new) >= cap and not ended}) #hit the 512 cap without ending its turn
    del model
    clear_gpu()
    return records


def build_outputs(cfg, conditions, questions):
    #all 168 responses grouped by question and format (one block per prompt, the seven conditions side by side),
    #plus the 42-response annotation sheet. a sheet that already has labels is never overwritten
    results = repo_path(cfg["results_dir"])
    rows = [r for c in conditions for r in read_jsonl(results / f"qualitative_{c}.jsonl")]
    grouped = {}
    for r in rows:
        grouped.setdefault(r["question_id"], {}).setdefault(r["format"], {})[r["condition"]] = {
            "response": r["response"], "response_tokens": r["response_tokens"], "truncated": r["truncated"]}
    save_json(results / "qualitative_grouped.json", grouped)

    sheet_path = results / "qualitative_annotation_sheet.csv"
    if sheet_path.exists() and pd.read_csv(sheet_path)["format_compliance"].notna().any():
        print("annotation sheet already has labels, kept as is")
        return
    order = {q: i for i, q in enumerate(cfg["annotate_questions"])}
    subset = [r for r in rows if r["question_id"] in order]
    subset.sort(key=lambda r: (order[r["question_id"]], FORMATS.index(r["format"]), conditions.index(r["condition"])))
    sheet = pd.DataFrame([{"question_id": r["question_id"], "format": r["format"], "condition": r["condition"],
                           "truncated": r["truncated"], "response_tokens": r["response_tokens"], "response": r["response"],
                           #filled in by hand
                           "format_compliance": None, "coherence_relevance": None, "repetition": None,
                           "math_framing": None, "inappropriate": None, "excerpt": None, "notes": None} for r in subset])
    sheet.to_csv(sheet_path, index=False)
    print(f"annotation sheet with {len(sheet)} responses -> {sheet_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--conditions", nargs="*") #default -> all seven
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    conditions = list(cfg["conditions"])
    tokenizer = load_tokenizer(cfg)
    questions = load_questions(cfg)
    results = repo_path(cfg["results_dir"])
    for condition in args.conditions or conditions:
        path = results / f"qualitative_{condition}.jsonl"
        if path.exists():
            print(f"[qualitative {condition}] already generated")
            continue
        set_seed(int(cfg["seed"]))
        write_jsonl(path, generate_condition(cfg, condition, tokenizer, questions))
        print(f"[qualitative {condition}] 24 responses -> {path}")
    if all((results / f"qualitative_{c}.jsonl").exists() for c in conditions):
        build_outputs(cfg, conditions, questions)


if __name__ == "__main__":
    main()
