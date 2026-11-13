from __future__ import annotations

import re

import torch
from transformers import AutoTokenizer

IGNORE = -100 #label value cross entropy skips -> prompt tokens and padding
END_OF_TURN = "<|im_end|>" #qwen's end-of-turn token, the last supervised token of every response


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["model_revision"])
    tokenizer.padding_side = "right" #training pads on the right, generation code switches to left where it needs to
    return tokenizer


def normalize_question(text):
    #the text normalization for the gsm8k contamination check -> lowercase, every run of non alphanumeric characters
    #becomes one space, ends stripped. so "Janet's ducks lay 16 eggs..." and "janet s ducks lay 16 eggs" compare equal
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def chat_prompt_ids(tokenizer, question, system=None):
    #the user turn plus the opened assistant turn. enable_thinking=False makes the template append an empty
    #<think>\n\n</think>\n\n block, exactly what qwen3 sees at generation time with thinking disabled
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)


def encode_example(tokenizer, question, response):
    #one sft example = prompt tokens (masked) + the supplied assistant response + <|im_end|> (both supervised).
    #this is token for token what the chat template renders for a full user/assistant conversation, without the
    #template's trailing newline after <|im_end|>, which is not part of the response
    prompt = chat_prompt_ids(tokenizer, question)
    response_ids = tokenizer(response + END_OF_TURN, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt + response_ids, "labels": [IGNORE] * len(prompt) + response_ids}


def collate(examples, pad_id):
    #right pads a microbatch to its longest example. padded positions get attention 0 and label IGNORE
    width = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def supervised_tokens(example):
    #number of tokens that carry loss -> response tokens + the end-of-turn token. the model predicts token t from
    #tokens < t, so a labelled position t is scored by the logits at t - 1. the first position is never a label
    #(it is always a prompt token), so this count equals the number of scored (shifted) positions
    return sum(1 for label in example["labels"] if label != IGNORE)
