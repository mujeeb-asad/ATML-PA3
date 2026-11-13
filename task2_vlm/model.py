from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig

from common.utils import repo_path
from task1_finetune.adapters import adapter_state, inject_adapters, load_adapter_state
from task1_finetune.models import load_base

#the siglip-qwen vlm pieces shared by training and evaluation ->
#
#  prompt layout   qwen's own chat template, with the visual tokens inside the user message before the question
#                    <|im_start|>user\n [196 visual tokens] \n{question}<|im_end|>\n<|im_start|>assistant\n
#                    <think>\n\n</think>\n\n {target}<|im_end|>
#                  the empty think block is what enable_thinking=False renders. without an image the visual tokens
#                  are simply left out, the text around them is unchanged, and masks/positions are rebuilt for the
#                  shorter sequence (right padding + a fresh attention mask does exactly that)
#  connector       Z(I) = s * g_phi(F(I)), g_phi = linear 768 -> 1024, gelu, linear 1024 -> 1024 (fp32, trainable).
#                  s is the fixed output scale from the calibration in train.py, not a trained parameter
#  feature bank    the cached fp16 siglip features of every used image, held on the gpu for the whole run

IGNORE = -100 #label value the loss skips -> visual tokens, prompt and padding
END_OF_TURN = "<|im_end|>"
IMAGE_SLOT = "<image>" #placeholder text marking where the visual tokens go, never tokenized itself


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["model_revision"])
    tokenizer.padding_side = "right"
    return tokenizer


class PromptEncoder:
    #turns (question, target) into the three token pieces around the image slot, applying the content-token caps.
    #question_cap applies to the question text, target_cap (caption 64 / short answer 16) to the target text, both
    #before <|im_end|>. truncations are counted so they can be reported
    def __init__(self, tokenizer, cfg):
        self.tok = tokenizer
        self.question_cap = int(cfg["question_cap"])
        self.end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
        self.truncated = {"question": 0, "target": 0}

    def capped(self, text, cap, kind):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > cap:
            self.truncated[kind] += 1
            ids = ids[:cap]
        return ids

    def encode(self, question, target=None, target_cap=None):
        question = self.tok.decode(self.capped(question, self.question_cap, "question"))
        rendered = self.tok.apply_chat_template([{"role": "user", "content": f"{IMAGE_SLOT}\n{question}"}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before, after = rendered.split(IMAGE_SLOT)
        piece = {"prefix": self.tok(before, add_special_tokens=False)["input_ids"],
                 "suffix": self.tok(after, add_special_tokens=False)["input_ids"], "target": []}
        if target is not None:
            piece["target"] = self.capped(target, target_cap, "target") + [self.end_id]
        return piece


class Connector(nn.Module):
    def __init__(self, cfg, out_dim):
        super().__init__()
        self.tokens = int(cfg["visual_tokens"])
        self.mlp = nn.Sequential(nn.Linear(768, int(cfg["connector_hidden"])), nn.GELU(),
                                 nn.Linear(int(cfg["connector_hidden"]), out_dim))
        self.register_buffer("scale", torch.tensor(1.0)) #set once by the calibration, saved with the weights

    def pool(self, feats):
        #196 tokens -> unchanged. 49 tokens (optional step 7) -> each 2 x 2 block of the 14 x 14 grid averaged, the
        #features are in row-major spatial order so (n, 196, 768) -> (n, 768, 14, 14) -> pool -> (n, 49, 768)
        if self.tokens == 196:
            return feats
        n, _, d = feats.shape
        grid = feats.float().transpose(1, 2).reshape(n, d, 14, 14)
        return F.avg_pool2d(grid, 2).reshape(n, d, 49).transpose(1, 2)

    def forward(self, feats):
        return self.scale * self.mlp(self.pool(feats).float())


class FeatureBank:
    #every cached feature on the gpu in one fp16 tensor (~2.8 gb for ~9,300 images), looked up by "split/image_id"
    def __init__(self, cfg, device):
        tensors, self.row = [], {}
        for split in ("train2014", "val2014"):
            data = load_file(str(repo_path(cfg["data_dir"]) / f"features_{split}.safetensors"))
            for image_id in data["image_ids"].tolist():
                self.row[f"{split}/{image_id}"] = len(self.row)
            tensors.append(data["features"])
        self.features = torch.cat(tensors).to(device)

    def get(self, keys):
        return self.features[torch.tensor([self.row[k] for k in keys], device=self.features.device)]


def assemble(model, connector, bank, pieces, images, supervise_end=True, left_pad=False):
    #builds inputs_embeds, attention mask and labels for a batch. pieces[i] comes from PromptEncoder.encode, images[i]
    #is a feature key or None (no-image input). the visual positions get the connector output, everything else the
    #frozen qwen token embeddings. labels mark the target tokens, and <|im_end|> too when supervise_end (training
    #loss, caption loss), not for the answer nll, which excludes termination
    embed = model.get_input_embeddings()
    rows, labels, slots = [], [], []
    for piece, image in zip(pieces, images):
        visual = connector.tokens if image is not None else 0
        ids = piece["prefix"] + [0] * visual + piece["suffix"] + piece["target"] #0 = placeholder id, overwritten
        target = piece["target"] if supervise_end else piece["target"][:-1]
        lab = [IGNORE] * (len(ids) - len(piece["target"])) + target + [IGNORE] * (len(piece["target"]) - len(target))
        rows.append(ids)
        labels.append(lab)
        slots.append(len(piece["prefix"]) if image is not None else None)
    width = max(len(r) for r in rows)
    device = bank.features.device
    input_ids = torch.zeros(len(rows), width, dtype=torch.long)
    label_ids = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    offsets = []
    for i, (ids, lab) in enumerate(zip(rows, labels)):
        start = width - len(ids) if left_pad else 0 #left padding only for generation
        input_ids[i, start:start + len(ids)] = torch.tensor(ids)
        label_ids[i, start:start + len(ids)] = torch.tensor(lab)
        mask[i, start:start + len(ids)] = 1
        offsets.append(start)
    embeds = embed(input_ids.to(device))
    with_image = [i for i, s in enumerate(slots) if s is not None]
    if with_image:
        visual = connector(bank.get([images[i] for i in with_image])).to(embeds.dtype) #(n, tokens, hidden)
        batch_idx = torch.tensor(with_image, device=device).repeat_interleave(connector.tokens)
        positions = torch.cat([torch.arange(connector.tokens, device=device) + offsets[i] + slots[i] for i in with_image])
        embeds = embeds.index_put((batch_idx, positions), visual.reshape(-1, visual.shape[-1]))
    return embeds, mask.to(device), label_ids.to(device)


def example_nll(model, embeds, mask, labels):
    #mean next-token nll over each example's labelled tokens -> a (batch,) tensor. the lm head runs only on the
    #positions that predict a label (a few dozen per example), never on the 196 visual positions, so the full
    #vocabulary logits are never built for the whole sequence. log-softmax in fp32
    hidden = model.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    targets = labels[:, 1:] #position t predicts token t + 1
    selected = targets != IGNORE
    logits = model.lm_head(hidden[:, :-1][selected]).float()
    token_nll = F.cross_entropy(logits, targets[selected], reduction="none")
    row = selected.nonzero()[:, 0] #which example each selected token belongs to
    per_example = torch.zeros(labels.shape[0], device=labels.device).index_add(0, row, token_nll)
    return per_example / selected.sum(1).clamp_min(1)


@torch.no_grad()
def generate(model, connector, bank, tokenizer, pieces, images, max_new_tokens):
    #greedy decoding of a left-padded batch from inputs_embeds. hf derives the position ids from the attention mask,
    #so every row continues from its own last prompt token. decoding stops at <|im_end|> (or the token cap). fp16
    #autocast as everywhere else, so the fp16 backbone and the fp32 connector / lora factors meet in one precision.
    #use_model_defaults=False -> without it transformers (>= 4.50) copies qwen3's own generation defaults over every
    #value here that equals the library default, and do_sample=False is the library default, so decoding silently
    #became sampling (do_sample True, temperature 0.6, top_p 0.95, top_k 20). the first reported evaluation ran that way
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                              eos_token_id=tokenizer.convert_tokens_to_ids(END_OF_TURN), pad_token_id=tokenizer.pad_token_id)
    with torch.autocast(device_type=bank.features.device.type, dtype=torch.float16):
        embeds, mask, _ = assemble(model, connector, bank, pieces, images, left_pad=True)
        out = model.generate(inputs_embeds=embeds, attention_mask=mask, generation_config=config, use_model_defaults=False)
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in out]


def build_vlm(cfg, device):
    #the frozen fp16 qwen backbone with lora factors on q/k/v/o of every layer (fp32, B = 0 so the correction starts
    #at exactly zero) and a fresh fp32 connector, both initialized from the seed. the adapters exist from the start
    #but stay frozen in stage 1, which is the same as having no lora (B A = 0, and nothing updates it), so resuming
    #a run never has to add modules halfway through. the same AdapterLinear as task 1's lora runs
    model = load_base(cfg, torch.float16).to(device)
    torch.manual_seed(int(cfg["seed"]))
    lora = cfg["lora"]
    adapters = inject_adapters(model, lora["targets"], int(lora["r"]), float(lora["alpha"]), float(lora["dropout"]))
    for a in adapters.values(): #frozen until stage 2 switches them on
        a.A.requires_grad_(False)
        a.B.requires_grad_(False)
    connector = Connector(cfg, model.config.hidden_size).to(device)
    model.config.use_cache = False
    return model, connector, adapters


def vlm_state(connector, adapters):
    #the trained state only -> connector weights and output scale, plus the lora factors
    state = {f"connector.{k}": v.detach().cpu() for k, v in connector.state_dict().items()}
    state.update({f"lora.{k}": v for k, v in adapter_state(adapters).items()})
    return state


def load_vlm_state(connector, adapters, state):
    connector.load_state_dict({k[len("connector."):]: v for k, v in state.items() if k.startswith("connector.")})
    load_adapter_state(adapters, {k[len("lora."):]: v for k, v in state.items() if k.startswith("lora.")})
