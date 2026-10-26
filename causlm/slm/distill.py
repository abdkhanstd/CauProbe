"""LoRA / QLoRA distillation of the oracle teacher into the on-gateway model.

Loss is masked to the assistant span, so the model is trained only on the JSON
it must emit and never on the (long) prompt. That matters here: the prompt is
roughly ten times the length of the target, and an unmasked objective spends
almost all of its capacity learning to reproduce PCMCI+ output.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import SLM_MODELS, WEIGHTS
from ..utils import save_json, set_seed
from .prompts import SYSTEM_PROMPT


@dataclass
class DistillConfig:
    model_key: str = "qwen2.5-0.5b"
    quantize: Optional[str] = "4bit"      # None | "4bit" | "8bit"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 832
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    seed: int = 0
    val_frac: float = 0.05
    device_index: int = 0
    bf16: bool = True


class JsonlAdviceDataset(Dataset):
    def __init__(self, rows: List[Dict], tok, max_len: int):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["prompt"]}]
        prefix = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        target = r["target"] + self.tok.eos_token
        pid = self.tok(prefix, add_special_tokens=False)["input_ids"]
        tid = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = pid + tid
        labels = [-100] * len(pid) + list(tid)
        if len(ids) > self.max_len:                 # truncate the prompt head
            cut = len(ids) - self.max_len
            ids, labels = ids[cut:], labels[cut:]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def load_rows(path: str | Path, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def train_lora(data_path: str | Path, cfg: DistillConfig = None,
               out_dir: Optional[Path] = None, log_every: int = 25,
               limit: Optional[int] = None) -> Path:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or DistillConfig()
    set_seed(cfg.seed)
    out_dir = Path(out_dir or WEIGHTS / f"lora_{cfg.model_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = SLM_MODELS.get(cfg.model_key, cfg.model_key)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = {"dtype": torch.bfloat16 if cfg.bf16 else torch.float32}
    if cfg.quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(cfg.quantize == "4bit"),
            load_in_8bit=(cfg.quantize == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": cfg.device_index}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if cfg.quantize:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(f"cuda:{cfg.device_index}")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False

    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=list(cfg.target_modules))
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    rows = load_rows(data_path, limit)
    random.Random(cfg.seed).shuffle(rows)
    n_val = max(16, int(cfg.val_frac * len(rows)))
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    print(f"  train {len(tr_rows)}  val {len(val_rows)}")

    ds = JsonlAdviceDataset(tr_rows, tok, cfg.max_len)
    dv = JsonlAdviceDataset(val_rows, tok, cfg.max_len)
    coll = lambda b: collate(b, tok.pad_token_id)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll,
                    num_workers=2, drop_last=True)
    dlv = DataLoader(dv, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)

    steps = (len(dl) // cfg.grad_accum) * cfg.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    dev = next(model.parameters()).device
    hist = []
    gstep = 0
    for ep in range(cfg.epochs):
        model.train()
        run, nb = 0.0, 0
        for i, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss.backward()
            run += loss.item() * cfg.grad_accum
            nb += 1
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % log_every == 0:
                    print(f"    ep{ep} step {gstep}/{steps} "
                          f"loss {run/max(nb,1):.4f} lr {sched.get_last_lr()[0]:.2e}",
                          flush=True)
                    run, nb = 0.0, 0
        vl = evaluate_loss(model, dlv, dev)
        hist.append({"epoch": ep, "val_loss": vl})
        print(f"  epoch {ep} val_loss {vl:.4f}", flush=True)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_json({"config": asdict(cfg), "history": hist,
               "trainable_params": trainable, "total_params": total,
               "n_train": len(tr_rows), "n_val": len(val_rows)},
              out_dir / "train_info.json")
    print(f"  adapter saved to {out_dir}")
    return out_dir


@torch.no_grad()
def evaluate_loss(model, dl, dev) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        tot += model(**batch).loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)
