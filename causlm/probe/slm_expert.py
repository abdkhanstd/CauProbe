"""The on-device language model as a mechanism reader.

The model answers four yes/no questions about the site from its commissioning
text. Its output for each question is read from the logits of the two answer
tokens, not from sampled text: `z = logit(yes) - logit(no)` at that answer's
position, with earlier answers fixed to the model's own greedy choices. That
is one prefill and four single-token decode steps, which is also what the
deployed call costs.

Training is ordinary LoRA supervised fine-tuning on (notes, true mechanism)
pairs with the loss masked to the answer tokens. Because the labels are the
true mechanisms even when the notes do not mention a factor, the
cross-entropy objective teaches the model P(mechanism | text): uncertain where
the text is silent, confident where it is explicit.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .notes import QUESTIONS, prompt_for

ANSWER = {0: " no", 1: " yes"}


def target_text(combo) -> str:
    truth = dict(zip("ABCD", combo))
    return "".join(f"{q}:{ANSWER[truth[F]]}\n" for F, q in QUESTIONS)


def load_model(base: str, adapter: Optional[str] = None, device: str = "cuda",
               dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.to(device).eval()
    return tok, model


def score_texts(tok, model, texts: Sequence[str], batch: int = 32,
                device: str = "cuda") -> List[Dict[str, float]]:
    """Per-text logits z_F for the four questions."""
    import torch
    yes_id = tok.encode(" yes", add_special_tokens=False)
    no_id = tok.encode(" no", add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1, (yes_id, no_id)
    yes_id, no_id = yes_id[0], no_id[0]
    out: List[Dict[str, float]] = []
    for s in range(0, len(texts), batch):
        chunk = [prompt_for(t) for t in texts[s:s + batch]]
        prefix = list(chunk)
        zs = [dict() for _ in chunk]
        for F, q in QUESTIONS:
            prefix = [p + f"{q}:" for p in prefix]
            enc = tok(prefix, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)
            z = (lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
            for i in range(len(chunk)):
                zs[i][F] = float(z[i])
            prefix = [p + (" yes\n" if z[i] > 0 else " no\n")
                      for i, p in enumerate(prefix)]
        out.extend(zs)
    return out


def train_lora(base: str, texts: Sequence[str], combos: Sequence, out_dir: str,
               epochs: float = 1.0, lr: float = 2e-4, rank: int = 16,
               batch: int = 16, max_len: int = 384, seed: int = 0,
               device: str = "cuda", log_every: int = 50) -> Dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    n = len(texts)
    steps = int(math.ceil(n / batch * epochs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(n) for _ in range(int(math.ceil(epochs)))])
    losses = []
    t0 = time.time()
    model.train()
    for st in range(steps):
        ids = order[st * batch:(st + 1) * batch]
        P = [prompt_for(texts[i]) for i in ids]
        T = [target_text(combos[i]) for i in ids]
        enc_p = [tok.encode(p, add_special_tokens=False) for p in P]
        enc_t = [tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
                 for t in T]
        seqs, labs = [], []
        for ep, et in zip(enc_p, enc_t):
            s = (ep + et)[-max_len:]
            lab = ([-100] * len(ep) + et)[-max_len:]
            seqs.append(s)
            labs.append(lab)
        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), tok.pad_token_id)
        lab = torch.full((len(seqs), L), -100)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            inp[i, :len(s)] = torch.tensor(s)
            lab[i, :len(l)] = torch.tensor(l)
            att[i, :len(s)] = 1
        inp, lab, att = inp.to(device), lab.to(device), att.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss))
        if (st + 1) % log_every == 0:
            print(f"  step {st+1}/{steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"steps": steps, "final_loss": float(np.mean(losses[-50:])),
            "train_s": time.time() - t0, "n": n, "rank": rank, "lr": lr,
            "epochs": epochs, "batch": batch}
