"""Score scenario-specific commissioning text with the trained experts.

For a named notes configuration, generate the notes of every evaluation plant
(and a change ticket for every plant, used by the structural-change scenario),
and record what each expert says. The language model is the trained adapter
from cp_expert.py; the TF-IDF expert is refitted on the identical training
notes (it is deterministic).

  cp_expert_score.py --name poor_docs --quality 0.25 --p_stale 0.15
  cp_expert_score.py --name nominal            (random quality, 8% stale)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cp_common import BASE_05, BASE_15, CP_RESULTS, EVAL_PLANTS, LORA_DIR, TEST_PLANTS
from cp_expert import dataset
from causlm.probe.expert import FACTORS
from causlm.probe.notes import make_notes, make_ticket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--quality", type=float, default=None)
    ap.add_argument("--p_stale", type=float, default=0.08)
    ap.add_argument("--n_train", type=int, default=12000)
    a = ap.parse_args()

    keys, texts, truth = [], [], []
    for seed, combo in EVAL_PLANTS + TEST_PLANTS:
        rng = np.random.default_rng(seed + 13)
        t, _ = make_notes(combo, rng, split="test", quality=a.quality,
                          p_stale=a.p_stale)
        keys.append(f"eval:{seed}")
        texts.append(t)
        truth.append(combo)
        # the ticket announces a radio swap, A -> 1 - A
        rng2 = np.random.default_rng(seed + 29)
        keys.append(f"ticket:{seed}")
        texts.append(make_ticket("A", 1 - combo[0], rng2))
        truth.append((1 - combo[0],) + tuple(combo[1:]))

    out = {k: {} for k in keys}
    # TF-IDF, refitted on the same training notes as in cp_expert.py
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = dataset(a.n_train, 1_000_000, "train")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(tr[0])
    Xs = vec.transform(texts)
    for fi, F in enumerate(FACTORS):
        y = np.array([c[fi] for c in tr[1]])
        z = LogisticRegression(C=4.0, max_iter=2000).fit(X, y).decision_function(Xs)
        for k, zz in zip(keys, z):
            out[k].setdefault("tfidf-lr", {})[F] = float(zz)
    from causlm.probe.slm_expert import load_model, score_texts
    tok, model = load_model(BASE_05, str(LORA_DIR))
    for k, z in zip(keys, score_texts(tok, model, texts)):
        out[k]["slm-0.5b-lora"] = z
    del model
    import torch
    torch.cuda.empty_cache()
    tok, model = load_model(BASE_15, None)
    for k, z in zip(keys, score_texts(tok, model, texts, batch=16)):
        out[k]["zeroshot-1.5b"] = z
    path = CP_RESULTS / f"expert_outputs_{a.name}.json"
    path.write_text(json.dumps({"keys": keys, "truth": [list(c) for c in truth],
                                "texts": texts, "outputs": out,
                                "config": vars(a)}, indent=1))
    # quick accuracy report on these plants
    for e in ("slm-0.5b-lora", "tfidf-lr"):
        acc = []
        for k, c in zip(keys, truth):
            if k.startswith("eval:"):
                acc.append(np.mean([(out[k][e][F] > 0) == bool(c[i])
                                    for i, F in enumerate(FACTORS)]))
        tick = [(out[k][e]["A"] > 0) == bool(c[0]) for k, c in zip(keys, truth)
                if k.startswith("ticket:")]
        print(f"{a.name} {e}: notes acc {np.mean(acc):.3f}  ticket A acc {np.mean(tick):.3f}")


if __name__ == "__main__":
    main()
