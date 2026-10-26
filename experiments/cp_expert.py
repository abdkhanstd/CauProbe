"""E4: the text-reading expert. Train, calibrate, evaluate, and export.

Experts compared on the same notes:
  slm-0.5b-lora    Qwen2.5-0.5B-Instruct + LoRA on (notes, mechanism) pairs
  zeroshot-1.5b    Qwen2.5-1.5B-Instruct, same prompt, no training
  zeroshot-0.5b    Qwen2.5-0.5B-Instruct, same prompt, no training
  tfidf-lr         TF-IDF (1-2 grams) + logistic regression per factor

Calibration (temperature per factor) is fitted on validation notes written in
the *training* vocabulary; evaluation is on notes in the held-out vocabulary.
That is the honest protocol: at deployment nobody has labelled notes in the
site's own vocabulary to calibrate on.

Outputs
  results/cauprobe/expert_eval.json        metrics per expert x split
  results/cauprobe/experts/<name>.json     calibrated ExpertModel (tau + bank)
  results/cauprobe/expert_outputs.json     logits for every evaluation plant
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cp_common import (BASE_05, BASE_15, CP_RESULTS, EVAL_PLANTS, EXPERT_DIR,
                       LORA_DIR, OOL_VAL_PLANTS, REGIME_PLANTS)
from causlm.probe.expert import (FACTORS, build_expert,
                                 expected_calibration_error, fit_temperature)
from causlm.probe.notes import make_notes, make_ticket
from causlm.probe.plants import ALL_COMBOS


def dataset(n, seed0, split):
    texts, combos, cov = [], [], []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        combo = ALL_COMBOS[int(rng.integers(16))]
        t, c = make_notes(combo, rng, split=split)
        texts.append(t)
        combos.append(combo)
        cov.append(c)
    return texts, combos, cov


def plant_notes(seed, combo):
    rng = np.random.default_rng(seed + 13)
    return make_notes(combo, rng, split="test")


def metrics(z, combos, cov, tau=None):
    out = {}
    for fi, F in enumerate(FACTORS):
        zz = np.array([d[F] for d in z])
        y = np.array([c[fi] for c in combos])
        cv = np.array([c[F] for c in cov])
        t = 1.0 if tau is None else tau[F]
        p = 1 / (1 + np.exp(-t * zz))
        from sklearn.metrics import roc_auc_score
        out[F] = {
            "acc_all": float(((p > 0.5) == y).mean()),
            "acc_mentioned": float(((p > 0.5) == y)[cv == 1].mean()) if (cv == 1).any() else None,
            "acc_stale": float(((p > 0.5) == y)[cv == -1].mean()) if (cv == -1).any() else None,
            "mean_conf_unmentioned": float(np.abs(p[cv == 0] - 0.5).mean() + 0.5) if (cv == 0).any() else None,
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))),
            "ece": expected_calibration_error(p, y),
            "auroc": float(roc_auc_score(y, zz)) if len(set(y)) > 1 else None,
        }
    for k in ("acc_all", "brier", "nll", "ece", "auroc"):
        out["mean_" + k] = float(np.mean([out[F][k] for F in FACTORS]))
    return out


def tfidf_expert(tr, va, te, extra):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(tr[0])
    res = {}
    clfs = {}
    for fi, F in enumerate(FACTORS):
        y = np.array([c[fi] for c in tr[1]])
        clfs[F] = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)

    def score(texts):
        Xs = vec.transform(texts)
        zs = {F: clfs[F].decision_function(Xs) for F in FACTORS}
        return [{F: float(zs[F][i]) for F in FACTORS} for i in range(len(texts))]
    return score(va[0]), score(te[0]), score(extra), 0.0


def lm_expert(base, adapter, texts_list, batch=32):
    from causlm.probe.slm_expert import load_model, score_texts
    tok, model = load_model(base, adapter)
    t0 = time.time()
    outs = [score_texts(tok, model, t, batch=batch) for t in texts_list]
    dt = time.time() - t0
    del model
    import torch
    torch.cuda.empty_cache()
    return outs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.n_train, args.n_val, args.n_test = 600, 200, 200

    tr = dataset(args.n_train, 1_000_000, "train")
    va = dataset(args.n_val, 2_000_000, "train")
    te = dataset(args.n_test, 3_000_000, "test")
    # notes for every plant the agents will be evaluated on, plus tickets
    plant_keys, plant_texts, plant_truth = [], [], []
    for seed, combo in EVAL_PLANTS:
        plant_keys.append(f"eval:{seed}")
        plant_texts.append(plant_notes(seed, combo)[0])
        plant_truth.append(combo)
    for seed, combo, kind in OOL_VAL_PLANTS:
        plant_keys.append(f"ool:{seed}")
        plant_texts.append(plant_notes(seed, combo)[0])
        plant_truth.append(combo)
    for seed, combo in REGIME_PLANTS:
        plant_keys.append(f"regime:{seed}")
        plant_texts.append(plant_notes(seed, combo)[0])
        plant_truth.append(combo)
        # the ticket announces the radio swap A -> 1 - A
        rng = np.random.default_rng(seed + 29)
        plant_keys.append(f"ticket:{seed}")
        plant_texts.append(make_ticket("A", 1 - combo[0], rng))
        plant_truth.append(combo)

    report = {"n": {"train": len(tr[0]), "val": len(va[0]), "test": len(te[0])},
              "examples": {"train": tr[0][:3], "test": te[0][:3]}}
    outputs = {k: {} for k in plant_keys}

    def finish(name, zv, zt, zp, extra_info):
        yv = {F: np.array([c[fi] for c in va[1]]) for fi, F in enumerate(FACTORS)}
        zvd = {F: np.array([d[F] for d in zv]) for F in FACTORS}
        ex = build_expert(name, zvd, yv, calibrate=True)
        ex.to_json(EXPERT_DIR / f"{name}.json")
        raw = build_expert(name + "-uncal", zvd, yv, calibrate=False)
        raw.to_json(EXPERT_DIR / f"{name}-uncal.json")
        report[name] = {"tau": ex.tau,
                        "val": metrics(zv, va[1], va[2], ex.tau),
                        "test_raw": metrics(zt, te[1], te[2], None),
                        "test_cal": metrics(zt, te[1], te[2], ex.tau),
                        **extra_info}
        for k, z in zip(plant_keys, zp):
            outputs[k][name] = z
        print(name, "val acc %.3f test acc %.3f test ECE(cal) %.3f" % (
            report[name]["val"]["mean_acc_all"],
            report[name]["test_cal"]["mean_acc_all"],
            report[name]["test_cal"]["mean_ece"]), flush=True)

    zv, zt, zp, _ = tfidf_expert(tr, va, te, plant_texts)
    finish("tfidf-lr", zv, zt, zp, {})

    if not args.skip_train:
        from causlm.probe.slm_expert import train_lora
        info = train_lora(BASE_05, tr[0], tr[1], str(LORA_DIR),
                          epochs=args.epochs)
        (CP_RESULTS / "lora_train_info.json").write_text(json.dumps(info, indent=1))
        report["lora_train"] = info
    (zv, zt, zp), dt = lm_expert(BASE_05, str(LORA_DIR),
                                 [va[0], te[0], plant_texts])
    finish("slm-0.5b-lora", zv, zt, zp, {"score_s": dt})
    (zv, zt, zp), dt = lm_expert(BASE_15, None, [va[0], te[0], plant_texts],
                                 batch=16)
    finish("zeroshot-1.5b", zv, zt, zp, {"score_s": dt})
    (zv, zt, zp), dt = lm_expert(BASE_05, None, [va[0], te[0], plant_texts])
    finish("zeroshot-0.5b", zv, zt, zp, {"score_s": dt})

    (CP_RESULTS / "expert_eval.json").write_text(json.dumps(report, indent=1))
    (CP_RESULTS / "expert_outputs.json").write_text(json.dumps(
        {"keys": plant_keys, "truth": [list(c) for c in plant_truth],
         "texts": plant_texts, "outputs": outputs}, indent=1))


if __name__ == "__main__":
    main()
