"""Score the commissioning notes of the fresh out-of-library test plants.

The notes are generated exactly as for every other evaluation plant
(cp_expert.plant_notes: test vocabulary, random record quality, 8% stale
statements); the out-of-library mechanism itself is not mentioned, because the
library has no word for it. Writes results/cauprobe/expert_outputs_ool_test.json
with keys "ool:<seed>", which cp_e1_commission merges into the nominal outputs.
"""
from __future__ import annotations

import json

import numpy as np

from cp_common import BASE_05, CP_RESULTS, LORA_DIR, OOL_TEST_PLANTS
from cp_expert import dataset, plant_notes
from causlm.probe.expert import FACTORS


def main(n_train=12000):
    keys, texts, truth = [], [], []
    for seed, combo, _ in OOL_TEST_PLANTS:
        keys.append(f"ool:{seed}")
        texts.append(plant_notes(seed, combo)[0])
        truth.append(list(combo))
    out = {k: {} for k in keys}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = dataset(n_train, 1_000_000, "train")
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
    path = CP_RESULTS / "expert_outputs_ool_test.json"
    path.write_text(json.dumps({"keys": keys, "truth": truth, "texts": texts,
                                "outputs": out}, indent=1))
    acc = [np.mean([(out[k]["slm-0.5b-lora"][F] > 0) == bool(c[i])
                    for i, F in enumerate(FACTORS)]) for k, c in zip(keys, truth)]
    print(f"wrote {path}; slm accuracy on these notes {np.mean(acc):.3f}")


if __name__ == "__main__":
    main()
