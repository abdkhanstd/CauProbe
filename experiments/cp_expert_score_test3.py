"""Score the commissioning notes of the third (or, with argument test4, the
fourth) fresh test set.

The third test set (cp_common.TEST3_PLANTS, seeds 920_000+) was created after
the fleet-calibrated expert (cp_build_fleet_expert.py) had been frozen. Its
notes are generated with exactly the recipe of the first two test sets
(nominal: random record quality, 8% stale; poor_docs: quality 0.25, 15%
stale) and read by the same trained 0.5B adapter; what differs between the
methods run on it is only how the adapter's output is calibrated. Writes
results/cauprobe/expert_outputs_test3_{nominal,poor_docs}.json (keys
eval:<seed>, ticket:<seed>), which cp_e1_commission merges in.
"""
from __future__ import annotations

import json

import numpy as np

import sys

from cp_common import BASE_05, CP_RESULTS, LORA_DIR, TEST3_PLANTS, TEST4_PLANTS
from causlm.probe.expert import FACTORS
from causlm.probe.notes import make_notes, make_ticket

CONFIGS = {"nominal": {"quality": None, "p_stale": 0.08},
           "poor_docs": {"quality": 0.25, "p_stale": 0.15}}
EX = "slm-0.5b-lora"


def main(tag="test3"):
    plants = {"test3": TEST3_PLANTS, "test4": TEST4_PLANTS}[tag]
    from causlm.probe.slm_expert import load_model, score_texts
    tok, model = load_model(BASE_05, str(LORA_DIR))
    for name, cfg in CONFIGS.items():
        keys, texts, truth = [], [], []
        for seed, combo in plants:
            rng = np.random.default_rng(seed + 13)
            t, _ = make_notes(combo, rng, split="test", **cfg)
            keys.append(f"eval:{seed}")
            texts.append(t)
            truth.append(list(combo))
            rng2 = np.random.default_rng(seed + 29)
            keys.append(f"ticket:{seed}")
            texts.append(make_ticket("A", 1 - combo[0], rng2))
            truth.append([1 - combo[0]] + list(combo[1:]))
        out = {k: {EX: z} for k, z in zip(keys, score_texts(tok, model, texts))}
        path = CP_RESULTS / f"expert_outputs_{tag}_{name}.json"
        path.write_text(json.dumps({"keys": keys, "truth": truth, "texts": texts,
                                    "outputs": out, "config": cfg}, indent=1))
        acc = [np.mean([(out[k][EX][F] > 0) == bool(c[i])
                        for i, F in enumerate(FACTORS)])
               for k, c in zip(keys, truth) if k.startswith("eval:")]
        print(f"{tag} {name}: notes acc {np.mean(acc):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test3")
