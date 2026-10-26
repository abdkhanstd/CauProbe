"""Score the commissioning notes of the second fresh test set (iteration 2).

The second test set (cp_common.TEST2_PLANTS, seeds 910_000+, and
OOL_TEST2_PLANTS, 960_000+) was created after the second iteration of the
method was frozen. Its notes are generated with exactly the recipe of the
first test set:

  nominal    cp_expert_score.py --name nominal   (random record quality, 8% stale)
  poor_docs  cp_expert_score.py --name poor_docs (quality 0.25, 15% stale)
  tickets    the radio-swap ticket of the structural-change scenario
  ool        cp_expert_score_ool.py              (plant_notes: nominal recipe)

and read by the same trained 0.5B adapter. Only the adapter is scored: the
methods run on this set use no other expert. Writes
results/cauprobe/expert_outputs_test2_{nominal,poor_docs}.json (keys
eval:<seed>, ticket:<seed>) and expert_outputs_ool_test2.json (ool:<seed>),
which cp_e1_commission merges in.
"""
from __future__ import annotations

import json

import numpy as np

from cp_common import BASE_05, CP_RESULTS, LORA_DIR, OOL_TEST2_PLANTS, TEST2_PLANTS
from cp_expert import plant_notes
from causlm.probe.expert import FACTORS
from causlm.probe.notes import make_notes, make_ticket

CONFIGS = {"nominal": {"quality": None, "p_stale": 0.08},
           "poor_docs": {"quality": 0.25, "p_stale": 0.15}}
EX = "slm-0.5b-lora"


def main():
    from causlm.probe.slm_expert import load_model, score_texts
    tok, model = load_model(BASE_05, str(LORA_DIR))
    for name, cfg in CONFIGS.items():
        keys, texts, truth = [], [], []
        for seed, combo in TEST2_PLANTS:
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
        path = CP_RESULTS / f"expert_outputs_test2_{name}.json"
        path.write_text(json.dumps({"keys": keys, "truth": truth, "texts": texts,
                                    "outputs": out, "config": cfg}, indent=1))
        acc = [np.mean([(out[k][EX][F] > 0) == bool(c[i])
                        for i, F in enumerate(FACTORS)])
               for k, c in zip(keys, truth) if k.startswith("eval:")]
        tick = [(out[k][EX]["A"] > 0) == bool(c[0])
                for k, c in zip(keys, truth) if k.startswith("ticket:")]
        print(f"test2 {name}: notes acc {np.mean(acc):.3f}  ticket A acc {np.mean(tick):.3f}")
    keys, texts, truth = [], [], []
    for seed, combo, _ in OOL_TEST2_PLANTS:
        keys.append(f"ool:{seed}")
        texts.append(plant_notes(seed, combo)[0])
        truth.append(list(combo))
    out = {k: {EX: z} for k, z in zip(keys, score_texts(tok, model, texts))}
    path = CP_RESULTS / "expert_outputs_ool_test2.json"
    path.write_text(json.dumps({"keys": keys, "truth": truth, "texts": texts,
                                "outputs": out}, indent=1))
    acc = [np.mean([(out[k][EX][F] > 0) == bool(c[i]) for i, F in enumerate(FACTORS)])
           for k, c in zip(keys, truth)]
    print(f"test2 ool: notes acc {np.mean(acc):.3f}")


if __name__ == "__main__":
    main()
