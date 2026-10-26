"""Build the fleet-calibrated text expert from interventional labels.

The fleet is every site commissioned before the third test set: the 48
validation plants, the 48 first-test plants and the 48 second-test plants.
For each site and each version of its notes that a controller read (nominal
notes at all sites, poor documentation at the first- and second-test sites),
the label of each mechanism factor is the side of one-half on which the final
posterior of CauProbe's run there ended (CauProbe-2 where it ran, otherwise
CauProbe-1; run 0) -- never the true mechanism. From those (logit, label)
pairs the per-factor temperature is refitted by maximum likelihood, and the
logit bank that values a reading before it is made is replaced by the fleet's
logits grouped by label. The adapter itself is unchanged.

Writes results/cauprobe/experts/slm-0.5b-lora-fleet.json and, for the record,
results/cauprobe/fleet_expert_provenance.json (sizes, temperatures, and the
labels' error rate against the true mechanisms, which the fit does not see).
"""
from __future__ import annotations

import json

import numpy as np

from cp_common import CP_RESULTS, HELDOUT_PLANTS, TEST2_PLANTS, TEST_PLANTS
from cp_fleetcal import EXP, labels, logits
from causlm.probe.expert import FACTORS, ExpertModel, fit_temperature


def main():
    sites = ([s for s, _ in HELDOUT_PLANTS] + [s for s, _ in TEST_PLANTS]
             + [s for s, _ in TEST2_PLANTS])
    Z = {F: [] for F in FACTORS}
    Y = {F: [] for F in FACTORS}
    err = {F: [] for F in FACTORS}
    used = {}
    for notes in ("nominal", "poor_docs"):
        z, lab = logits(notes), labels(notes)
        n = 0
        for s in sites:
            if s not in z or s not in lab:
                continue
            src = "cauprobe-v2" if "cauprobe-v2" in lab[s] else "cauprobe"
            if src not in lab[s]:
                continue
            n += 1
            for F in FACTORS:
                Z[F].append(float(z[s][F]))
                Y[F].append(lab[s][src][F])
                err[F].append(lab[s][src][F] != lab[s]["oracle"][F])
        used[notes] = n
    tau, bank = {}, {}
    for F in FACTORS:
        zz, yy = np.asarray(Z[F]), np.asarray(Y[F], int)
        tau[F] = fit_temperature(zz, yy)
        bank[F] = {v: zz[yy == v] for v in (0, 1)}
    ex = ExpertModel(EXP + "-fleet", tau, bank)
    ex.to_json(CP_RESULTS / "experts" / f"{EXP}-fleet.json")
    base = ExpertModel.from_json(CP_RESULTS / "experts" / f"{EXP}.json")
    prov = {"sites_per_notes": used, "pairs_per_factor": {F: len(Z[F]) for F in FACTORS},
            "tau_fleet": tau, "tau_train_vocab": base.tau,
            "bank_sizes": {F: {v: int(len(bank[F][v])) for v in (0, 1)} for F in FACTORS},
            "label_error_vs_truth": {F: float(np.mean(err[F])) for F in FACTORS}}
    (CP_RESULTS / "fleet_expert_provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
