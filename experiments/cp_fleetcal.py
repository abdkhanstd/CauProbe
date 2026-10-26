"""Fleet calibration of the text expert from interventional labels (offline).

At a new site nobody has labelled notes in that site's vocabulary, so the
expert's temperature is fitted on training-vocabulary notes and is
miscalibrated under the shift. But a commissioning run ends with a posterior
over the mechanisms, learned from the plant's response to the gateway's own
actions: an *interventional label* for the questions the expert was asked.
Across a fleet, those labels can recalibrate the expert for the deployment
vocabulary -- if they are right.

Protocol (prequential): sites arrive in a fixed order -- the 48 validation
plants, then the 48 first-test plants, then the 48 second-test plants. Before
site i is commissioned, the per-factor temperature is refitted on the
(logit, label) pairs of every earlier site, with labels taken from the final
posterior of one controller's run there (run 0); the calibrated probabilities
for site i are scored against its *true* mechanisms. Scores are reported over
the 48 second-test sites. Label sources: the true mechanisms (oracle), and
the final posteriors of CauProbe-1, CauProbe-2 (where it ran), greedy control
on the same belief, and expert-plus-greedy control. Reference points: the
temperature fitted on training-vocabulary notes (what the controllers use)
and the raw logits.

Writes results/cauprobe/fleetcal.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, HELDOUT_PLANTS, TEST2_PLANTS, TEST_PLANTS
from causlm.probe.expert import (FACTORS, ExpertModel, expected_calibration_error,
                                 fit_temperature)

EXP = "slm-0.5b-lora"
SOURCES = ["oracle", "cauprobe-v2", "cauprobe", "greedy", "cauprobe-noprobe"]


def logits(notes):
    z = {}
    files = {"nominal": ["expert_outputs_nominal.json", "expert_outputs_test2_nominal.json"],
             "poor_docs": ["expert_outputs_poor_docs.json",
                           "expert_outputs_test2_poor_docs.json"]}[notes]
    for f in files:
        d = json.loads((CP_RESULTS / f).read_text())
        for k in d["keys"]:
            if k.startswith("eval:") and EXP in d["outputs"][k]:
                z[int(k[5:])] = d["outputs"][k][EXP]
    return z


def run_files(notes):
    if notes == "nominal":
        return [CP_RESULTS / "validation_v1" / "e1_main.jsonl",
                CP_RESULTS / "validation_v2" / "e1_nominal.jsonl",
                CP_RESULTS / "e1_test_main.jsonl",
                CP_RESULTS / "test2" / "e1_nominal.jsonl"]
    return [CP_RESULTS / "e1_test_poor_docs.jsonl",
            CP_RESULTS / "test2" / "e1_poor_docs.jsonl"]


def labels(notes):
    """site -> source -> {F: label}, from run 0 of each controller."""
    lab = defaultdict(dict)
    for p in run_files(notes):
        if not p.exists():
            continue
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if "error" in r or r["seed"] % 10 != 0:
                continue
            site = r["seed"] // 10
            c = r["combo"]
            lab[site]["oracle"] = {F: int(c[i]) for i, F in enumerate(FACTORS)}
            fp = r.get("final_factor_post")
            if r["method"] in SOURCES and fp:
                y = {F: int(fp[F] > 0.5) for F in "ABC"}
                y["D"] = int(r["final_mass_post"] < 0.5)      # mass_post = P(heavy)
                lab[site][r["method"]] = y
    return lab


def score(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return {"nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece": expected_calibration_error(p, y),
            "overconf_err": float(np.mean((np.abs(p - 0.5) > 0.4) & ((p > 0.5) != (y > 0.5))))}


def analyse(notes):
    z = logits(notes)
    lab = labels(notes)
    order = ([s for s, _ in HELDOUT_PLANTS] if notes == "nominal" else []) \
        + [s for s, _ in TEST_PLANTS] + [s for s, _ in TEST2_PLANTS]
    order = [s for s in order if s in z and s in lab]
    evalset = [s for s, _ in TEST2_PLANTS if s in z and s in lab]
    base = ExpertModel.from_json(CP_RESULTS / "experts" / f"{EXP}.json")
    out = {"n_sites": len(order), "n_eval": len(evalset)}
    # label quality per source over every site, and how often a label repeats
    # the expert's own mistake ("echo")
    q = {}
    for src in SOURCES[1:]:
        err, echo, nwrong = [], 0, 0
        for s in order:
            if src not in lab[s]:
                continue
            for F in FACTORS:
                t, yl = lab[s]["oracle"][F], lab[s][src][F]
                err.append(yl != t)
                if (z[s][F] > 0) != bool(t):
                    nwrong += 1
                    echo += yl != t
        q[src] = {"n_labels": len(err), "label_error": float(np.mean(err)) if err else None,
                  "expert_wrong": nwrong, "echo": echo,
                  "echo_rate": echo / nwrong if nwrong else None}
    out["label_quality"] = q

    def probs(tau, s):
        return [1 / (1 + np.exp(-tau[F] * z[s][F])) for F in FACTORS]

    truth = lambda s: [lab[s]["oracle"][F] for F in FACTORS]
    res = {}
    res["train-vocab"] = score(sum((probs(base.tau, s) for s in evalset), []),
                               sum((truth(s) for s in evalset), []))
    res["raw"] = score(sum((probs({F: 1.0 for F in FACTORS}, s) for s in evalset), []),
                       sum((truth(s) for s in evalset), []))
    curves = {}
    for src in SOURCES:
        P, Y, taus, curve = [], [], [], []
        for s in evalset:
            hist = [h for h in order[:order.index(s)] if src in lab[h]]
            if len(hist) < 8:
                continue
            tau = {F: fit_temperature([z[h][F] for h in hist],
                                      [lab[h][src][F] for h in hist]) for F in FACTORS}
            taus.append(tau)
            P += probs(tau, s)
            Y += truth(s)
            curve.append(len(hist))
        if P:
            res[f"fleet:{src}"] = score(P, Y)
            res[f"fleet:{src}"]["n_eval_sites"] = len(curve)
            res[f"fleet:{src}"]["tau_final"] = taus[-1]
            res[f"fleet:{src}"]["fleet_size_range"] = (min(curve), max(curve))
    out["calibration"] = res
    # calibration as a function of fleet size (oracle and CauProbe-1 labels):
    # fit on the first n sites of the order, score on all second-test sites
    for src in ("oracle", "cauprobe", "greedy", "cauprobe-noprobe"):
        pts = []
        hist_all = [h for h in order if h not in evalset and src in lab[h]]
        for n in (4, 8, 16, 32, 64, 96):
            if n > len(hist_all):
                break
            hist = hist_all[:n]
            tau = {F: fit_temperature([z[h][F] for h in hist],
                                      [lab[h][src][F] for h in hist]) for F in FACTORS}
            sc = score(sum((probs(tau, s) for s in evalset), []),
                       sum((truth(s) for s in evalset), []))
            pts.append({"n": n, **sc})
        curves[src] = pts
    out["by_fleet_size"] = curves
    return out


def ool_sites():
    """Out-of-library sites: logits of their notes and labels from CauProbe's
    final posterior there, with the run's misfit-alarm count and the smallest
    factor confidence |p - 1/2|."""
    z, lab = {}, {}
    for f in ("expert_outputs_ool_test.json", "expert_outputs_ool_test2.json"):
        d = json.loads((CP_RESULTS / f).read_text())
        for k in d["keys"]:
            z[int(k[4:])] = d["outputs"][k][EXP]
    for p, src in ((CP_RESULTS / "e1_test_ool.jsonl", "cauprobe"),
                   (CP_RESULTS / "test2" / "e1_ool.jsonl", "cauprobe-v2")):
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if "error" in r or r["method"] != src or r["seed"] % 10 != 0:
                continue
            fp = r["final_factor_post"]
            y = {F: int(fp[F] > 0.5) for F in "ABC"}
            y["D"] = int(r["final_mass_post"] < 0.5)
            conf = min([abs(fp[F] - 0.5) for F in "ABC"] + [abs(r["final_mass_post"] - 0.5)])
            lab[r["seed"] // 10] = {"label": y, "truth": dict(zip(FACTORS, r["combo"])),
                                    "ool": r["ool"], "alarms": r.get("misspec_alarms", 0),
                                    "conf": conf}
    return z, lab


def contamination():
    """Fleet calibration when sites outside the library join the fleet, with
    and without an admissibility gate, scored on the second-test sites."""
    z, lab = logits("nominal"), labels("nominal")
    zo, lo = ool_sites()
    hist = [s for s, _ in HELDOUT_PLANTS] + [s for s, _ in TEST_PLANTS]
    hist = [s for s in hist if s in z and s in lab and "cauprobe" in lab[s]]
    evalset = [s for s, _ in TEST2_PLANTS if s in z and s in lab]
    truth = lambda s: [lab[s]["oracle"][F] for F in FACTORS]
    out = {}
    gates = {"in-library fleet only": None,
             "with out-of-library sites, no gate": lambda o: True,
             "gate: no misfit alarm": lambda o: o["alarms"] == 0,
             "gate: confident posterior (all |p-1/2| > 0.45)": lambda o: o["conf"] > 0.45,
             "gate: both": lambda o: o["alarms"] == 0 and o["conf"] > 0.45}
    for name, g in gates.items():
        Z = {F: [z[s][F] for s in hist] for F in FACTORS}
        Y = {F: [lab[s]["cauprobe"][F] for s in hist] for F in FACTORS}
        added, wrong = 0, 0
        if g is not None:
            for s, o in lo.items():
                if s in zo and g(o):
                    added += 1
                    for F in FACTORS:
                        Z[F].append(zo[s][F])
                        Y[F].append(o["label"][F])
                        wrong += o["label"][F] != o["truth"][F]
        tau = {F: fit_temperature(Z[F], Y[F]) for F in FACTORS}
        P = sum(([1 / (1 + np.exp(-tau[F] * z[s][F])) for F in FACTORS] for s in evalset), [])
        sc = score(P, sum((truth(s) for s in evalset), []))
        out[name] = {**sc, "ool_sites_added": added,
                     "ool_label_error": wrong / (4 * added) if added else None,
                     "tau": tau}
    # how wrong are out-of-library labels, by kind and factor
    per = {}
    for s, o in lo.items():
        for F in FACTORS:
            per.setdefault(o["ool"], {}).setdefault(F, []).append(o["label"][F] != o["truth"][F])
    out["ool_label_error_by_kind"] = {k: {F: float(np.mean(v)) for F, v in d.items()}
                                      for k, d in per.items()}
    out["ool_alarm_rate_by_kind"] = {k: float(np.mean([o["alarms"] > 0 for o in lo.values()
                                                       if o["ool"] == k])) for k in per}
    out["ool_confident_rate_by_kind"] = {k: float(np.mean([o["conf"] > 0.45 for o in lo.values()
                                                           if o["ool"] == k])) for k in per}
    return out


def main():
    import sys
    if "--contamination" in sys.argv:
        c = contamination()
        (CP_RESULTS / "fleetcal_contamination.json").write_text(json.dumps(c, indent=1, default=float))
        for k, v in c.items():
            print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()
                      if a != "tau"} if isinstance(v, dict) else v)
        return
    out = {nt: analyse(nt) for nt in ("nominal", "poor_docs")}
    (CP_RESULTS / "fleetcal.json").write_text(json.dumps(out, indent=1, default=float))
    for nt, o in out.items():
        print("==", nt, "sites", o["n_sites"], "eval", o["n_eval"])
        for k, v in o["label_quality"].items():
            print("  labels", k, v)
        for k, v in o["calibration"].items():
            print("  ", k, {a: (round(b, 4) if isinstance(b, float) else b)
                           for a, b in v.items() if a != "tau_final"},
                  {F: round(t, 2) for F, t in v.get("tau_final", {}).items()})
        for k, v in o["by_fleet_size"].items():
            print("  curve", k, [(p["n"], round(p["ece"], 3)) for p in v])


if __name__ == "__main__":
    main()
