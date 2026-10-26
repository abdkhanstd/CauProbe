"""How reliable is the text expert's evidence, and how should it enter?

On the 2000 held-out test-vocabulary notes of cp_expert.py (seeds 3_000_000+)
and the 1500 training-vocabulary validation notes (2_000_000+), re-scored with
the trained 0.5B adapter:

  reliability   per factor, binned predicted probability vs observed
                frequency, with the training-vocabulary temperature; also by
                what the note says about the factor (mentioned correctly,
                stale/opposite, not mentioned) and by record quality
  correlation   correlation of the four answers' errors on the same note --
                the agent multiplies the four likelihoods as if independent
  sensitivity   NLL and ECE as the temperature is scaled by 1/4 .. 2
  likelihood    temperature scaling (a likelihood ratio exp(tau z)) against
                an empirical class-conditional likelihood: Gaussian kernel
                density of the validation logits given the factor value
                (training vocabulary), and of the fleet's logits given their
                interventional labels (cp_build_fleet_expert.py)

Writes results/cauprobe/expert_reliability.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import gaussian_kde

from cp_common import BASE_05, CP_RESULTS, LORA_DIR
from cp_expert import dataset
from causlm.probe.expert import FACTORS, ExpertModel, expected_calibration_error

EX = "slm-0.5b-lora"


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def qualities(n, seed0):
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        rng.integers(16)
        out.append(float(rng.choice([0.25, 0.6, 0.9])))
    return out


def main():
    from causlm.probe.slm_expert import load_model, score_texts
    tok, model = load_model(BASE_05, str(LORA_DIR))
    va = dataset(1500, 2_000_000, "train")
    te = dataset(2000, 3_000_000, "test")
    zv = score_texts(tok, model, va[0])
    zt = score_texts(tok, model, te[0])
    qt = qualities(2000, 3_000_000)
    base = ExpertModel.from_json(CP_RESULTS / "experts" / f"{EX}.json")
    fleet = ExpertModel.from_json(CP_RESULTS / "experts" / f"{EX}-fleet.json")
    out = {"n_test": len(zt), "n_val": len(zv)}
    rel, bycov, byq, sens, lik = {}, {}, {}, {}, {}
    E = np.zeros((len(zt), 4), bool)
    for fi, F in enumerate(FACTORS):
        z = np.array([d[F] for d in zt])
        y = np.array([c[fi] for c in te[1]], float)
        cov = np.array([c[F] for c in te[2]])
        E[:, fi] = (z > 0) != (y > 0)
        p = 1 / (1 + np.exp(-base.tau[F] * z))
        edges = np.linspace(0, 1, 11)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if m.sum() >= 5:
                bins.append({"p_mean": float(p[m].mean()), "freq": float(y[m].mean()),
                             "n": int(m.sum())})
        rel[F] = {"bins": bins, "ece": expected_calibration_error(p, y), "nll": nll(p, y)}
        bycov[F] = {}
        for name, v in (("mentioned", 1), ("stale", -1), ("absent", 0)):
            m = cov == v
            if m.sum():
                bycov[F][name] = {"n": int(m.sum()), "acc": float(((p[m] > 0.5) == (y[m] > 0.5)).mean()),
                                  "mean_conf": float(np.maximum(p[m], 1 - p[m]).mean()),
                                  "ece": expected_calibration_error(p[m], y[m])}
        byq[F] = {}
        for q in (0.25, 0.6, 0.9):
            m = np.array(qt) == q
            byq[F][str(q)] = {"n": int(m.sum()), "acc": float(((p[m] > 0.5) == (y[m] > 0.5)).mean()),
                              "ece": expected_calibration_error(p[m], y[m])}
        sens[F] = {}
        for k in (0.25, 0.5, 1.0, 2.0):
            pk = 1 / (1 + np.exp(-k * base.tau[F] * z))
            sens[F][str(k)] = {"nll": nll(pk, y), "ece": expected_calibration_error(pk, y)}
        # empirical class-conditional likelihoods
        zvF = np.array([d[F] for d in zv])
        yvF = np.array([c[fi] for c in va[1]])
        res = {"temperature_trainvocab": {"nll": nll(p, y),
                                          "ece": expected_calibration_error(p, y)}}
        for tag, z0, z1 in (("kde_trainvocab", zvF[yvF == 0], zvF[yvF == 1]),
                            ("kde_fleet", fleet.bank[F][0], fleet.bank[F][1])):
            k0, k1 = gaussian_kde(z0), gaussian_kde(z1)
            l0, l1 = np.log(k0(z) + 1e-12), np.log(k1(z) + 1e-12)
            pk = 1 / (1 + np.exp(-(l1 - l0)))
            res[tag] = {"nll": nll(pk, y), "ece": expected_calibration_error(pk, y)}
        pf = 1 / (1 + np.exp(-fleet.tau[F] * z))
        res["temperature_fleet"] = {"nll": nll(pf, y), "ece": expected_calibration_error(pf, y)}
        lik[F] = res
    C = np.corrcoef(E.T.astype(float))
    out.update({"reliability": rel, "by_coverage": bycov, "by_quality": byq,
                "temperature_sensitivity": sens, "likelihood": lik,
                "error_corr": C.tolist(),
                "error_rate": E.mean(0).tolist(),
                "p_two_or_more_errors": float((E.sum(1) >= 2).mean()),
                "p_two_or_more_if_independent": float(
                    1 - np.prod(1 - E.mean(0))
                    - sum(E.mean(0)[i] * np.prod(np.delete(1 - E.mean(0), i)) for i in range(4)))})
    (CP_RESULTS / "expert_reliability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("reliability",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
