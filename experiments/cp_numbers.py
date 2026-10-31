"""Every number the results text quotes that is not read off a table.

Reads results/cauprobe/summary.json (written by cp_make_tables.py) and the raw
test-set rows, prints the values and writes results/cauprobe/numbers.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, testrows

STEPS = 14 * 288


def rows_of(tag):
    p = CP_RESULTS / f"e1_test_{tag}.jsonl"
    if not p.exists():
        return []
    return testrows([json.loads(l) for l in p.read_text().splitlines() if l.strip()])


def wrong_factors(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    return [F for i, F in enumerate("ABC") if (fp[F] > 0.5) != bool(r["combo"][i])]


def main():
    S = json.loads((CP_RESULTS / "summary.json").read_text())
    st = S["main"]
    g = lambda m: st[m]["gap"][0]
    vs = lambda m: st[m]["vs_ref"]["diff"][0]
    ph = lambda m: st[m]["vs_ref"]["p_holm"]
    N = {}
    bel = ["cauprobe", "cauprobe-noask", "cauprobe-noprobe", "greedy", "thompson",
           "ids", "identify"]
    prac = ["rule", "linucb", "advice-plan", "nominal", "fallback"]
    N["belief_gap_range"] = (min(g(m) for m in bel), max(g(m) for m in bel))
    N["practical_gap_range"] = (min(g(m) for m in prac), max(g(m) for m in prac))
    for m in ["advice-plan", "linucb", "cauprobe", "greedy", "thompson", "ids",
              "identify", "rule", "nominal", "fallback", "log-naive",
              "cauprobe+logs-naive", "cauprobe-noprobe", "cauprobe-noask"]:
        N[f"gap:{m}"] = g(m)
    for m in st:
        if "vs_ref" in st[m]:
            N[f"vs:{m}"] = (vs(m), ph(m))
        N[f"probes:{m}"] = st[m]["probes"][0]
        N[f"reconf:{m}"] = st[m]["switches"][0]
        N[f"misses:{m}"] = st[m]["misses"][0]
        N[f"cvar:{m}"] = st[m].get("cvar10")
        if "wrong" in st[m]:
            N[f"wrong:{m}"] = st[m]["wrong"][0]
        N[f"day1:{m}"] = st[m]["day1"][0]
        N[f"gap_th:{m}"] = st[m]["gap_th"][0]
        N[f"gap_net:{m}"] = st[m]["gap_net"][0]
    N["particle_probe_pct"] = 100 * st["cauprobe-particle"]["probes"][0] / STEPS
    N["cauprobe_probe_pct"] = 100 * st["cauprobe"]["probes"][0] / STEPS
    N["belief_miss_range"] = (min(st[m]["misses"][0] for m in bel),
                              max(st[m]["misses"][0] for m in bel))

    # lock-in analysis on the raw runs
    rows = rows_of("main")
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["seed"]] = r
    resc, over = [], []
    for s, gr in by["greedy"].items():
        w = wrong_factors(gr)
        cp = by["cauprobe"].get(s)
        if not w or cp is None:
            continue
        rec = {"seed": s, "plant": gr["plant"], "wrong": w,
               "cp_minus_greedy": cp["return"] - gr["return"],
               "cp_probes": cp["probe_steps"], "cp_wrong": wrong_factors(cp)}
        (resc if gr["combo"][1] == 0 else over).append(rec)
    N["lockin_edge_bound"] = {
        "n": len(resc),
        "gain_mean": float(np.mean([x["cp_minus_greedy"] for x in resc])) if resc else None,
        "gain_range": (min(x["cp_minus_greedy"] for x in resc),
                       max(x["cp_minus_greedy"] for x in resc)) if resc else None,
        "probes_range": (min(x["cp_probes"] for x in resc),
                         max(x["cp_probes"] for x in resc)) if resc else None}
    N["lockin_wan_bound"] = {
        "n": len(over),
        "loss_mean": float(-np.mean([x["cp_minus_greedy"] for x in over])) if over else None,
        "probes_range": (min(x["cp_probes"] for x in over),
                         max(x["cp_probes"] for x in over)) if over else None,
        "probes_mean": float(np.mean([x["cp_probes"] for x in over])) if over else None}
    N["greedy_wrong_factors"] = sorted({f for x in resc + over for f in x["wrong"]})
    N["n_runs_greedy"] = len(by["greedy"])
    # expert + greedy: which greedy lock-ins does the expert avoid?
    npw = [s for s in by["cauprobe-noprobe"] if wrong_factors(by["cauprobe-noprobe"][s])]
    N["noprobe_wrong_runs"] = len(npw)
    N["greedy_wrong_runs"] = len(resc) + len(over)
    # per-plant extremes, CauProbe vs greedy
    pp = defaultdict(list)
    for s, c in by["cauprobe"].items():
        gr = by["greedy"].get(s)
        if gr is not None:
            pp[s // 10].append(c["return"] - gr["return"])
    d = np.array([np.mean(v) for v in pp.values()])
    N["cp_vs_greedy_per_plant"] = {"max_gain": float(d.max()), "max_loss": float(-d.min()),
                                   "n_better": int((d > 0).sum()),
                                   "n_worse": int((d < 0).sum()), "n": len(d)}
    # paired tests of the tail claims against each belief-based rival
    from math import comb
    orc = by["oracle"]
    for rival in ["greedy", "cauprobe-noprobe", "thompson", "log-naive"]:
        # exact McNemar on wrong-mechanism runs, paired by plant and noise
        b = c_ = 0
        for s, cp in by["cauprobe"].items():
            rv = by[rival].get(s)
            if rv is None:
                continue
            wc, wr = bool(wrong_factors(cp)), bool(wrong_factors(rv))
            b += wr and not wc
            c_ += wc and not wr
        n = b + c_
        p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c_) + 1)) / 2 ** n) if n else 1.0
        # the same comparison with plants as the unit (a plant's runs are not
        # independent): plants where the rival ends more runs on a wrong
        # mechanism than CauProbe, and fewer, with an exact two-sided sign test
        wp = defaultdict(lambda: [0, 0])
        for s, cp in by["cauprobe"].items():
            rv = by[rival].get(s)
            if rv is not None:
                wp[s // 10][0] += bool(wrong_factors(cp))
                wp[s // 10][1] += bool(wrong_factors(rv))
        more = sum(1 for v in wp.values() if v[1] > v[0])
        fewer = sum(1 for v in wp.values() if v[1] < v[0])
        ns = more + fewer
        sp = min(1.0, 2 * sum(comb(ns, k) for k in range(0, min(more, fewer) + 1))
                 / 2 ** ns) if ns else 1.0
        # bootstrap CI for the CVaR10 difference of per-plant gaps (paired plants)
        gp = defaultdict(lambda: defaultdict(list))
        for m in ("cauprobe", rival):
            for s, r in by[m].items():
                if s in orc:
                    gp[m][s // 10].append(orc[s]["return"] - r["return"])
        plants = sorted(set(gp["cauprobe"]) & set(gp[rival]))
        A = np.array([np.mean(gp["cauprobe"][k]) for k in plants])
        B = np.array([np.mean(gp[rival][k]) for k in plants])

        def cv(x):
            x = np.sort(x)[::-1]
            return x[:max(1, int(round(0.1 * len(x))))].mean()

        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(plants), (10000, len(plants)))
        diffs = np.array([cv(B[i]) - cv(A[i]) for i in idx])
        N[f"tail_vs:{rival}"] = {"wrong_discordant": (int(b), int(c_)), "mcnemar_p": p,
                                 "plants_rival_more_fewer": (more, fewer),
                                 "n_plants": len(wp), "sign_p_plants": sp,
                                 "cvar10_rival_minus_cp": float(cv(B) - cv(A)),
                                 "cvar10_ci": (float(np.percentile(diffs, 2.5)),
                                               float(np.percentile(diffs, 97.5))),
                                 "p_cvar_le0": float((diffs <= 0).mean())}
    # scenarios: value of the expert without probing (greedy - noprobe gap),
    # value of probing on top of the expert (noprobe - cauprobe), paired by plant
    from cp_summarize import boot_ci, paired_p
    for sc in ["main", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]:
        rr = rows_of(sc)
        if not rr:
            continue
        b2 = defaultdict(dict)
        for r in rr:
            b2[r["method"]][r["seed"]] = r

        def pdiff(m1, m2):
            acc = defaultdict(list)
            for s, r in b2[m1].items():
                if s in b2[m2]:
                    acc[s // 10].append(r["return"] - b2[m2][s]["return"])
            d = np.array([np.mean(v) for v in acc.values()])
            return {"ci": boot_ci(d), "p": paired_p(d)}

        def wrong_rate(m):
            v = []
            for r in b2[m].values():
                fp = r.get("final_factor_post")
                if not fp:
                    continue
                c = list(r["combo"])
                if sc == "shift":
                    c[0] = 1 - c[0]
                v.append(any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC")))
            return float(np.mean(v)) if v else None

        N[f"scen:{sc}"] = {
            "expert_value_greedy": pdiff("cauprobe-noprobe", "greedy"),
            "expert_value_cauprobe": pdiff("cauprobe", "cauprobe-noask"),
            "probe_value_given_expert": pdiff("cauprobe", "cauprobe-noprobe"),
            "probe_value_no_expert": pdiff("cauprobe-noask", "greedy"),
            "wrong": {m: wrong_rate(m) for m in ("cauprobe", "cauprobe-noprobe",
                                                  "greedy", "thompson")},
            "probes": float(np.mean([r["probe_steps"] for r in b2["cauprobe"].values()])),
            "switch_cost_cp": float(np.mean([r["switch_cost"] for r in b2["cauprobe"].values()])),
            "switch_cost_greedy": float(np.mean([r["switch_cost"] for r in b2["greedy"].values()])),
        }
        if sc == "shift":
            tk = {}
            for m in ("cauprobe", "cauprobe-noask", "cauprobe-noprobe", "greedy"):
                for s, r in b2[m].items():
                    o = b2["oracle"].get(s)
                    if o is None:
                        continue
                    late = sum(o["return_per_day"][3:]) - sum(r["return_per_day"][3:])
                    tk.setdefault(m, {}).setdefault(
                        "ticket" if r.get("ticket") else "no_ticket", []).append(late)
            N["shift_after_change"] = {m: {k: float(np.mean(v)) for k, v in t.items()}
                                       for m, t in tk.items()}
            # value of the ticket for the expert-using agents: same plants,
            # with vs without the expert (ticket plants only)
            # (ticket plants are the even seeds, so compare the expert's
            # after-change value on ticket plants with that on the others)
            for a, b in (("cauprobe", "cauprobe-noask"), ("cauprobe-noprobe", "greedy")):
                for flag in (True, False):
                    d = [(sum(b2[a][s]["return_per_day"][3:])
                          - sum(b2[b][s]["return_per_day"][3:]))
                         for s in b2[a] if s in b2[b] and bool(b2[a][s].get("ticket")) == flag]
                    N[f"shift_expert_value_after:{a}-vs-{b}:{'ticket' if flag else 'none'}"] = \
                        float(np.mean(d)) if d else None
    # second iteration (cp_iter2.py): second fresh test set, and the
    # validation runs it was frozen on
    import cp_iter2 as I2
    N["iter2"] = I2.numbers()
    N["iter2_val"] = {k: v for k, v in I2.numbers(val=True).items() if k != "price"}
    (CP_RESULTS / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
    for k, v in N.items():
        print(k, v)


if __name__ == "__main__":
    main()
