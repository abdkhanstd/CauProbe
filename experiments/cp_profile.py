"""Embedded profile of the decision procedure: per-step compute and memory.

Runs each controller for 14 days on four development plants (nominal
scenario, one process per run, single-threaded BLAS) and records, for every
control step, the wall time of the controller's own work -- act() plus
observe() plus, on a reading, receive_expert() -- excluding the plant
simulator; and the process's peak resident set before the controller is
built and at the end of the run. The language model itself is not run here
(its outputs are cached); its cost is measured separately (cp_e8_edge.py).
Writes results/cauprobe/profile.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, EVAL_PLANTS
from causlm.probe import mech
from causlm.probe.run import AskCost, run_commissioning
from causlm.probe.plants import sample_plant

METHODS = ["greedy", "cauprobe", "cauprobe-v2", "dual-rollout"]
PLANTS = [EVAL_PLANTS[i] for i in (0, 5, 10, 15)]      # development plants


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def job(a):
    method, seed, combo = a
    mech.set_switch_costs(0.3, 0.05)
    E.NOTES = "nominal"
    base = rss_mib()
    spec = sample_plant(seed, tuple(combo))
    agent, z = E.make_agent(method, spec, f"eval:{seed}", seed * 10)
    times, cur = [], [0.0]

    def timed(f):
        def g(*x, **k):
            t0 = time.perf_counter()
            try:
                return f(*x, **k)
            finally:
                cur[0] += time.perf_counter() - t0
        return g

    act = timed(agent.act)

    def act_and_close(*x, **k):
        if cur[0] > 0.0:
            times.append(cur[0])
        cur[0] = 0.0
        return act(*x, **k)

    agent.act = act_and_close
    agent.observe = timed(agent.observe)
    if hasattr(agent, "receive_expert"):
        agent.receive_expert = timed(agent.receive_expert)
    t0 = time.perf_counter()
    out = run_commissioning(spec, agent, days=14, seed=seed * 10, expert_out=z,
                            ask=AskCost(energy_j=40.0, latency_ms=6000.0, price_per_kj=0.4))
    times.append(cur[0])
    wall = time.perf_counter() - t0
    t = np.array(times) * 1000.0
    return {"method": method, "seed": seed, "steps": len(t), "wall_s": wall,
            "ms_mean": float(t.mean()), "ms_median": float(np.median(t)),
            "ms_p99": float(np.percentile(t, 99)), "ms_max": float(t.max()),
            "step_of_max": int(t.argmax()), "ms_first": float(t[0]),
            "ms_max_after_first": float(t[1:].max()),
            "slowest_steps": [int(i) for i in np.argsort(t)[::-1][:5]],
            "share_over_1s": float(np.mean(t > 1000.0)),
            "rss_base_mib": base, "rss_peak_mib": rss_mib(),
            "return": out["return"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true", help="only rewrite the table from profile.json")
    a = ap.parse_args()
    if a.table:
        table(json.loads((CP_RESULTS / "profile.json").read_text())["summary"])
        return
    jobs = [(m, s, c) for m in METHODS for s, c in PLANTS]
    rows = []
    with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            print(r["method"], r["seed"], f"{r['ms_mean']:.1f} ms mean, {r['ms_max']:.0f} ms max, "
                  f"{r['rss_peak_mib']:.0f} MiB", flush=True)
    summ = {}
    for m in METHODS:
        rr = [r for r in rows if r["method"] == m]
        summ[m] = {k: float(np.mean([r[k] for r in rr])) for k in
                   ("ms_mean", "ms_median", "ms_p99", "share_over_1s", "rss_base_mib", "rss_peak_mib")}
        summ[m]["ms_max"] = float(max(r["ms_max"] for r in rr))
        summ[m]["ms_first_max"] = float(max(r["ms_first"] for r in rr))
        summ[m]["ms_max_after_first"] = float(max(r["ms_max_after_first"] for r in rr))
    out = {"host": {"cpu": platform.processor() or platform.machine(),
                    "python": platform.python_version()},
           "note": "single-threaded BLAS; host shared with other runs", "runs": rows, "summary": summ}
    try:
        out["host"]["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                    if l.startswith("model name")][0]
    except OSError:
        pass
    (CP_RESULTS / "profile.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))
    table(summ)


LABEL = {"greedy": "greedy (same belief)", "cauprobe": r"\ours-1 (static valuation)",
         "cauprobe-v2": r"\ours-2 (learning-aware)", "dual-rollout": "approximate dual control"}


def table(summ):
    from cp_common import ROOT
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{The controller's own work per five-minute control step (belief "
         r"update, valuation, safety filter, detectors; plant simulator excluded) on one "
         r"core of the host (Xeon E5-2698 v3, 2.3\,GHz) with single-threaded linear "
         r"algebra, measured while 56 of its 64 cores ran other jobs, over 14-day runs on four "
         r"development plants: mean, median and 99th percentile over all steps, the "
         r"slowest step of all runs, and the slowest step after the first; and the peak "
         r"resident memory of the process (in parentheses: the Python runtime and "
         r"libraries, before the plant and the controller are built).}",
         r"\label{tab:profile}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{5}{c}{ms per step} & MiB \\", r"\cmidrule(lr){2-6}",
         r"Controller & mean & median & p99 & max & max after step 1 & peak (base) \\",
         r"\midrule"]
    for m in METHODS:
        s = summ[m]
        L.append(f"{LABEL[m]} & {s['ms_mean']:.0f} & {s['ms_median']:.0f} & {s['ms_p99']:.0f} & "
                 f"{s['ms_max']:.0f} & {s['ms_max_after_first']:.0f} & "
                 f"{s['rss_peak_mib']:.0f} ({s['rss_base_mib']:.0f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (ROOT / "paper" / "tables_cp_profile.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()
