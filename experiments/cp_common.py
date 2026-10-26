"""Shared definitions for the CauProbe experiments: plant sets, seeds, paths.

Seed ranges are disjoint by construction:
  expert training notes      1_000_000 + i     (train vocabulary)
  expert validation notes    2_000_000 + i     (train vocabulary)
  expert test notes          3_000_000 + i     (test vocabulary)
  evaluation plants            500_000 + i     (test vocabulary notes)
  out-of-library plants        600_000 + i     (validation)
  out-of-library test plants   950_000 + i     (fresh, final runs)
  final test plants            900_000 + i     (fresh, final runs)
  regime-change plants         700_000 + i
  plant noise per run        plant seed * 10 + run
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causlm.config import RESULTS, WEIGHTS  # noqa: E402
from causlm.probe.plants import ALL_COMBOS  # noqa: E402

CP_RESULTS = RESULTS / "cauprobe"
CP_RESULTS.mkdir(parents=True, exist_ok=True)
EXPERT_DIR = CP_RESULTS / "experts"
EXPERT_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR = WEIGHTS / "cp_lora_qwen2.5-0.5b"
HF_DIR = Path("/mnt/d/Experiments/third_party/dl/hf")
BASE_05 = str(HF_DIR / "Qwen2.5-0.5B-Instruct")
BASE_15 = str(HF_DIR / "Qwen2.5-1.5B-Instruct")

PLANTS_PER_COMBO = 4
EVAL_PLANTS = [(500_000 + i, ALL_COMBOS[i % len(ALL_COMBOS)])
               for i in range(len(ALL_COMBOS) * PLANTS_PER_COMBO)]
def _ool_set(base):
    return ([(base + i, ALL_COMBOS[(3 * i) % 16], "tsch") for i in range(8)]
            + [(base + 100 + i, ALL_COMBOS[(5 * i + 1) % 16], "congested_wan")
               for i in range(8)]
            + [(base + 200 + i, ALL_COMBOS[(7 * i + 2) % 16], "broker_fault")
               for i in range(8)])


# Out-of-library plants: the validation set (600_000+) was run with the first
# frozen method; the test set (950_000+, fresh) is what the paper reports.
OOL_VAL_PLANTS = _ool_set(600_000)
OOL_TEST_PLANTS = _ool_set(950_000)
OOL_PLANTS = OOL_TEST_PLANTS
REGIME_PLANTS = [(700_000 + i, ALL_COMBOS[(7 * i) % 16]) for i in range(16)]

# The first 16 evaluation plants (seeds 500_000-500_015, one per mechanism
# combination) were used while the method was being developed. They are kept
# out of every primary result; the 48 others were never looked at before the
# final runs.
DEV_EVAL_SEEDS = set(range(500_000, 500_016))
HELDOUT_PLANTS = [(s, c) for s, c in EVAL_PLANTS if s not in DEV_EVAL_SEEDS]


# The final test set: 48 fresh plants (three per mechanism combination),
# generated after the method was frozen. Nothing was run on them before the
# final experiments.
TEST_PLANTS = [(900_000 + i, ALL_COMBOS[i % len(ALL_COMBOS)]) for i in range(48)]
TEST_SEEDS = {s for s, _ in TEST_PLANTS}


# Seed ranges reserved for a second fresh test set, for a second iteration of
# the method developed after the first test runs. Nothing runs on them until
# that iteration is frozen; the development scripts assert they never do.
TEST2_SEEDS_GUARD = set(range(910_000, 911_000)) | set(range(960_000, 961_000))


# The second test set (2026-10-17), created after the second iteration
# (cauprobe-v2) had been developed on the dev plants, validated once on the
# validation plants (results/cauprobe/validation_v2) and frozen: 48 fresh
# plants (three per mechanism combination) and 24 fresh out-of-library plants.
# Nothing ran on them before the final runs (run_cauprobe_test2_grid.sh).
TEST2_PLANTS = [(910_000 + i, ALL_COMBOS[i % len(ALL_COMBOS)]) for i in range(48)]
OOL_TEST2_PLANTS = _ool_set(960_000)
TEST2_SEEDS = {s for s, _ in TEST2_PLANTS} | {s for s, _, _ in OOL_TEST2_PLANTS}
assert TEST2_SEEDS <= TEST2_SEEDS_GUARD


# The third test set (2026-10-17), created after fleet calibration of the
# expert (cp_build_fleet_expert.py: temperatures and logit bank refitted on the
# interventional labels of the validation, first-test and second-test sites)
# had been frozen. Used once, to test whether a fleet-calibrated expert changes
# commissioning at new sites (run_cauprobe_test3_grid.sh).
TEST3_PLANTS = [(920_000 + i, ALL_COMBOS[i % len(ALL_COMBOS)]) for i in range(48)]
TEST3_SEEDS = {s for s, _ in TEST3_PLANTS}
assert not TEST3_SEEDS & (TEST2_SEEDS | TEST_SEEDS)


# The fourth test set (2026-10-18): 48 fresh plants for the prior-generator
# mismatch scenarios, in which plant parameters are drawn from ranges twice as
# wide as the agent's prior. No method was changed for it; every controller is
# run exactly as frozen.
TEST4_PLANTS = [(930_000 + i, ALL_COMBOS[i % len(ALL_COMBOS)]) for i in range(48)]
TEST4_SEEDS = {s for s, _ in TEST4_PLANTS}
assert not TEST4_SEEDS & (TEST3_SEEDS | TEST2_SEEDS | TEST_SEEDS)


def testrows(rows):
    return [r for r in rows if r["seed"] // 10 in TEST_SEEDS]


def heldout(rows):
    """Rows on plants never used during development (OOL plants included)."""
    return [r for r in rows if r["seed"] // 10 not in DEV_EVAL_SEEDS]


def devrows(rows):
    return [r for r in rows if r["seed"] // 10 in DEV_EVAL_SEEDS]
