"""Run one BOPTEST job in-process (debugging aid)."""
import sys
import traceback

sys.path.insert(0, "/mnt/d/Experiments/CauSLM-Edge/experiments")
import cp_e6_boptest as E  # noqa: E402

method = sys.argv[1] if len(sys.argv) > 1 else "incumbent"
try:
    r = E.run((method, 190, 1, 1000, None))
    print("OK", method, r["tdis_kh"], r["tou_kwh_weighted"], r["kpis"])
except Exception:
    traceback.print_exc()
