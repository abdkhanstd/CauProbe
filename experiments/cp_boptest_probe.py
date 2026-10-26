"""Check that BOPTEST bestest_air runs in-process and list its points."""
import time

from cp_common import ROOT  # noqa: F401  (sets sys.path)
from causlm.probe.boptest_plant import BOPTEST_ROOT, _cwd, load_testcase

t0 = time.time()
tc = load_testcase("bestest_air")
print("loaded in %.1fs" % (time.time() - t0))
with _cwd(BOPTEST_ROOT):
    ins = tc.get_inputs()
    meas = tc.get_measurements()
    print("INPUTS:", {k: v for k, v in (ins[1] if isinstance(ins, tuple) else ins).items()} if False else "")
    print(type(ins), type(meas))
    print("inputs keys:", list((ins[2] if isinstance(ins, tuple) else ins).keys())[:60])
    print("meas keys:", list((meas[2] if isinstance(meas, tuple) else meas).keys())[:80])
    print(tc.set_step(300.0))
    t0 = time.time()
    r = tc.initialize(153 * 86400.0, 86400.0)
    print("init %.1fs" % (time.time() - t0), type(r))
    t0 = time.time()
    for k in range(288):
        y = tc.advance({"con_oveTSetCoo_u": 297.15, "con_oveTSetCoo_activate": 1})
    print("one day of 5-min steps in %.1fs" % (time.time() - t0))
    y = y[2] if isinstance(y, tuple) else y
    print({k: round(v, 3) for k, v in y.items() if isinstance(v, float)})
