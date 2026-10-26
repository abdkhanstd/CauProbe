from .real_traces import (load_intel_lab, load_ton_iot, load_ukdale,
                          load_causaltime_official, DatasetUnavailable)
from .synthetic_var import make_var_dataset


def __getattr__(name):
    # The CausalTime-IoT generator needs torch; import it only when asked for,
    # so the trace loaders work in a numpy-only environment.
    if name in ("build_causaltime_iot", "load_causaltime_iot"):
        from . import causaltime_iot
        return getattr(causaltime_iot, name)
    raise AttributeError(name)
