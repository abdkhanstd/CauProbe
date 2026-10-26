"""The JSON contract between the gateway and the small language model.

The model is never asked for free text. It must emit exactly one object:

{
  "do": {"hvac_setpoint": 22.0, "sample_rate_hz": 0.25},
  "predicted_ATE": {"energy_kwh": -0.18, "indoor_temp": -0.4, "latency_ms": 0.0},
  "confounders": ["occupancy", "outdoor_temp"],
  "spurious_edges": [["wifi_rssi", "indoor_temp"]],
  "safety": "ok",
  "explain": "..."
}

`parse_advice` is deliberately forgiving about *formatting* (fenced blocks,
trailing prose, single quotes, trailing commas) and strict about *content*:
unknown actuators, out-of-range values and non-finite ATEs are rejected. A
small language model that emits something unusable therefore degrades to the
statistical fallback instead of driving the actuators with garbage.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from ..config import ACTUATORS, VARS

SAFETY_VALUES = ("ok", "caution", "unsafe")

ADVICE_SCHEMA = {
    "type": "object",
    "required": ["do", "predicted_ATE", "confounders", "safety", "explain"],
    "properties": {
        "do": {"type": "object"},
        "predicted_ATE": {"type": "object"},
        "confounders": {"type": "array"},
        "spurious_edges": {"type": "array"},
        "safety": {"enum": list(SAFETY_VALUES)},
        "explain": {"type": "string"},
    },
}

ATE_TARGETS = ["energy_cost", "indoor_temp", "latency_ms", "pmv"]
ATE_BOUNDS = {"energy_kwh": 6.0, "energy_cost": 12.0, "tariff": 3.0,
              "indoor_temp": 12.0, "latency_ms": 4000.0,
              "pmv": 3.5, "hvac_power": 8.0, "wifi_rssi": 30.0}


@dataclass
class CausalAdvice:
    do: Dict[str, float] = field(default_factory=dict)
    predicted_ATE: Dict[str, float] = field(default_factory=dict)
    confounders: List[str] = field(default_factory=list)
    spurious_edges: List[Tuple[str, str]] = field(default_factory=list)
    safety: str = "ok"
    explain: str = ""
    valid: bool = True
    parse_error: Optional[str] = None
    raw: str = ""

    def to_dict(self):
        d = asdict(self)
        d.pop("raw", None)
        return d


def advice_to_json(a: CausalAdvice) -> str:
    return json.dumps({"do": {k: round(float(v), 4) for k, v in a.do.items()},
                       "predicted_ATE": {k: round(float(v), 4)
                                         for k, v in a.predicted_ATE.items()},
                       "confounders": list(a.confounders),
                       "spurious_edges": [list(e) for e in a.spurious_edges],
                       "safety": a.safety,
                       "explain": a.explain}, separators=(",", ":"))


_OBJ = re.compile(r"\{.*\}", re.S)


def _extract_json_blob(text: str) -> Optional[str]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"```\s*$", "", t).strip()
    m = _OBJ.search(t)
    if not m:
        return None
    blob = m.group(0)
    # balance braces in case the model ran out of tokens mid-object
    depth, end = 0, None
    for i, ch in enumerate(blob):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end:
        blob = blob[:end]
    else:
        blob = blob + "}" * depth
    return blob


def _loads(blob: str):
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", blob)          # trailing commas
    fixed = re.sub(r"(?<![\\])'", '"', fixed)            # single quotes
    fixed = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "0", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def parse_advice(text: str, strict_actuators: bool = True) -> CausalAdvice:
    a = CausalAdvice(raw=text)
    blob = _extract_json_blob(text or "")
    if blob is None:
        a.valid, a.parse_error = False, "no_json_object"
        return a
    obj = _loads(blob)
    if not isinstance(obj, dict):
        a.valid, a.parse_error = False, "unparseable_json"
        return a

    do = obj.get("do", {})
    if not isinstance(do, dict):
        a.valid, a.parse_error = False, "do_not_object"
        return a
    clean_do: Dict[str, float] = {}
    for k, v in do.items():
        k = str(k).strip()
        if k not in ACTUATORS:
            if strict_actuators:
                a.parse_error = f"unknown_actuator:{k}"
                continue
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            a.parse_error = f"non_numeric:{k}"
            continue
        lo, hi = ACTUATORS[k]
        if not math.isfinite(fv):
            a.parse_error = f"non_finite:{k}"
            continue
        clean_do[k] = float(min(max(fv, lo), hi))
    a.do = clean_do

    ate = obj.get("predicted_ATE", obj.get("predicted_ate", {}))
    clean_ate: Dict[str, float] = {}
    if isinstance(ate, dict):
        for k, v in ate.items():
            k = str(k).strip()
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            b = ATE_BOUNDS.get(k)
            if b is not None:
                fv = float(min(max(fv, -b), b))
            clean_ate[k] = fv
    a.predicted_ATE = clean_ate

    conf = obj.get("confounders", [])
    a.confounders = [str(c).strip() for c in conf
                     if isinstance(conf, list) and str(c).strip() in VARS]

    sp = obj.get("spurious_edges", [])
    edges = []
    if isinstance(sp, list):
        for e in sp:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u, v = str(e[0]).strip(), str(e[1]).strip()
                if u in VARS and v in VARS and u != v:
                    edges.append((u, v))
            elif isinstance(e, str) and "->" in e:
                u, v = [x.strip() for x in e.split("->", 1)]
                if u in VARS and v in VARS and u != v:
                    edges.append((u, v))
    a.spurious_edges = edges

    s = str(obj.get("safety", "ok")).strip().lower()
    a.safety = s if s in SAFETY_VALUES else "ok"
    a.explain = str(obj.get("explain", ""))[:400]

    if not a.do:
        a.valid = False
        a.parse_error = a.parse_error or "empty_do"
    return a
