"""Loaders for the four public datasets used in the paper.

* Intel Berkeley Research Lab sensor deployment (Madden, 2004) -- 2.3M rows of
  temperature/humidity/light/voltage from 54 motes.
* TON_IoT telemetry (Moustafa, UNSW Canberra) -- labelled IoT device telemetry
  with seven attack families.
* UK-DALE (Kelly & Knottenbelt, 2015) -- disaggregated domestic appliance power.
* CausalTime (Cheng et al., ICLR 2024) -- generated series with ground-truth
  causal graphs (medical / pm25 / traffic).

Each loader returns a tidy `pandas.DataFrame` resampled to the 5-minute control
step used everywhere else, plus a small provenance dict.
"""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import DATA_PROC, DATA_RAW


class DatasetUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- Intel Lab
INTEL_COLS = ["date", "time", "epoch", "moteid", "temperature", "humidity",
              "light", "voltage"]


def load_intel_lab(max_rows: int | None = None, resample: str = "5min",
                   cache: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Intel Berkeley Lab motes -> zone-level temperature/humidity/light/voltage."""
    cache_f = DATA_PROC / f"intel_lab_{resample}.parquet"
    if cache and cache_f.exists():
        df = pd.read_parquet(cache_f)
        return df, {"source": "Intel Berkeley Research Lab (2004)", "cached": True,
                    "n": len(df)}

    src = DATA_RAW / "intel_lab_data.txt.gz"
    if not src.exists():
        raise DatasetUnavailable(f"missing {src}")

    rows = []
    with gzip.open(src, "rt", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            p = line.split()
            if len(p) != 8:
                continue
            rows.append(p)
    raw = pd.DataFrame(rows, columns=INTEL_COLS)
    for c in ["epoch", "moteid", "temperature", "humidity", "light", "voltage"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["ts"] = pd.to_datetime(raw["date"] + " " + raw["time"], errors="coerce")
    raw = raw.dropna(subset=["ts", "temperature", "humidity", "moteid"])

    # The deployment has well-known sensor faults (voltage brown-out drives the
    # temperature reading to >100 C). Filter to physically plausible values.
    raw = raw[(raw.temperature.between(5, 45)) & (raw.humidity.between(0, 100))
              & (raw.voltage.between(2.0, 3.2)) & (raw.light.between(0, 2000))]
    raw = raw[raw.moteid.between(1, 54)]

    # Aggregate the mote fleet into zone-level KPIs at the control step.
    raw = raw.set_index("ts").sort_index()
    agg = raw.resample(resample).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        light=("light", "mean"),
        voltage=("voltage", "mean"),
        n_reports=("moteid", "count"),
        n_motes=("moteid", "nunique"),
    )
    agg = agg.dropna(subset=["temperature"])
    # Packet-delivery-ratio proxy: reports received / reports expected.
    expected = agg["n_motes"].clip(lower=1) * (pd.Timedelta(resample).seconds / 31.0)
    agg["pdr_proxy"] = (agg["n_reports"] / expected).clip(0, 1)
    agg = agg.interpolate(limit=6).dropna()
    agg = agg.reset_index()

    if cache:
        agg.to_parquet(cache_f, index=False)
    return agg, {"source": "Intel Berkeley Research Lab (2004)",
                 "url": "http://db.csail.mit.edu/labdata/labdata.html",
                 "n": len(agg), "cached": False}


# ------------------------------------------------------------------ TON_IoT
TON_FILES = {
    "thermostat": "Train_Test_IoT_Thermostat.csv",
    "weather": "Train_Test_IoT_Weather.csv",
    "fridge": "Train_Test_IoT_Fridge.csv",
    "motion_light": "Train_Test_IoT_Motion_Light.csv",
    "garage_door": "Train_Test_IoT_Garage_Door.csv",
    "modbus": "Train_Test_IoT_Modbus.csv",
    "gps": "Train_Test_IoT_GPS_Tracker.csv",
}


def load_ton_iot(devices=("thermostat", "weather", "fridge", "motion_light")
                 ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """TON_IoT telemetry with per-row attack labels."""
    base = DATA_RAW / "ton_iot"
    if not base.exists():
        raise DatasetUnavailable(f"missing {base}")
    out = {}
    for d in devices:
        f = base / TON_FILES[d]
        if not f.exists():
            continue
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        df["type"] = df["type"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        out[d] = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if not out:
        raise DatasetUnavailable("no TON_IoT telemetry files found")
    meta = {"source": "TON_IoT telemetry, UNSW Canberra (Moustafa 2021)",
            "url": "https://research.unsw.edu.au/projects/toniot-datasets",
            "devices": list(out), "rows": {k: len(v) for k, v in out.items()},
            "attack_types": sorted(set().union(*[set(v["type"]) for v in out.values()]))}
    return out, meta


def ton_iot_attack_windows(df: pd.DataFrame, attack: str) -> np.ndarray:
    """Return the row indices belonging to one attack family."""
    return np.flatnonzero((df["type"].values == attack))


# ------------------------------------------------------------------ UK-DALE
def load_ukdale(house: int = 1, resample: str = "5min",
                channels=(1,), max_days: int = 120,
                cache: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """UK-DALE 6-second disaggregated appliance power for one house.

    channel 1 is the whole-house aggregate; the loader also pulls a handful of
    named appliance channels so that the environment can drive a realistic
    non-HVAC base load.
    """
    cache_f = DATA_PROC / f"ukdale_h{house}_{resample}.parquet"
    if cache and cache_f.exists():
        df = pd.read_parquet(cache_f)
        return df, {"source": "UK-DALE (Kelly & Knottenbelt 2015)", "cached": True,
                    "n": len(df)}
    z = DATA_RAW / "ukdale.zip"
    if not z.exists():
        raise DatasetUnavailable(f"missing {z}")

    series = {}
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        labels_name = [n for n in names if n.endswith(f"house_{house}/labels.dat")]
        labels = {}
        if labels_name:
            for line in zf.read(labels_name[0]).decode().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    labels[int(parts[0])] = "_".join(parts[1:])
        for ch in channels:
            cand = [n for n in names
                    if n.endswith(f"house_{house}/channel_{ch}.dat")]
            if not cand:
                continue
            raw = zf.read(cand[0]).decode(errors="ignore")
            arr = np.loadtxt(io.StringIO(raw))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            ts = pd.to_datetime(arr[:, 0], unit="s")
            s = pd.Series(arr[:, 1], index=ts).sort_index()
            if max_days:
                s = s[s.index <= s.index[0] + pd.Timedelta(days=max_days)]
            series[labels.get(ch, f"channel_{ch}")] = s.resample(resample).mean()
    if not series:
        raise DatasetUnavailable(f"no UK-DALE channels for house {house}")
    df = pd.DataFrame(series).interpolate(limit=12).dropna().reset_index(
        names="ts")
    if cache:
        df.to_parquet(cache_f, index=False)
    return df, {"source": "UK-DALE (Kelly & Knottenbelt 2015)",
                "url": "https://jack-kelly.com/data/", "house": house,
                "channels": list(series), "n": len(df), "cached": False}


# -------------------------------------------------------------- CausalTime
CT_DATASETS = ("medical", "pm25", "traffic")


def load_causaltime_official(name: str = "medical") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Official CausalTime release: (samples, T, 2N) data and the N x N GT graph."""
    base = DATA_RAW / "causaltime" / "causaltime_gen_ver1.0" / name
    if not base.exists():
        raise DatasetUnavailable(f"missing {base}")
    x = np.load(base / "gen_data.npy")
    g = np.load(base / "graph.npy").astype(np.float32)
    meta = {"source": "CausalTime v1.0 (Cheng et al., ICLR 2024)",
            "url": "https://www.causaltime.cc/", "dataset": name,
            "data_shape": list(x.shape), "graph_shape": list(g.shape),
            "n_edges": int(g.sum())}
    return x, g, meta
