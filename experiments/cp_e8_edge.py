"""E8: what one expert invocation costs at the edge -- measured where we can,
emulated where we cannot, and never confused.

Measured
  host-cpu   llama.cpp, 4 threads pinned, GGUF f16 / Q8_0 / Q4_K_M:
             prompt-processing and generation rates (llama-bench), and the
             latency of one deployed call (prefill P tokens + G decode tokens)
  host-gpu   the HF fp16 model on an RTX 2080 Ti: latency and NVML energy per
             call, idle draw subtracted only after the board has settled
Emulated
  qemu-arm   llama.cpp cross-compiled for AArch64 (Cortex-A76 target) and run
             under QEMU user-mode: does the ARM build run, how much memory does
             it need, and do its answers match the x86 build? (timing under
             QEMU is not representative and is not reported as such)
Modelled
  roofline   per-device latency: decode scaled from the host by effective
             memory bandwidth, prefill by effective compute; energy as active
             power above idle times latency. Device figures and the published
             Qwen2.5-0.5B measurements we compare against are cited in the
             paper.

Writes results/cauprobe/edge.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from cp_common import BASE_05, CP_RESULTS, LORA_DIR
from causlm.probe.notes import QUESTIONS, prompt_for

EDGE = Path("/root/cp_edge")
LLAMA = Path("/root/llama.cpp")
BIN = LLAMA / "build-x86" / "bin"
ARM_BIN = LLAMA / "build-arm" / "bin"
MODELS = {"f16": EDGE / "cp-expert-f16.gguf", "q8_0": EDGE / "cp-expert-q8_0.gguf",
          "q4_k_m": EDGE / "cp-expert-q4_k_m.gguf"}
CORES = os.environ.get("EDGE_CORES", "56,58,60,62")
HOST_ONLY = os.environ.get("EDGE_HOST_ONLY", "0") == "1"

# Device figures (nominal; sources in the paper). bw: theoretical memory
# bandwidth GB/s; stream: measured/attainable bandwidth GB/s where published;
# p_idle / p_load: board power W; cpu_gflops: nominal FP32 peak of the cores
# used by llama.cpp (4 threads), for the compute-bound prefill.
DEVICES = {
    "raspberry-pi-4": {"bw": 12.8, "stream": 5.8, "p_idle": 2.7, "p_load": 6.4,
                       "cpu_gflops": 96.0},
    "raspberry-pi-5": {"bw": 17.1, "stream": 8.8, "p_idle": 3.0, "p_load": 8.8,
                       "cpu_gflops": 153.6},
    "intel-n100": {"bw": 38.4, "stream": 25.0, "p_idle": 6.0, "p_load": 15.0,
                   "cpu_gflops": 217.6},
    "jetson-orin-nano-cpu": {"bw": 102.0, "stream": 60.0, "p_idle": 5.0,
                             "p_load": 15.0, "cpu_gflops": 108.8},
}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def prompts(n=40):
    from cp_expert import dataset
    te = dataset(n, 3_000_000, "test")
    return [prompt_for(t) for t in te[0]], te[1]


def token_counts(ps):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_05)
    P = [len(tok.encode(p, add_special_tokens=False)) for p in ps]
    ans = "".join(f"{q}: yes\n" for _, q in QUESTIONS)
    G = len(tok.encode(ans, add_special_tokens=False))
    return int(np.mean(P)), G


def host_cpu(P, G):
    out = {}
    for name, path in MODELS.items():
        r = sh(f"taskset -c {CORES} {BIN}/llama-bench -m {path} -t 4 -p {P} -n {G} "
               f"-r 5 -o json")
        try:
            js = json.loads(r.stdout)
        except Exception:
            out[name] = {"error": r.stderr[-400:]}
            continue
        pp = [x for x in js if x.get("n_prompt", 0) > 0][0]["avg_ts"]
        tg = [x for x in js if x.get("n_gen", 0) > 0][0]["avg_ts"]
        out[name] = {"pp_tps": pp, "tg_tps": tg,
                     "call_ms": 1000.0 * (P / pp + G / tg),
                     "size_mib": path.stat().st_size / 2 ** 20}
        print("host-cpu", name, out[name], flush=True)
    return out


def _copy_bw(core, q):
    os.sched_setaffinity(0, {core})
    a = np.ones(64 * 2 ** 20 // 8)
    b = np.empty_like(a)
    np.copyto(b, a)
    t = time.perf_counter()
    for _ in range(20):
        np.copyto(b, a)
    q.put(2 * a.nbytes * 20 / (time.perf_counter() - t) / 1e9)


def host_stream(reps=5):
    """Attainable host memory bandwidth with one copy stream on each of the four
    cores llama.cpp uses, run concurrently (the aggregate is what four decode
    threads share); median over repetitions."""
    from multiprocessing import get_context
    ctx = get_context("fork")
    cores = [int(c) for c in CORES.split(",")]
    agg = []
    for _ in range(reps):
        q = ctx.Queue()
        ps = [ctx.Process(target=_copy_bw, args=(c, q)) for c in cores]
        for x in ps:
            x.start()
        for x in ps:
            x.join()
        agg.append(sum(q.get() for _ in cores))
    return float(np.median(agg))


def host_gpu(ps, n=30):
    import pynvml
    import torch
    from causlm.probe.slm_expert import load_model, score_texts
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    tok, model = load_model(BASE_05, str(LORA_DIR), device="cuda:0")
    texts = [p.split("Site notes: ", 1)[1].rsplit("\nAnswers:", 1)[0] for p in ps]
    score_texts(tok, model, texts[:4], batch=1)                    # warm-up

    def power():
        return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0

    # settle: two consecutive 15 s windows within 1 W
    prev = None
    for _ in range(12):
        w = []
        t0 = time.time()
        while time.time() - t0 < 15:
            w.append(power())
            time.sleep(0.2)
        m = float(np.mean(w))
        if prev is not None and abs(m - prev) < 1.0:
            break
        prev = m
    p_idle = m
    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
        has_counter = True
    except Exception:
        has_counter = False
    samples = []
    t0 = time.time()
    lat = []
    for i in range(n):
        s = time.perf_counter()
        score_texts(tok, model, [texts[i % len(texts)]], batch=1)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - s)
        samples.append(power())
    dt = time.time() - t0
    if has_counter:
        e_tot = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0 - e0
    else:
        e_tot = float(np.mean(samples)) * dt
    e_call = (e_tot - p_idle * dt) / n
    out = {"device": pynvml.nvmlDeviceGetName(h), "p_idle_w": p_idle,
           "energy_counter": has_counter, "call_ms": 1000 * float(np.mean(lat)),
           "call_ms_sd": 1000 * float(np.std(lat)), "j_per_call": e_call,
           "n": n}
    print("host-gpu", out, flush=True)
    return out


def build_arm():
    if (ARM_BIN / "llama-cli").exists():
        return True
    r = sh(f"cd {LLAMA} && cmake -B build-arm -DCMAKE_SYSTEM_NAME=Linux "
           f"-DCMAKE_SYSTEM_PROCESSOR=aarch64 "
           f"-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc "
           f"-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DGGML_NATIVE=OFF "
           f"-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DBUILD_SHARED_LIBS=OFF "
           f"-DLLAMA_CURL=OFF -DCMAKE_EXE_LINKER_FLAGS=-static "
           f"-DCMAKE_BUILD_TYPE=Release && cmake --build build-arm -j 16 "
           f"--target llama-cli llama-completion", timeout=7200)
    print(r.stdout[-500:], r.stderr[-1500:], flush=True)
    return (ARM_BIN / "llama-cli").exists() or (ARM_BIN / "llama-completion").exists()


def answers_from(text):
    out = {}
    for F, q in QUESTIONS:
        m = re.search(rf"{q}:\s*(yes|no)", text)
        out[F] = None if m is None else (m.group(1) == "yes")
    return out


def run_cli(binary, model, prompt, qemu=False, threads=4):
    exe = "llama-completion" if (binary / "llama-completion").exists() else "llama-cli"
    pre = "/usr/bin/time -v qemu-aarch64-static -cpu cortex-a76 " if qemu else \
        "/usr/bin/time -v "
    pf = Path("/tmp/cp_prompt.txt")
    pf.write_text(prompt)
    r = sh(f"{pre}{binary}/{exe} -m {model} -f {pf} -n 28 --temp 0 -t {threads} "
           f"-no-cnv --no-warmup 2>&1", timeout=3600)
    txt = r.stdout
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    return txt, (int(rss.group(1)) / 1024.0 if rss else None)


def qemu_equivalence(ps, k=8):
    res = {"ran": False}
    if not build_arm():
        res["error"] = "arm build failed"
        return res
    agree, n, rss = 0, 0, []
    rows = []
    for p in ps[:k]:
        tx, _ = run_cli(BIN, MODELS["q4_k_m"], p)
        ta, m = run_cli(ARM_BIN, MODELS["q4_k_m"], p, qemu=True)
        a_x, a_a = answers_from(tx.split(p[-200:])[-1]), answers_from(ta.split(p[-200:])[-1])
        same = all(a_x[F] == a_a[F] for F in a_x)
        agree += int(same)
        n += 1
        if m:
            rss.append(m)
        rows.append({"x86": a_x, "arm": a_a})
        print("qemu", n, same, m, flush=True)
    res.update({"ran": True, "n": n, "answers_identical": agree,
                "peak_rss_mib": float(np.max(rss)) if rss else None, "rows": rows})
    return res


def roofline(cpu, stream_host, P, G):
    """Scale the measured host rates to each device."""
    out = {}
    host_gflops = float(os.environ.get("HOST_GFLOPS", "294.4"))  # 4 Haswell
    # cores x 2.3 GHz x 32 FP32 FLOP/cycle (AVX2 FMA)
    for dev, d in DEVICES.items():
        row = {}
        for q, c in cpu.items():
            if "tg_tps" not in c:
                continue
            # decode: memory-bound rate scaled by attainable bandwidth, but
            # never faster than the device's compute-bound prefill rate (a
            # decoded token costs at least the arithmetic of a prefilled one)
            pp = c["pp_tps"] * d["cpu_gflops"] / host_gflops
            tg_mem = c["tg_tps"] * d["stream"] / stream_host
            tg = min(tg_mem, pp)
            t = P / pp + G / tg
            row[q] = {"tg_tps": tg, "tg_mem_tps": tg_mem, "pp_tps": pp,
                      "decode_bound": "memory" if tg_mem <= pp else "compute",
                      "call_s": t,
                      "j_per_call": (d["p_load"] - d["p_idle"]) * t,
                      "j_per_call_total": d["p_load"] * t}
        out[dev] = row
    return out


def main():
    ps, _ = prompts()
    P, G = token_counts(ps)
    res = {"prompt_tokens": P, "answer_tokens": G}
    res["host_stream_gbps"] = host_stream()
    res["host_cpu"] = host_cpu(P, G)
    res["roofline"] = roofline(res["host_cpu"], res["host_stream_gbps"], P, G)
    prev = CP_RESULTS / "edge.json"
    if HOST_ONLY and prev.exists():
        # re-measure only the CPU side (e.g. on an idle host); keep the GPU
        # energy and the QEMU check from the previous run
        old = json.loads(prev.read_text())
        res["host_gpu"], res["qemu"] = old["host_gpu"], old["qemu"]
    else:
        try:
            res["host_gpu"] = host_gpu(ps)
        except Exception as e:
            res["host_gpu"] = {"error": repr(e)}
        res["qemu"] = qemu_equivalence(ps)
    res["devices"] = DEVICES
    (CP_RESULTS / "edge.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "qemu"}, indent=1,
                     default=str)[:3000])


if __name__ == "__main__":
    main()
