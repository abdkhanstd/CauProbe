"""Export the distilled expert for edge runtimes: merged HF -> GGUF (f16, q8, q4).

The merged model is written outside the repository (it is 1 GB); the GGUF
files are what an ARM gateway would load with llama.cpp.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cp_common import BASE_05, LORA_DIR

OUT = Path("/root/cp_edge")
LLAMA = Path("/root/llama.cpp")


def main():
    OUT.mkdir(exist_ok=True)
    merged = OUT / "merged"
    if not (merged / "config.json").exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_05)
        m = AutoModelForCausalLM.from_pretrained(BASE_05, dtype=torch.float16)
        m = PeftModel.from_pretrained(m, str(LORA_DIR)).merge_and_unload()
        m.save_pretrained(merged)
        tok.save_pretrained(merged)
    f16 = OUT / "cp-expert-f16.gguf"
    if not f16.exists():
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"),
                        str(merged), "--outfile", str(f16), "--outtype", "f16"],
                       check=True)
    q = LLAMA / "build-x86" / "bin" / "llama-quantize"
    for t in ("Q8_0", "Q4_K_M"):
        out = OUT / f"cp-expert-{t.lower()}.gguf"
        if not out.exists():
            subprocess.run([str(q), str(f16), str(out), t], check=True)
    for p in sorted(OUT.glob("*.gguf")):
        print(p.name, round(p.stat().st_size / 2 ** 20, 1), "MiB")


if __name__ == "__main__":
    main()
