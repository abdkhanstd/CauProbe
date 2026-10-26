# CauProbe

Code and model weights for the paper

**Ask, probe, or act? Learning-aware value of information for self-commissioning IoT gateways**

A gateway commissioned into an unknown building must learn which causal
mechanisms govern its plant while it is in control. It can learn by acting, by
reading its commissioning notes with an on-device language model, or by short
safe probes. CauProbe keeps a posterior over structural causal hypotheses and
prices every source by the regret it saves relative to what the controller's
own continued operation will reveal anyway (learning-aware value of
information), within the compute budget of an embedded gateway.

## Contents

| Path | Contents |
|---|---|
| `causlm/probe/` | plant family (`plants.py`, `mech.py`), particle beliefs (`belief.py`), CauProbe-1 and CauProbe-2 (`agent.py`), baselines, the text expert as a calibrated likelihood (`expert.py`), commissioning-text generator (`notes.py`), LoRA training and scoring of the expert (`slm_expert.py`), BOPTEST in-process plant (`boptest_plant.py`), run loop (`run.py`) |
| `causlm/` (other modules) | data loaders for the exogenous traces (`data/real_traces.py`), shared utilities, and modules of the authors' earlier gateway system that some scripts import |
| `experiments/cp_*.py` | one script per experiment, analysis, table and figure |
| `run_*.sh` | the experiment grids, in the order they were run |
| `frozen/test2/`, `frozen/test3/` | the exact sources frozen before the second and third fresh test sets, with diffs to the current code; `experiments/cp_verify_frozen.py --replay` checks them |
| `weights/cp_lora_qwen2.5-0.5b/` | LoRA adapter of the 0.5B text expert (base model: Qwen2.5-0.5B-Instruct, downloaded separately from Hugging Face) |
| `REPRODUCE.md` | the commands that regenerate every result |

The raw per-run results (`results/cauprobe/`), the cached exogenous traces
(`data/processed/probe_exogenous.npz`) and the LaTeX sources are not in this
repository; they are part of the reproducibility archive that accompanies the
paper. To regenerate them, run the grids (`run_*.sh`); create the folders
`results/cauprobe/` and `paper/tables/` first, since the analysis scripts write
there.

## Setup

```bash
pip install -r requirements-cauprobe.txt
```

All experiments were run with Python 3.12 on WSL Ubuntu 24.04 (single-threaded
BLAS, `OMP_NUM_THREADS=1`). BOPTEST runs need a separate environment with PyFMI.
The text expert needs the Qwen2.5-0.5B-Instruct base model; set `HF_DIR` in
`experiments/cp_common.py`. Worker counts in the run scripts were set for a
64-core host; lower `--workers` on smaller machines.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).
