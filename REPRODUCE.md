# CauProbe — reproducibility package

This archive contains everything behind the manuscript
*Ask, probe, or act? Learning-aware value of information for
self-commissioning IoT gateways*: the simulator and agents, every experiment
script, the raw per-run results behind every table and figure, the LoRA
adapter of the text expert, and the LaTeX sources.

## Layout

| Path | Contents |
|---|---|
| `causlm/probe/` | plant family (`plants.py`, `mech.py`), particle beliefs (`belief.py`), CauProbe-1/-2 (`agent.py`), baselines, text expert, run loop |
| `experiments/` | every experiment, analysis, table and figure script (`cp_*.py`) |
| `run_cauprobe_*.sh`, `run_review3.sh`, `run_review4.sh`, `run_review5.sh`, `run_audit3_powered.sh` | the grids, in the order they were run (worker counts are set for a 64-core host; lower `--workers` on smaller machines, or `WORKERS=n` for `run_audit3_powered.sh`) |
| `results/cauprobe/` | raw JSONL per run (one line per run: per-day return, violations, probes, asks, final posteriors) and every derived JSON |
| `results/cauprobe/*/FROZEN.txt` | checksums of the code frozen before each fresh test set |
| `frozen/test2/`, `frozen/test3/` | the exact frozen sources behind those checksums, and diffs to the current code (`frozen/*_to_current.diff`); `experiments/cp_verify_frozen.py --replay` checks the checksums and reproduces a stored test run bit for bit with both the frozen and the current code |
| `weights/cp_lora_qwen2.5-0.5b/` | LoRA adapter of the 0.5B text expert (base model: Qwen2.5-0.5B-Instruct) |
| `data/processed/probe_exogenous.npz` | cached exogenous traces (TON_IoT weather, Intel Lab light) |
| `paper/` | LaTeX sources: `main.tex` (manuscript, sections in `paper/ms/`), `supplement.tex` (Supplementary Material, sections in `paper/supp/`), generated tables (`tables_cp*.tex`, split per table into `paper/tables/`) and figures |

## Protocol (plant seeds)

| Set | Seeds | Used for |
|---|---|---|
| development | 500000–500015 | every design decision |
| validation | 500016–500063 | one run of each frozen iteration |
| test 1 | 900000–900047 (+OOL 950000+) | CauProbe-1, once |
| test 2 | 910000–910047 (+OOL 960000+) | CauProbe-2, once (frozen beforehand) |
| test 3 | 920000–920047 | fleet calibration, once (frozen beforehand) |
| test 4 | 930000–930047 | prior-generator mismatch, once (no method changed) |

## Regenerating the paper's numbers from the raw results

```bash
pip install -r requirements-cauprobe.txt
python experiments/cp_make_tables.py      # tables_cp*.tex, summary.json
python experiments/cp_make_figures.py     # figs/cp_fig*.pdf (full-size versions)
python experiments/cp_figs_small.py       # the panels the paper uses, paper/figs/cp_fig*_a.pdf, _b.pdf, _leg.pdf
python experiments/cp_numbers.py          # numbers quoted in the text (test 1)
python experiments/cp_iter2.py            # second iteration (test 2)
python experiments/cp_test3.py            # fleet calibration (test 3)
python experiments/cp_pooled.py           # pooled mean / tail / lock-in with plant-clustered CI (also writes paper/tables/pooled.tex)
python experiments/cp_pooled.py --ni      # non-inferiority with the margin from test 1
python experiments/cp_review3.py          # signed ablation, mismatch, detector
python experiments/cp_audit_summary.py    # paired probe audit
python experiments/cp_audit2_summary.py   # extended audit: what predicts a probe's payoff
python experiments/cp_voi_horizon.py     # value of a reading, predicted vs realised on a common horizon (test 2, 3)
python experiments/cp_voi_horizon.py --set test1   # the same on the first test set
python experiments/cp_audit3_power.py     # power calculation for the replicated audit (from the 48 x 6 pilot)
python experiments/cp_audit3_summary.py   # replicated audit, 157 decisions x 36 replicates (--src 48x6: the pilot)
python experiments/cp_dual.py             # against approximate dual control (results/cauprobe/dual2)
python experiments/cp_fleetcal.py && python experiments/cp_fleetcal.py --contamination
python experiments/cp_tausens.py          # sensitivity to mis-estimated free-learning horizons
python experiments/cp_profile.py          # per-step compute and memory of the controllers
python experiments/cp_false_alarms.py     # misfit-detector false alarms on the in-library test runs
python experiments/cp_verify_frozen.py --replay   # frozen code states
python experiments/cp_split_tables.py     # one file per table in paper/tables/
```

The table files in `paper/tables/` were edited by hand after splitting, for
punctuation and layout only: semicolons in captions and cells became commas or
slashes, `edge.tex` and `boptest.tex` use smaller type, `boptest.tex` has
no float of its own because the manuscript sets it beside the repeated-faults
table, and `ablation.tex`, now in the Supplementary Material, refers to the
main paper's Table 5 by the `M-` prefix. No number was changed. `cp_split_tables.py` overwrites those edits, so
run it into a copy of `paper/` to compare.

Building the paper: compile `supplement.tex` and `main.tex` once each with
pdflatex, run `python paper/make_bib.py` (each document's reference list, in
order of first citation), then compile both twice more; cross-references
between the two documents use the `xr` package. `python paper/make_clean.py`
then writes `paper_clean/`, with each document as a single file and the
cross-references resolved to numbers.

Re-running the simulations themselves uses the `run_*.sh` scripts (they were
run on a 64-core machine; a full grid takes 1–4 hours, the replicated probe
audit `run_audit3_powered.sh` about 22 hours, and it resumes with `--resume`
after an interruption). The expert needs the
Qwen2.5-0.5B-Instruct base model from Hugging Face; set `HF_DIR` in
`experiments/cp_common.py`.
