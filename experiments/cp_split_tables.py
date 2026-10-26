"""Split the generated table files into one file per table.

The table scripts (cp_make_tables.py, cp_iter2.py via cp_make_tables.py,
cp_dual.py, cp_audit*_summary.py, cp_pooled.py, cp_review3.py, cp_test3.py)
write paper/tables_cp*.tex, several tables to a file. The manuscript and the
supplementary material each include only some of them, so every table
environment is also written to paper/tables/<label>.tex (label without the
"tab:" prefix). Run after any of those scripts; it changes no content.
"""
from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
# typographic minus for a hyphen that starts a number in a cell
MINUS = re.compile(r"(?<=[\s\[(&])-(?=\d)")
# tables of the first test set and of BOPTEST, where \ours is the first
# iteration: name it, since the paper's text calls it CauProbe-1
FIRST_ITERATION = {"main", "ablation", "scenarios", "ool", "price", "devplants",
                   "boptest", "boptest-iter"}
SUPPLEMENT_ONLY = {"devplants", "boptest-iter", "iter2dev", "iter2val", "scenarios", "ool",
                   "expert", "price", "iter2price", "audit", "audit2"}
OURS = re.compile(r"\\ours(?![-\w])")


def main():
    out = PAPER / "tables"
    out.mkdir(exist_ok=True)
    pat = re.compile(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", re.S)
    n = 0
    for f in sorted(PAPER.glob("tables_cp*.tex")):
        for m in pat.finditer(f.read_text(encoding="utf-8")):
            lab = re.search(r"\\label\{tab:([^}]+)\}", m.group(0))
            if lab:
                t = MINUS.sub("$-$", m.group(0))
                if lab.group(1) in FIRST_ITERATION:
                    t = OURS.sub(r"\\ours-1", t)
                if lab.group(1) in ("main", "ablation"):
                    # the "-1" suffix widens the first column
                    t = t.replace(r"\setlength{\tabcolsep}{2.4pt}", r"\setlength{\tabcolsep}{2.1pt}")
                if lab.group(1) in SUPPLEMENT_ONLY:
                    # their captions point at sections of the main paper (xr prefix)
                    t = t.replace(r"\ref{sec:", r"\ref{M-sec:")
                (out / f"{lab.group(1)}.tex").write_text(t + "\n", encoding="utf-8")
                n += 1
    print(f"{n} tables written to {out}")


if __name__ == "__main__":
    main()
