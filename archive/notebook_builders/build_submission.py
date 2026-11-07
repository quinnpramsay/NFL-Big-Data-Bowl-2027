"""Builds the README-style submission from the same cells as build_nb.py.

  python3 build_submission.py work <dir>     -> <dir>/Juice+_work.ipynb (all the code; saves figures to figures/)
  python3 build_submission.py readme <dir>   -> <dir>/README.md (the write-up; run after the work notebook ran in <dir>)
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

MODE, DIR = sys.argv[1], Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
ns = {"__name__": "builder"}
sys.argv = ["build_nb.py", str(HERE / "_unused.ipynb")]
exec(compile((HERE / "build_nb.py").read_text(), "build_nb.py", "exec"), ns)
CELLS = ns["cells"]
WORK_NB = "Juice+_work.ipynb"

LINK = (f"*All the work is in [{WORK_NB}]({WORK_NB}). It runs the full pipeline on the competition data, draws "
        "every figure and table here, and rechecks every number quoted.*")
OLD_CLOSE = ("*All numbers come from one run of the code on the competition data. Code cells are hidden. "
             "Expand any cell to see how a figure was made. The appendix recomputes every quoted number.*")
NEW_CLOSE = ("*All numbers come from one run of the code on the competition data. The code for every figure and "
             f"table is in [{WORK_NB}]({WORK_NB}).*")


TABLE_IMG = r'''tbl.to_csv(FIG / "table1.csv", index=False)
# The same table as an image for the write-up (Change shaded: blue = grew in year 2, orange = fell)
from matplotlib.colors import to_rgba
show = tbl.copy()
for c in fmt:
    show[c] = show[c].map(lambda v: f"{v:+.2f}")
fig_t, ax_t = plt.subplots(figsize=(10, 0.3 * len(show) + 1.2))
ax_t.axis("off")
heads = [h.replace(" agility", "\nagility") for h in show.columns]
widths = [0.21, 0.07, 0.08, 0.08, 0.14, 0.15, 0.15, 0.12]
tb = ax_t.table(cellText=show.values, colLabels=heads, colWidths=widths, loc="upper center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.4)
for c_ in range(len(heads)):
    tb[0, c_].set_height(tb[0, c_].get_height() * 1.8)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r_ == 0:
        cell.set_text_props(weight="bold"); cell.set_facecolor("#f2f2f2")
    elif show.columns[c_] == "Change":
        v = tbl["Change"].iloc[r_ - 1]
        cell.set_facecolor(to_rgba(OI[0] if v > 0 else OI[3], 0.12 + 0.45 * min(abs(v) / 2, 1)))
ax_t.set_title("Table 1. Unused agility: WR and DB whose Combine agility ran 1+ SD ahead of their rookie game "
               "agility (SD within position)", loc="left", fontsize=11)
plt.savefig(FIG / "table1.png", dpi=150, bbox_inches="tight"); plt.close(fig_t)
'''


def fig_id(src):
    m = re.match(r"# (Figure|Table) (\d+)", src)
    return (f"{'fig' if m.group(1) == 'Figure' else 'table'}{m.group(2)}") if m else None


def build_work():
    out = []
    intro = nbf.v4.new_markdown_cell(
        "# Juice+: the work behind the write-up\n\n"
        "This notebook does all the work for the Juice+ write-up ([README.md](README.md)). It finds the competition "
        "data, runs the Juice+ pipeline (about 20 minutes the first time), draws every figure and table into "
        "`figures/`, and rechecks every number the write-up quotes (Appendix A). The pipeline code itself is in "
        "Appendix B.")
    out.append(intro)
    seen_title = False
    for c in CELLS:
        c = nbf.from_dict(c)
        if c.cell_type == "markdown":
            first = c.source.splitlines()[0]
            if not seen_title:
                seen_title = True  # the write-up title and key findings live in the README
                continue
            if first.startswith("## Appendix B"):
                out.append(c)
            elif first.startswith("## 7."):
                out.append(nbf.v4.new_markdown_cell("## Appendix A: every quoted number, rechecked"))
            elif first.startswith("## "):
                out.append(nbf.v4.new_markdown_cell(first))
            continue
        src = c.source
        meta = dict(c.metadata)
        if src.startswith("# Setup:"):
            src = src.replace('CODE.mkdir(exist_ok=True)',
                              'CODE.mkdir(exist_ok=True)\nFIG = WORK / "figures"  # the README embeds these\n'
                              'FIG.mkdir(exist_ok=True)')
            src = src.replace("import os, sys, re, json, glob, subprocess, warnings, time",
                              "import os, sys, re, json, glob, shutil, subprocess, warnings, time")
        fid = fig_id(src)
        if fid and fid.startswith("fig") and "plt.show()" in src:
            src = src.replace("plt.show()", f'plt.savefig(FIG / "{fid}.png", dpi=150, bbox_inches="tight"); plt.show()')
        if fid == "fig2":
            src = src.replace("display(Image(filename=str(gif_path)))",
                              'shutil.copy(gif_path, FIG / "fig2.gif")\ndisplay(Image(filename=str(gif_path)))')
        if fid == "table1":
            src = src.replace("try:\n    display(tbl.style", TABLE_IMG + "try:\n    display(tbl.style")
        if src.startswith("# The Juice+ pipeline (7 scripts)") or src.startswith("%%writefile"):
            pass  # keep: the packed line stays hidden; the readable scripts show code, hide the "Writing" line
        else:
            meta = {}  # this is the work file: show the code
        if src.startswith("%%writefile"):
            meta = {"_kg_hide-output": True, "jupyter": {"outputs_hidden": True}}
        cell = nbf.v4.new_code_cell(src)
        cell.metadata.update(meta)
        out.append(cell)
    nb = nbf.v4.new_notebook()
    nb.cells = out
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, DIR / WORK_NB)
    print("wrote", DIR / WORK_NB, len(out), "cells")


def table1_md():
    import pandas as pd
    t = pd.read_csv(DIR / "figures" / "table1.csv")
    num = ["Combine agility", "Rookie game agility", "Year-2 game agility", "Change"]
    lines = ["| " + " | ".join(t.columns) + " |",
             "|" + "|".join("---:" if c in num else "---" for c in t.columns) + "|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:+.2f}" if c in num else str(r[c]) for c in t.columns) + " |")
    return ("**Table 1. Unused agility.** WR and DB whose Combine agility ran 1+ SD ahead of their rookie game "
            "agility (SD within position).\n\n" + "\n".join(lines))


def build_readme():
    parts, first = [], True
    for c in CELLS:
        if c["cell_type"] == "markdown":
            s = c["source"]
            if s.startswith("## Appendix"):
                break
            s = s.replace(OLD_CLOSE, NEW_CLOSE)
            parts.append(s)
            if first:
                parts.append(LINK)
                first = False
        else:
            fid = fig_id(c["source"])
            if fid == "table1":
                assert (DIR / "figures" / "table1.png").exists()
                parts.append("![Table 1](figures/table1.png)")
            elif fid:
                ext = "gif" if fid == "fig2" else "png"
                assert (DIR / "figures" / f"{fid}.{ext}").exists(), fid
                parts.append(f"![Figure {fid[3:]}](figures/{fid}.{ext})")
    text = "\n\n".join(parts) + "\n"
    assert OLD_CLOSE not in text and "—" not in text
    (DIR / "README.md").write_text(text)
    words = [t for t in re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).split() if re.search(r"[A-Za-z0-9]", t)]
    print("wrote", DIR / "README.md", len(words), "words")


build_work() if MODE == "work" else build_readme()
