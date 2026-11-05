"""Shared settings and helpers for the scripts that run after juice.py.

juice.py stays self-contained so it can be pasted into a Kaggle notebook. The later scripts import from here, and
read the run ID that juice.py wrote, so every table from one run carries the same ID.
"""

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

COMP = "nfl-big-data-bowl-2027"
OUT = Path(os.getenv("JUICE_OUT_DIR", "juice_nb_output"))
CACHE = Path(os.getenv("JUICE_CACHE_DIR", "juice_cache"))
FIG = Path(os.getenv("JUICE_FIG_DIR", str(OUT / "figures")))
if os.path.isdir("/kaggle/working"):
    OUT, CACHE = Path("/kaggle/working/juice_nb_output"), Path("/kaggle/working/juice_cache")
    FIG = OUT / "figures"
for _p in (OUT, CACHE, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# Same position groups as juice.py: the ones the competition covers.
PGM = {"T": "OL", "G": "OL", "C": "OL", "DT": "DL", "NT": "DL", "DE": "EDGE", "OLB": "EDGE",
       "CB": "DB", "DB": "DB", "SS": "DB", "FS": "DB", "WR": "WR", "TE": "TE"}

_root = None


def _find_root():
    if os.getenv("JUICE_DATA_DIR"):
        return os.getenv("JUICE_DATA_DIR")
    spots = [Path("/kaggle/input"), Path.home() / ".cache" / "kagglehub" / "competitions" / COMP,
             Path(__file__).resolve().parent.parent / "data"]
    for spot in spots:
        if spot.is_dir():
            hits = sorted(spot.rglob("players.csv"))
            if hits:
                return str(hits[0].parent)
    import kagglehub
    return kagglehub.competition_download(COMP)


def get_file(name):
    global _root
    if _root is None:
        _root = _find_root()
    direct, nested = os.path.join(_root, name), os.path.join(_root, COMP, name)
    return direct if os.path.exists(direct) else nested


def load_run_id():
    """Run ID written by juice.py. The later scripts refuse to mix runs."""
    f = OUT / "run_manifest.json"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run juice.py first.")
    return json.loads(f.read_text())["run_id"]


def check_run(df, run_id, name):
    """Stop if a table read from OUT came from a different run."""
    if "run_id" in df.columns and df.run_id.nunique() == 1 and df.run_id.iloc[0] != run_id:
        raise SystemExit(f"{name} is from run {df.run_id.iloc[0]}, not {run_id}. Rerun the scripts in order.")
    return df.drop(columns=["run_id"], errors="ignore")


def read_out(name, run_id):
    return check_run(pd.read_csv(OUT / name), run_id, name)


def save_with_run(df, name, run_id):
    """Write a table with the run ID first. Floats are written to 10 significant digits, so the last-digit noise of
    floating-point sums does not make two runs of the same code differ byte for byte."""
    df = df.copy()
    df.insert(0, "run_id", run_id)
    df.to_csv(OUT / name, index=False, float_format="%.10g")


def section_rng(script, section, base=2027):
    """A random generator seeded by script and section label (e.g. "B."), so adding or reordering sections never
    moves another section's bootstrap draws."""
    return np.random.default_rng([base, zlib.crc32(f"{script}:{section}".encode())])


MIN_SIZE_GROUP = 10


def size_free(df, col, by="pgm", min_n=MIN_SIZE_GROUP):
    """Residual of col on weight and height within each group. Groups with fewer than min_n players are left blank:
    a 3-parameter fit on 2-3 players is exact, so its residual is rounding noise (and its sign flips between runs)."""
    o = pd.Series(np.nan, index=df.index)
    for _, g in df.dropna(subset=[col, "combine_weight", "combine_height"]).groupby(by):
        if len(g) < min_n:
            continue
        X = np.column_stack([np.ones(len(g)), g.combine_weight, g.combine_height])
        o.loc[g.index] = g[col] - X @ np.linalg.lstsq(X, g[col], rcond=None)[0]
    return o


def grouped_centered_mean(values, groups, half=2):
    """Centered rolling mean of `values` (window 2*half+1) that never crosses a change in `groups`.
    Both arrays must already be sorted so each group is contiguous. Fully vectorized."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    tot, cnt = np.zeros_like(v), np.zeros_like(v)
    n = len(v)
    for k in range(-half, half + 1):
        if k == 0:
            tot += np.nan_to_num(v)
            cnt += np.isfinite(v)
            continue
        src = np.arange(n) + k
        ok = (src >= 0) & (src < n)
        idx = np.clip(src, 0, n - 1)
        ok &= g[idx] == g
        val = v[idx]
        ok &= np.isfinite(val)
        tot += np.where(ok, val, 0)
        cnt += ok
    return tot / np.maximum(cnt, 1)


def group_codes(df, keys):
    """One integer per contiguous group of rows with the same keys (df must be sorted by keys)."""
    k = df[keys].to_numpy()
    change = np.r_[True, (k[1:] != k[:-1]).any(axis=1)]
    return np.cumsum(change)
