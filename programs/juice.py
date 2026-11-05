"""Juice+: Combine movement measured from tracking, and the same movement measured in NFL games.

Combine Juice+ is built from Combine tracking. Game Juice+ is built from 2023-2025 regular-season tracking.
Both are on one scale: 100 = average for the position group, 10 points = one standard deviation.
Position groups are the ones the competition covers: WR, TE, DB, OL, DL and EDGE (LB and RB are left out).

Changes from the November 3 audit (juice_plus_improvement_plan.md):
  - Burst (E) is the time from yard 2 to yard 10 of the tracked 40 (t2_10). The old d15/v1 pair started the clock
    at the first frame at 1 yd/s, so one frame of error moved it by most of an SD.
  - Nov 5 test pass: E now averages the yard 2-10 split with the mean acceleration over the same yards (the
    acceleration measure repeats at 0.66 attempt to attempt, the split at 0.47). Game burst is measured the same clock-free way from the snap spot on standing-start
    plays that reach 10 yards within 4 s. The old game burst (speed at 1.0 s, distance at 1.5 s after the snap) was
    mostly reaction time: it tracked Combine burst at only +0.07 and the 40 at +0.17. The new one tracks Combine burst
    at about +0.21, and Game Juice+ pieces agree far better (alpha: WR 0.50 -> 0.63, TE -0.11 -> 0.75).
    Old behavior: BURST_METHOD=split GAME_BURST=snap.
  - Game agility per play uses the second-highest frame, not the single highest (GAME_AGIL=max for the old one).
  - GAME_AGIL_WINDOW=prethrow measures WR/TE/DB game agility before the throw only. Off by default (tried and reverted,
    see PRIMARY_TESTS.md).
  - E shrinkage uses the retest reliability of the E piece itself, not the average of its parts' reliabilities.
  - Game Juice+ needs MIN_GAME_PLAYS usable plays (default 50). Players below the floor are reported, not scored.
  - Game Juice+ is the reliability-weighted average of its shrunk pieces on the 100 scale. It is no longer
    re-standardized, so a final score always sits inside the range of its own pieces.
  - The size model is plain out-of-fold least squares (no ridge). Unadjusted scores are saved next to it.
  - Every output carries a run_id, and run_manifest.json records the settings of the run.
  - The position-blend weight search is off by default (RUN_BLEND_SEARCH=1 to turn it on).

Run in Colab or locally. Needs the competition files (Kaggle login) or JUICE_DATA_DIR pointing at them.
Run order: juice.py -> juice_physics.py -> juice_transfer.py -> juice_predictive.py -> juice_casework.py
-> juice_routes.py -> juice_figures.py. Once, after the frozen run: juice_forward.py --freeze (2026 forward test).
"""

# ## 0. Setup and settings

import os
import glob
import json
import hashlib
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

warnings.filterwarnings("ignore")

# Folders. Set JUICE_DATA_DIR to a local copy of the competition files, or let kagglehub download them.
COMP = "nfl-big-data-bowl-2027"
DATA_DIR = os.getenv("JUICE_DATA_DIR")
CACHE = Path(os.getenv("JUICE_CACHE_DIR", "juice_cache"))
OUT = Path(os.getenv("JUICE_OUT_DIR", "juice_nb_output"))
if os.path.isdir("/kaggle/input"):
    hits = glob.glob("/kaggle/input/**/players.csv", recursive=True)
    if hits:
        DATA_DIR = os.path.dirname(hits[0])
    CACHE, OUT = Path("/kaggle/working/juice_cache"), Path("/kaggle/working/juice_nb_output")
for _p in (CACHE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# Model settings.
SEASONS = (2023, 2024, 2025)
CHUNK_SIZE = 2_000_000
MIN_PLAY_SECONDS = 3.0
MIN_FRAMES = int(round(MIN_PLAY_SECONDS * 10))   # plays shorter than this are dead plays
MIN_FRAMES_OPEN = 20                             # plays used for top speed and agility
# Which Combine score is the main one. "tracked_only" uses the tracked movement data alone.
# "blend_50_50" averages each tracked piece with the official tests of the same trait.
JUICE_VERSION = os.getenv("JUICE_VERSION", "blend_50_50")
BLEND_STOPWATCH = JUICE_VERSION == "blend_50_50"
DROP_BLOWOUTS, GARBAGE_WP = True, 0.05           # drop snaps when home win probability is under 5% or over 95%
GATE_ALPHA, MIN_PLAYERS_DRILL, MIN_OVERLAP = 0.5, 20, 15
RELIABILITY_MIN = (200, 50)                      # reliabilities are estimated on well-sampled players only
SEASON_MIN = (100, 30)                           # persistence check, one season at a time
MIN_GAME_PLAYS = int(os.getenv("MIN_GAME_PLAYS", "50"))   # usable open plays needed for a Game Juice+ score
# Burst measure for the Combine E piece. "split_macc" (default) = average of the yard 2-10 split time and the mean
# acceleration over yards 2-10 (retest 0.72 vs 0.56 for the split alone). "split" = split only. "legacy" = old d15/v1.
BURST_METHOD = os.getenv("BURST_METHOD", "split_macc")
SPLIT_FROM, SPLIT_TO = 2.0, 10.0
# Game burst. "clockfree" (default) = the same yard 2-10 split and mean acceleration, measured from the snap position on
# standing-start plays that reach 10 yards within 4 s. No snap clock, so reaction time is not mixed in, matching the
# Combine 40. "clockfree_split" = the split time only. "snap" = old speed at 1.0 s and distance at 1.5 s after the
# snap (mostly reaction / get-off).
GAME_BURST = os.getenv("GAME_BURST", "clockfree")
# Game agility per play: "second" (default) = second-highest frame of lateral acceleration (s >= 2 yd/s), robust to one
# noisy frame. "max" = highest frame (old). "p90" = 90th percentile of the play's frames. "max_s1" = max, s >= 1 yd/s.
GAME_AGIL = os.getenv("GAME_AGIL", "second")
AGIL_COL = {"max": "alat_max", "second": "alat_2nd", "p90": "alat_p90", "max_s1": "alat_max_s1"}[GAME_AGIL]
# Which frames count for game agility. "all" (default): every frame after the snap. "prethrow": for WR, TE and DB on
# plays with a forward pass, only the frames from the snap to the throw (at least PRE_THROW_MIN_FRAMES of them).
# "prethrow" was tried on November 5, 2026 and reverted: it did not link to separation better (check R4, +0.114 vs
# +0.139) and weakened P1 (+0.147 vs +0.172). See PRIMARY_TESTS.md.
GAME_AGIL_WINDOW = os.getenv("GAME_AGIL_WINDOW", "all")
PRE_THROW_MIN_FRAMES = 10
PRE_THROW_GROUPS = ("WR", "TE", "DB")
BURST_WINDOW = 40                                # frames after the snap searched for the 2- and 10-yard marks
EXPL_PARTS = {"clockfree": ["gsplit_c", "gmacc_c"], "clockfree_split": ["gsplit_c"]}.get(GAME_BURST, [])
# Top-end measure for the Combine T piece. "fly10" = fastest 10 yards (default). "peak" = peak smoothed speed.
TOP_METHOD = os.getenv("TOP_METHOD", "fly10")
RUN_BLEND_SEARCH = os.getenv("RUN_BLEND_SEARCH", "0") == "1"
BELOW_AVG, ELITE, LOW_JUICE, HIGH_JUICE = 100, 110, 90, 110
HIGH_PICK, LOW_PICK = 64, 140
BOOT = 1000
REUSE_JUICE_PY_CACHE = False   # True only if game_play_table_v2.csv in the cache folder was written by juice.py
YD_TO_MPH = 3600 / 1760
rng = np.random.default_rng(42)

# Position groups the competition covers: WR, TE, DB, OL and DL (DL is split into interior DL and EDGE).
PGM = {"T": "OL", "G": "OL", "C": "OL", "DT": "DL", "NT": "DL", "DE": "EDGE", "OLB": "EDGE",
       "CB": "DB", "DB": "DB", "SS": "DB", "FS": "DB", "WR": "WR", "TE": "TE"}

# Run ID: one per run, written into every output table, so every number in the writeup can be traced to one run.
SETTINGS = {k: v for k, v in dict(
    JUICE_VERSION=JUICE_VERSION, MIN_GAME_PLAYS=MIN_GAME_PLAYS, BURST_METHOD=BURST_METHOD, GAME_BURST=GAME_BURST, GAME_AGIL=GAME_AGIL, TOP_METHOD=TOP_METHOD,
    GAME_AGIL_WINDOW=GAME_AGIL_WINDOW, DROP_BLOWOUTS=DROP_BLOWOUTS, GARBAGE_WP=GARBAGE_WP, MIN_PLAY_SECONDS=MIN_PLAY_SECONDS,
    MIN_FRAMES_OPEN=MIN_FRAMES_OPEN, GATE_ALPHA=GATE_ALPHA, RUN_BLEND_SEARCH=RUN_BLEND_SEARCH).items()}
_here = Path(__file__).resolve().parent if "__file__" in globals() else None
try:
    _git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_here).stdout.strip()
    # Uncommitted edits to the pipeline mean the commit alone can't say which code made this run.
    _dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                                 cwd=_here).stdout.strip()) if _git else False
except Exception:
    _git, _dirty = "", False
if _git and _dirty:
    _git += "-dirty"
# Fingerprint of the pipeline code itself (every .py next to this file), recorded in the manifest.
try:
    _h = hashlib.sha1()
    for _f in sorted((_here or Path(".")).glob("*.py")):
        _h.update(_f.name.encode())
        _h.update(_f.read_bytes())
    CODE_SHA = _h.hexdigest()[:12]
except Exception:
    CODE_SHA = ""
_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"{_stamp}-{hashlib.sha1(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:6]}"
print(f"Run ID: {RUN_ID}  (git {_git or 'n/a'}, code {CODE_SHA or 'n/a'})  settings: {SETTINGS}")
ALIAS = {"PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL", "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
         "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST"}
KEYS = ["game_id", "play_id", "nfl_id"]

# ## 1. Helpers

_root = None


def kaggle_login():
    """Read Kaggle credentials from a .env file (this folder or any parent folder), or from Colab secrets.
    Supports KAGGLE_USERNAME + KAGGLE_KEY, or a single KAGGLE_API_TOKEN. Never prompts."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in [Path.cwd(), here, *here.parents]:
        env_file = folder / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.replace("export ", "").strip()
                    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
            print(f"Read Kaggle settings from {env_file}")
            break
    # Also accept the name KAGGLE_API. New-style tokens (KGAT_...) are a token; older ones are a key.
    api = os.environ.get("KAGGLE_API", "").strip()
    if api:
        os.environ["KAGGLE_API_TOKEN" if api.startswith("KGAT_") else "KAGGLE_KEY"] = api
    try:
        from google.colab import userdata
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            try:
                os.environ.setdefault(k, userdata.get(k))
            except Exception:
                pass
    except Exception:
        pass
    have = [k for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_API") if os.environ.get(k)]
    print("Kaggle variables found:", have or "none")


def find_local_data():
    """Folder that already holds the competition files: the kagglehub cache, or a data folder near this script."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    spots = [Path.home() / ".cache" / "kagglehub" / "competitions" / COMP,
             here / "data", here.parent / "data", Path.cwd() / "data"]
    for spot in spots:
        if spot.is_dir():
            hits = sorted(spot.rglob("players.csv"))
            if hits:
                print(f"Using local competition files in {hits[0].parent}")
                return str(hits[0].parent)
    return None


def get_file(name):
    """Path to a competition file, from JUICE_DATA_DIR or a Kaggle download."""
    global _root
    if DATA_DIR:
        return os.path.join(DATA_DIR, name)
    if _root is None:
        _root = find_local_data()
    if _root is None:
        import kagglehub
        kaggle_login()
        _root = kagglehub.competition_download(COMP)
    direct, nested = os.path.join(_root, name), os.path.join(_root, COMP, name)
    return direct if os.path.exists(direct) else nested


def load_csv(name, **kw):
    return pd.read_csv(get_file(name), **kw)


def save(df, name):
    """Write a table with the run ID as its first column."""
    df = df.copy()
    df.insert(0, "run_id", RUN_ID)
    df.to_csv(OUT / name, index=False, float_format="%.10g")   # 10 digits: no last-digit noise between runs


def circ(deg):
    return (deg + 180.0) % 360.0 - 180.0


def zs(s):
    sd = s.std()
    return pd.Series(np.nan, index=s.index) if not np.isfinite(sd) or sd == 0 else (s - s.mean()) / sd


def safe_spearman(x, y):
    p = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if len(p) < 4 or p.x.nunique() < 2 or p.y.nunique() < 2:
        return np.nan
    return float(spearmanr(p.x, p.y)[0])


def fisher_ci(r, n):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def pooled_fisher(rs_ns):
    """Pool correlations across groups by Fisher z weighted by n-3. Returns r, lo, hi."""
    rs_ns = [(r, n) for r, n in rs_ns if np.isfinite(r) and abs(r) < 1 and n > 3]
    if not rs_ns:
        return np.nan, np.nan, np.nan
    w = sum(n - 3 for _, n in rs_ns)
    r = float(np.tanh(sum(np.arctanh(r) * (n - 3) for r, n in rs_ns) / w))
    se = 1.96 / np.sqrt(w)
    return r, float(np.tanh(np.arctanh(r) - se)), float(np.tanh(np.arctanh(r) + se))


def sb(k, r):
    """Spearman-Brown: reliability of the mean of k observations, each with reliability r."""
    k = np.asarray(k, dtype=float)
    if not np.isfinite(r) or r <= 0:
        return np.zeros_like(k, dtype=float)
    return np.clip(k * r / (1 + (k - 1) * r), 0, 1)


def single_obs_reliability(r_full, k):
    """Invert Spearman-Brown: reliability of ONE observation."""
    if not np.isfinite(r_full) or r_full <= 0 or not np.isfinite(k) or k <= 1:
        return 0.0
    r_full = min(float(r_full), 0.999)
    return r_full / (k - (k - 1) * r_full)


def scale100(df, col, by="pgm"):
    g = df.groupby(by)[col]
    return 100 + 10 * (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


def auc_ci(y, s, reps=1000, seed=3):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 10 or y.min() == y.max():
        return np.nan, np.nan, np.nan
    r, boots = np.random.default_rng(seed), []
    for _ in range(reps):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def adjust(df, col, use_class=True, size_adj=True):
    """Leave-one-out residual of `col` within each position group after removing body size
    (weight, weight squared, height) and draft class, standardized to mean 0, sd 1."""
    out = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("pgm"):
        d = d.dropna(subset=[col, "combine_weight", "combine_height"])
        if len(d) < 15:
            continue
        y = d[col].to_numpy()
        feats = (pd.get_dummies(d.draft_year, prefix="yr", drop_first=True).astype(float)
                 if use_class else pd.DataFrame(index=d.index))
        if size_adj and len(d) >= 25:
            feats["w"] = d.combine_weight.to_numpy()
            feats["h"] = d.combine_height.to_numpy()
            feats["w2"] = ((d.combine_weight - d.combine_weight.mean()) ** 2 / 100).to_numpy()
        if feats.shape[1] == 0:
            pred = np.array([np.delete(y, i).mean() for i in range(len(y))])
        else:
            # Plain least squares, fit out of fold. Ridge left a weight link near -0.10 (audit item 9).
            pred = cross_val_predict(LinearRegression(), feats.to_numpy(), y, cv=LeaveOneOut())
        r = y - pred
        if r.std() > 0:
            out.loc[d.index] = (r - r.mean()) / r.std()
    return out

# ## 2. Players and Combine tracking
#
# Only players in the six position groups **with Combine tracking** are scored, so the game data covers exactly the
# positions the Combine data covers.

players = load_csv("players.csv")
combine = load_csv("combine_results.csv")
games = load_csv("games.csv")

LOWER_BETTER = [c for c in ["forty", "ten_yd_split", "three_cone", "short_shuttle"] if c in combine.columns]
HIGHER_BETTER = [c for c in ["vertical", "broad_jump", "bench_reps"] if c in combine.columns]
TESTS = combine[["nfl_id"] + LOWER_BETTER + HIGHER_BETTER].copy()

pcols = [c for c in ["nfl_id", "display_name", "nfl_position", "draft_year", "draft_round",
                     "draft_overall_pick"] if c in players.columns]
D = players[pcols].copy()
D["pgm"] = D.nfl_position.map(PGM)
D = D.dropna(subset=["pgm"]).merge(
    combine[["nfl_id", "combine_weight", "combine_height", "ngs_athleticism_score"]], on="nfl_id", how="left")

ct = load_csv("combine_tracking.csv",
              usecols=["nfl_id", "entity_type", "event_id", "drill_name", "drill_type", "time", "x", "y", "s", "a"])
ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.notna()].copy()
ct["nfl_id"] = ct.nfl_id.astype(int)
ct = ct[ct.nfl_id.isin(D.nfl_id)]
D = D[D.nfl_id.isin(ct.nfl_id)].copy()
ct["drill_name"] = ct.drill_name.replace(ALIAS)
ct["time"] = pd.to_datetime(ct.time)
ct = ct.sort_values(["event_id", "time"])
ct["t"] = (ct.time - ct.groupby("event_id").time.transform("min")).dt.total_seconds()
print(f"Players with Combine tracking in the scored position groups: {len(D)}")
print(D.nfl_position.value_counts().to_dict())

def show(df, digits=2):
    print(df.round(digits).to_string())


# ## 3. Combine Juice+
#
# Three pieces from the tracked 40-yard dash and the position drills, then one score per player.
# - **E (explosiveness):** distance at 1.5 s and speed at 1.0 s.
# - **T (top end):** the fastest 10 yards of the run.
# - **A (agility):** position drills, used only for groups whose drills agree with each other.
#
# Each piece is adjusted for body size and draft class, then shrunk by how repeatable it is.
# Section 3 ends by averaging each piece with the stopwatch tests of the same trait.

def asp_fit(speed, acc, min_speed=2.0):
    """Acceleration-speed line on the rising part of one run: acc = a0 - (a0 / v0) * speed.
    a0 = first-step power (yd/s^2), v0 = top gear (yd/s). Needs no start time."""
    m = np.isfinite(speed) & np.isfinite(acc) & (speed >= min_speed) & (acc > 0)
    if m.sum() < 5:
        return np.nan, np.nan
    slope, a0 = np.polyfit(speed[m], acc[m], 1)
    if slope >= 0 or a0 <= 0:
        return np.nan, np.nan
    return float(a0), float(-a0 / slope)


# The tracked 40: burst, top end and the acceleration-speed line for each attempt.
def feats40(a):
    s_all, x_all, t_all = a.s.to_numpy(), a.x.to_numpy(), a.t.to_numpy()
    # Distance split from the start of the clip. Clips start near rest (median first-frame speed 0.6 yd/s,
    # 7% at 1 yd/s or more), so progress is measured from the first frame, not from a speed threshold.
    prog_all = np.maximum.accumulate(np.abs(x_all - x_all[0])) + np.arange(len(x_all)) * 1e-9
    moving = np.where(s_all >= 1.0)[0]
    if len(moving) < 10 or prog_all[-1] < 36:
        return None
    t_split = float(np.interp(SPLIT_TO, prog_all, t_all) - np.interp(SPLIT_FROM, prog_all, t_all))
    i0 = moving[0]
    s, x, t = s_all[i0:], x_all[i0:], t_all[i0:]
    progress = np.maximum.accumulate(np.abs(x - x[0])) + np.arange(len(x)) * 1e-9
    elapsed = t - t[0]
    best = 0.0
    for start in np.arange(5.0, min(progress[-1] - 10, 30.0) + 1e-6, 2.5):
        mask = (progress >= start) & (progress <= start + 10)
        if mask.sum() >= 5:
            best = max(best, float(s[mask].mean()))
    smooth = pd.Series(s_all).rolling(3, center=True, min_periods=1).mean().to_numpy()
    acc = np.gradient(smooth, t_all)
    rising = np.arange(len(smooth)) <= int(np.argmax(smooth))
    a0, v0 = asp_fit(np.where(rising, smooth, np.nan), np.where(rising, acc, np.nan))
    # Mean acceleration from yard 2 to yard 10: speed gained over the split, divided by its time. Clock-free.
    v_from, v_to = np.interp(SPLIT_FROM, prog_all, smooth), np.interp(SPLIT_TO, prog_all, smooth)
    macc = (v_to - v_from) / t_split if t_split > 0 else np.nan
    return pd.Series({
        "d15": np.interp(1.5, elapsed, progress),
        "v1": np.interp(1.0, elapsed, pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()),
        "split": -t_split,             # faster = higher, so every piece reads "higher is better"
        "macc": macc,
        "fly10": best if best > 0 else np.nan,
        "peak": float(smooth.max()),
        "a0_40": a0, "v0_40": v0,
        "first_s": float(s_all[0]),
    })

# Retest reliability: how well attempt 1 predicts attempt 2, within position.
f40 = ct[ct.drill_name == "FORTY_YARD_DASH"]
a40 = f40.groupby("event_id").apply(feats40).dropna(subset=["fly10"])
a40["nfl_id"] = f40.groupby("event_id").nfl_id.first().reindex(a40.index).astype(int)
a40 = a40.sort_index()
a40["k"] = a40.groupby("nfl_id").cumcount()

print("First-frame speed of tracked 40 clips (yd/s):",
      a40.first_s.quantile([0.1, 0.5, 0.9]).round(2).to_dict(),
      f"share at 1 yd/s or more: {(a40.first_s >= 1).mean():.3f}")

E_COLS = {"split_macc": ["split", "macc"], "split": ["split"]}.get(BURST_METHOD, ["d15", "v1"])
T_COL = "peak" if TOP_METHOD == "peak" else "fly10"
# The E piece itself, per attempt: the average of its parts standardized within position. Its retest reliability is
# what the shrinkage should use (an average of two measures repeats better than either part).
_pg = a40.nfl_id.map(D.set_index("nfl_id").pgm)
a40["E_comp"] = pd.concat([a40[c_].groupby(_pg).transform(zs) for c_ in E_COLS], axis=1).mean(axis=1)
RT_FEATS = ["d15", "v1", "split", "macc", "fly10", "peak", "E_comp"]
w2 = a40[a40.k < 2].pivot(index="nfl_id", columns="k", values=RT_FEATS).dropna()
w2.columns = [f"{f}_{k}" for f, k in w2.columns]
w2 = w2.reset_index().merge(D[["nfl_id", "pgm", "draft_year"]], on="nfl_id")


def retest_reliability(w):
    """Attempt-to-attempt Spearman WITHIN position after removing class means,
    pooled by Fisher z. Returns (explosiveness reliability, top-end reliability, dict)."""
    out = {}
    for feat in RT_FEATS:
        zsum, wsum = 0.0, 0.0
        for _, d in w.groupby("pgm"):
            if len(d) < 15:
                continue
            x0 = d[feat + "_0"] - d.groupby("draft_year")[feat + "_0"].transform("mean")
            x1 = d[feat + "_1"] - d.groupby("draft_year")[feat + "_1"].transform("mean")
            r = spearmanr(x0, x1)[0]
            if np.isfinite(r) and abs(r) < 1:
                zsum += np.arctanh(r) * (len(d) - 3)
                wsum += len(d) - 3
        out[feat] = float(np.tanh(zsum / wsum)) if wsum > 0 else 0.6
    return out["E_comp"], out[T_COL], out


REL_E, REL_T, RELI = retest_reliability(w2)
print("Combine retest reliability (within position):", {k: round(v, 2) for k, v in RELI.items()})
print(f"Burst method: {BURST_METHOD} {E_COLS} (reliability {REL_E:.2f}); top-end: {T_COL} ({REL_T:.2f})")

# One row per player: average over attempts.
P40 = a40.groupby("nfl_id").agg(
    d15=("d15", "mean"), v1=("v1", "mean"), split=("split", "mean"), macc=("macc", "mean"), fly10=("fly10", "mean"),
    peak=("peak", "mean"), a0_40=("a0_40", "median"), v0_40=("v0_40", "median"),
    attempts=("fly10", "size"),
).reset_index()

# Validity check for the burst fix: each tracked 40 feature against the official 40 and 10-yard split, within position.
_val = P40.merge(D[["nfl_id", "pgm"]], on="nfl_id").merge(combine[["nfl_id", "forty", "ten_yd_split"]], on="nfl_id")
VALID_ROWS = []
for feat in ["d15", "v1", "split", "macc", "fly10", "peak", "a0_40", "v0_40"]:
    row = {"feature": feat, "retest": RELI.get(feat, np.nan)}
    for ref in ["forty", "ten_yd_split"]:
        parts = [(abs(safe_spearman(g[feat], g[ref])), g[[feat, ref]].dropna().shape[0])
                 for _, g in _val.groupby("pgm") if g[[feat, ref]].dropna().shape[0] >= 15]
        row[f"vs_{ref}"] = pooled_fisher(parts)[0]
        for grp, g in _val.groupby("pgm"):
            row[f"vs_{ref}_{grp}"] = abs(safe_spearman(g[feat], g[ref]))
    VALID_ROWS.append(row)
BURST_VALIDITY = pd.DataFrame(VALID_ROWS)
print("Tracked 40 features vs the stopwatch (|Spearman| within position, pooled):")
show_cols = ["feature", "retest", "vs_forty", "vs_ten_yd_split"]
print(BURST_VALIDITY[show_cols].round(2).to_string(index=False))

# Agility drills. A group's agility counts only if its drills agree with each other (alpha >= 0.5).
def drill_feats(a):
    s, t, acc = a.s.to_numpy(), a.t.to_numpy(), a.a.to_numpy()
    moving = np.where(s >= 1.0)[0]
    if len(moving) < 10:
        return None
    i0, i1 = moving[0], moving[-1]
    if t[i1] - t[i0] < 1.0:
        return None
    return pd.Series({"mean_s": np.nanmean(s[i0:i1 + 1]),
                      "acc95": np.nanpercentile(acc[i0:i1 + 1], 95)})


od = ct[ct.drill_name != "FORTY_YARD_DASH"]
ad = od.groupby("event_id").apply(drill_feats).dropna(subset=["mean_s"])
ad = ad.join(od.groupby("event_id").agg(
    nfl_id=("nfl_id", "first"), drill_type=("drill_type", "first"),
    drill_name=("drill_name", "first"),
))

DL_SUBS = ["PASS_RUSH_DRILL", "FRONT_START_WAVE", "FOUR_BAG_AGILITY",
           "RUN_AND_CLUB", "BODY_CONTROL", "RUN_THE_HOOP"]
FAM = {
    "DL": ("SKILL_DRILLS_DL", DL_SUBS),
    "EDGE": ("SKILL_DRILLS_DL", DL_SUBS),
    "OL": ("SKILL_DRILLS_OL", ["PASS_RUSH_DROP", "PASS_PRO", "FIVE_YARD_WAVE",
                               "OL_PULL", "SCREEN"]),
    "DB": ("SKILL_DRILLS_DB", ["BACK_PEDAL_AND_90", "BACK_PEDAL_AND_TRANSITION",
                               "BOX_DRILL", "W_DRILL", "TERYL_AUSTIN", "LINE_DRILL"]),
    "WR": ("SKILL_DRILLS_WR", ["SLANT_ROUTE", "DAGGER_ROUTE", "CURL_ROUTE",
                               "SLOT_STRIKE", "COMEBACK", "SLOT_SAIL", "POST_CORNER",
                               "SPEED_OUT", "GO_ROUTE", "RED_ZONE"]),
    "TE": ("SKILL_DRILLS_TE", ["FLAT_ROUTE", "HOOK_ROUTE", "IN_ROUTE",
                               "WHEEL_ROUTE", "CORNER_ROUTE"]),
}
COMMON = ["THREE_CONE_DRILL", "SHORT_SHUTTLE"]
print("Drill types found in Combine tracking:", sorted(ad.drill_type.dropna().unique()))

gate, rbar, agil_parts, GATE_ROWS, WIDE = {}, {}, [], [], {}
for grp, (dtype, subs) in FAM.items():
    ids = set(D.loc[D.pgm == grp, "nfl_id"])
    keep = ((ad.drill_type == dtype)
            & ad.drill_name.apply(lambda n: any(s_ in n for s_ in subs))) | ad.drill_name.isin(COMMON)
    sub = ad[keep & ad.nfl_id.isin(ids)]
    pdr = sub.groupby(["nfl_id", "drill_name"])[["mean_s", "acc95"]].mean().reset_index()
    cnt = pdr.groupby("drill_name").nfl_id.nunique()
    pdr = pdr[pdr.drill_name.isin(cnt[cnt >= MIN_PLAYERS_DRILL].index)].copy()
    if pdr.drill_name.nunique() < 2:
        gate[grp], rbar[grp] = False, np.nan
        continue
    for col in ["mean_s", "acc95"]:
        pdr[col + "_z"] = pdr.groupby("drill_name")[col].transform(zs)
    pdr["pace"] = pdr[["mean_s_z", "acc95_z"]].mean(axis=1)
    wide = pdr.pivot(index="nfl_id", columns="drill_name", values="pace")
    cors, cols = [], list(wide.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            both = wide[[cols[i], cols[j]]].dropna()
            if len(both) >= MIN_OVERLAP:
                cors.append(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
    rb = float(np.nanmean(cors)) if cors else np.nan
    kmed = float(wide.notna().sum(axis=1).median())
    alpha = kmed * rb / (1 + (kmed - 1) * rb) if np.isfinite(rb) and kmed >= 2 else np.nan
    gate[grp], rbar[grp] = bool(np.isfinite(alpha) and alpha >= GATE_ALPHA), rb
    GATE_ROWS.append((grp, len(cols), round(rb, 2) if np.isfinite(rb) else rb,
                      kmed, round(alpha, 2) if np.isfinite(alpha) else alpha, gate[grp]))
    WIDE[grp] = wide
    nd = wide.notna().sum(axis=1)
    agil_parts.append(pd.DataFrame({
        "agil": wide.mean(axis=1).where(nd >= 2), "n_drills": nd}).reset_index())
AGL = (pd.concat(agil_parts, ignore_index=True) if agil_parts
       else pd.DataFrame(columns=["nfl_id", "agil", "n_drills"]))
print("Combine agility gate (group, drills, mean r, median drills, alpha, passes):")
for row in GATE_ROWS:
    print("  ", row)

C = D.merge(P40, on="nfl_id", how="left").merge(AGL, on="nfl_id", how="left")


# Official tests that measure the same trait as each tracked piece (+1 higher is better, -1 lower is better).
OFFICIAL = {
    "E": {"vertical": 1, "broad_jump": 1, "ten_yd_split": -1},
    "T": {"forty": -1},
    "A": {"three_cone": -1, "short_shuttle": -1},
}
TRACKED_COLS = E_COLS + [T_COL, "agil"]
OFFICIAL_COLS = sorted({c_ for d_ in OFFICIAL.values() for c_ in d_} & set(TESTS.columns))


def with_tests(df):
    """Same rows as df, plus the official Combine tests."""
    return df.join(TESTS.set_index("nfl_id"), on="nfl_id")


def combine_pieces(z, c, rel_e, rel_t, rbar_map, gate_map, shrink=True, blend=True):
    """E, T, A from standardized columns z. Tracked pieces are shrunk by their reliability;
    with blend=True each is then averaged with the official tests of the same trait."""
    E = pd.concat([z[c_] for c_ in E_COLS], axis=1).mean(axis=1)
    T = z[T_COL]
    A = z["agil"].where(c.pgm.map(gate_map).fillna(False).astype(bool))
    if shrink:
        k = c.attempts.fillna(1).clip(lower=1)
        m = c.n_drills.fillna(1).clip(lower=1)
        rb = c.pgm.map(rbar_map)
        E, T = E * sb(k, rel_e), T * sb(k, rel_t)
        A = A * np.clip(m * rb / (1 + (m - 1) * rb), 0, 1)
    if blend:
        pieces = {}
        for name, tracked in (("E", E), ("T", T), ("A", A)):
            cols = [(col, sg) for col, sg in OFFICIAL[name].items() if col in z]
            if cols:
                off = pd.concat([sg * z[col] for col, sg in cols], axis=1).mean(axis=1)
                tracked = ((tracked + off) / 2).where(off.notna(), tracked)
            pieces[name] = tracked
        E, T, A = pieces["E"], pieces["T"], pieces["A"]
    return E, T, A


def build_scores(Cin, size_adj=True, shrink=True, blend=True):
    c = Cin.copy()
    src = with_tests(Cin)
    z = {col: adjust(src, col, size_adj=size_adj) for col in TRACKED_COLS + (OFFICIAL_COLS if blend else [])}
    for col in TRACKED_COLS:
        c[col + "_z"] = z[col]
    c["E"], c["T"], c["A"] = combine_pieces(z, c, REL_E, REL_T, rbar, gate, shrink, blend)
    c["raw"] = c[["E", "T", "A"]].mean(axis=1, skipna=True)
    c.loc[c["E"].isna() | c["T"].isna(), "raw"] = np.nan   # E and T are required
    for comp in ["E", "T", "A", "raw"]:
        c[comp + "_plus"] = scale100(c, comp)
    c["juice"] = c["raw_plus"]
    return c

# Build the scores. S is the main score; TRK is tracked-only, kept to show what the blend adds.
S = build_scores(C, blend=BLEND_STOPWATCH)
S["juice_unadjusted"] = build_scores(C, size_adj=False, blend=BLEND_STOPWATCH)["juice"].values
# Tracked-only version, saved so outcome.py can show what the blend adds.
TRK = build_scores(C, blend=False)
S["juice_tracked_only"] = TRK["juice"].values
S["juice_5050"] = build_scores(C, blend=True)["juice"].values
for comp in ["E", "T", "A"]:
    S[f"{comp}_trk_plus"] = TRK[f"{comp}_plus"].values
FINAL = S[S.E_plus.notna() & S.T_plus.notna()].copy()
FINAL["juice_final"] = FINAL["juice"]
print(f"Players with a Combine Juice+ score: {len(FINAL)}")

# ## 4. Game tracking
#
# Every regular-season snap in 2023 to 2025 is reduced to a few numbers per player-play: peak speed, peak lateral
# acceleration, speed at 1.0 s, distance at 1.5 s and the starting speed. This is the slow step, so the result is cached.
# Game tracking is read in chunks and only for the columns needed.

REG = set(games.loc[games.season_type == "REG", "game_id"])
KEEP_IDS = set(D.nfl_id)


def summarize_chunk(df):
    cols = KEYS + ["n_after", "peak", "alat_max", "alat_2nd", "alat_p90", "alat_max_s1", "s0", "s10", "d15",
                   "t2", "t6", "t10", "v2", "v6", "v10", "n_pre", "alat_2nd_pre"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    # Only the Combine-tracked players are needed, which makes this pass faster and the cache small.
    df = df[df.nfl_id.isin(KEEP_IDS)]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = df.sort_values(KEYS + ["time"])
    snap = df.loc[df["event"] == "ball_snap"].groupby(KEYS).time.min().rename("snap")
    df = df.join(snap, on=KEYS)
    df = df[df.time >= df.snap].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    gk = [df.game_id, df.play_id, df.nfl_id]
    df["k"] = df.groupby(KEYS).cumcount()
    om = (circ(df.groupby(KEYS)["dir"].diff()) * np.pi / 180.0) / 0.1  # rad/s
    omg = om.groupby(gk)
    om_s = pd.concat([om, omg.shift(1), omg.shift(-1)], axis=1).mean(axis=1)
    df["alat"] = (df.s * om_s.abs()).where(df.s >= 2.0)
    df["alat1"] = (df.s * om_s.abs()).where(df.s >= 1.0)
    out = df.groupby(KEYS).agg(
        n_after=("k", "size"), peak=("s", "max"), alat_max=("alat", "max"), alat_max_s1=("alat1", "max"))
    # Robust versions of the per-play peak (one noisy frame can set a max): second-highest frame, and 90th pct.
    out["alat_p90"] = df.groupby(KEYS).alat.quantile(0.9)
    rk = df.groupby(KEYS).alat.rank(ascending=False, method="first")
    out["alat_2nd"] = df.loc[rk == 2].set_index(KEYS).alat
    # Pre-throw window: frames from the snap up to the forward pass (pass plays only; NaN on other plays).
    thr = df.loc[df["event"].isin(["pass_forward", "pass_shovel"])].groupby(KEYS).time.min().rename("thr")
    pre = df.join(thr, on=KEYS)
    pre = pre[pre.thr.notna() & (pre.time <= pre.thr)]
    out["n_pre"] = pre.groupby(KEYS).size()
    rk_pre = pre.groupby(KEYS).alat.rank(ascending=False, method="first")
    out["alat_2nd_pre"] = pre.loc[rk_pre == 2].set_index(KEYS).alat
    out["s0"] = df.loc[df.k == 0].set_index(KEYS).s
    out["s10"] = df.loc[df.k == 10].set_index(KEYS).s
    # Distance covered in the first 1.5 s after the snap (same idea as the Combine d15).
    p0 = df.loc[df.k == 0].set_index(KEYS)[["x", "y"]]
    p15 = df.loc[df.k == 15].set_index(KEYS)[["x", "y"]]
    out["d15"] = np.hypot(p15.x - p0.x.reindex(p15.index), p15.y - p0.y.reindex(p15.index))
    # Clock-free burst: time and speed when the player first gets 2 and 10 yards from his snap spot (first 4 s).
    w = df[df.k <= BURST_WINDOW].copy()
    w = w.join(p0.rename(columns={"x": "x0", "y": "y0"}), on=KEYS)
    w["d"] = np.hypot(w.x - w.x0, w.y - w.y0)
    w["d"] = w.groupby(KEYS, sort=False).d.cummax()
    wg = w.groupby(KEYS, sort=False).s
    w["sm"] = pd.concat([w.s, wg.shift(1), wg.shift(-1)], axis=1).mean(axis=1)
    w = w.reset_index(drop=True)
    for mark in (2, 6, 10):
        hit = w[w.d >= mark].groupby(KEYS, sort=False).head(1)
        hit = hit[hit.k > 0]
        prev = w.loc[hit.index - 1]
        frac = ((mark - prev.d.to_numpy()) / np.maximum(hit.d.to_numpy() - prev.d.to_numpy(), 1e-6)).clip(0, 1)
        key = pd.MultiIndex.from_frame(hit[KEYS])
        out[f"t{mark}"] = pd.Series((prev.k.to_numpy() + frac) / 10.0, index=key)
        out[f"v{mark}"] = pd.Series(prev.sm.to_numpy() + frac * (hit.sm.to_numpy() - prev.sm.to_numpy()), index=key)
    return out.reset_index()[cols]


def game_pass():
    parts = []
    for yr in SEASONS:
        name = f"game_tracking_{yr}.csv"
        print(f"  Processing {name} in chunks...")
        left = None
        for chunk in pd.read_csv(
            get_file(name), usecols=KEYS + ["time", "s", "dir", "event", "x", "y"],
            chunksize=CHUNK_SIZE,
        ):
            chunk = chunk[chunk.game_id.isin(REG)]
            if left is not None:
                chunk = pd.concat([left, chunk], ignore_index=True)
            if len(chunk) == 0:
                left = None
                continue
            last = chunk.iloc[-1]
            same = (chunk.game_id == last.game_id) & (chunk.play_id == last.play_id)
            left, body = chunk[same], chunk[~same]
            parts.append(summarize_chunk(body))
        parts.append(summarize_chunk(left))
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return out.drop_duplicates(KEYS)


cache_name = "game_play_table_v2.csv" if REUSE_JUICE_PY_CACHE else "game_play_table_v7.csv"
cache_file = CACHE / cache_name
if cache_file.exists():
    print(f"Using cached play table: {cache_file}")
    PT0 = pd.read_csv(cache_file)
else:
    PT0 = game_pass()
    try:
        PT0.to_csv(cache_file, index=False)
    except OSError as e:
        cache_file.unlink(missing_ok=True)
        print(f"Could not save the cache ({e}). Continuing without it; free disk space to keep it.")
print(f"Player-plays in table (all players): {len(PT0):,}")

# ## 5. Who is in the game data, and why
#
# Plays are attached to the Combine-tracked players and to each play's role (route, coverage job, alignment,
# win probability). Three filters follow:
# - **Dead plays** (shorter than 3 s) carry no movement, so they are dropped.
# - **Decided games** (home win probability under 5% or over 95%) are dropped *unless the player has no other snaps*.
# - **Snap floor.** Game Juice+ needs MIN_GAME_PLAYS usable plays (default 50). Players under it are counted, not scored.
#   The 95th percentile of a few plays sits below average, and shrinkage did not fully fix that (audit item 4).

pp = load_csv(
    "player_play.csv",
    usecols=KEYS + ["lined_up_position", "route_ran", "player_get_off", "coverage_assignment",
                    "pass_rushers_encountered", "pre_snap_home_team_win_probability"],
)
PT_ALL = (
    PT0.merge(games[["game_id", "season"]], on="game_id")
       .merge(D[["nfl_id", "pgm", "nfl_position", "draft_year"]], on="nfl_id")
       .merge(pp, on=KEYS, how="left")
)
ids_any = set(PT_ALL.nfl_id)

# Dead plays out first, so a player is judged on plays that carry movement.
PT = PT_ALL[PT_ALL.n_after >= MIN_FRAMES].copy()
ids_usable = set(PT.nfl_id)

# Decided games out, unless that would remove every snap the player has.
wp = PT.pre_snap_home_team_win_probability
PT["blowout"] = (wp <= GARBAGE_WP) | (wp >= 1 - GARBAGE_WP)
has_clean = (~PT.blowout).groupby(PT.nfl_id).transform("any")
PT["blowout_only_player"] = ~has_clean
if DROP_BLOWOUTS:
    PT = PT[~PT.blowout | PT.blowout_only_player].copy()
print(f"Players who only have decided-game snaps (kept anyway): {PT.loc[PT.blowout_only_player, 'nfl_id'].nunique()}")

PT["role"] = (
    (PT.pgm.isin(["WR", "TE"]) & PT.route_ran.notna())
    | (PT.pgm.isin(["DL", "EDGE"]) & PT.player_get_off.notna())
    | ((PT.pgm == "DB") & PT.coverage_assignment.notna() & (PT.coverage_assignment != "NO_ASSIGNMENT"))
    | ((PT.pgm == "OL") & PT.pass_rushers_encountered.notna())
)
PT["align"] = np.where(PT.pgm.isin(["DL", "EDGE", "DB", "WR"]), PT.lined_up_position.fillna("UNK"), "ALL")
PT["len_s"] = np.minimum(PT.n_after // 10, 10)


def coverage_family(a):
    """Collapse coverage assignments into the jobs a DB actually has. Snaps with no assignment get their own key."""
    if pd.isna(a) or a == "NO_ASSIGNMENT":
        return "NO_JOB"
    a = str(a)
    if a == "MAN":
        return "MAN"
    if a.startswith("DEEP_THIRD") or a == "DEEP_FREE":
        return "DEEP_THIRD"
    if a.startswith("QUARTERS"):
        return "QUARTERS"
    if a.startswith(("DEEP_HALF", "PREVENT")):
        return "DEEP_HALF"
    if a.startswith(("CURL_FLAT", "FLAT", "HOOK_CURL")) or a == "HOLE":
        return "SHORT_ZONE"
    return "OTHER"


# One "route" key for everyone: the route for WR/TE, the coverage job for DBs, nothing for the rest.
PT["route"] = np.where(PT.pgm.isin(["WR", "TE"]), PT.route_ran.fillna("NO_ROUTE"),
                       np.where(PT.pgm == "DB", PT.coverage_assignment.map(coverage_family), "NONE"))
print("DB coverage keys:", PT.loc[PT.pgm == "DB", "route"].value_counts().to_dict())
leftover = PT.loc[(PT.pgm == "DB") & (PT.route == "OTHER"), "coverage_assignment"].value_counts()
print("Assignments still in the OTHER bucket (check these):", leftover.to_dict())

# ### Roster accounting
# Every player with Combine tracking ends up in exactly one row.

n_tracked = len(D)
rows = [("Players with Combine tracking", n_tracked),
        ("  no regular-season game snap in the tracking data", n_tracked - len(ids_any & set(D.nfl_id))),
        ("  only dead plays (under 3 s)", len(ids_any - ids_usable)),
        ("  at least one usable game snap", len(ids_usable))]
ACCOUNT_ROWS = rows
PT_PLAYS = PT.groupby("nfl_id").size()

# ## 6. Context adjustment and game components
#
# A burst or a top speed means little without context, so each play is compared with plays at the same position,
# alignment, route or coverage job, and play length. The player's score is the 95th percentile of those adjusted values.

def demean(df, col, key_sets, min_cell=100):
    mu = df.groupby(key_sets[-1])[col].transform("mean")
    for keys in reversed(key_sets[:-1]):
        m = df.groupby(keys)[col].transform("mean")
        n = df.groupby(keys)[col].transform("count")
        mu = m.where(n >= min_cell, mu)
    return df[col] - mu


def backfit(df, col, factors, min_cell=100, base=("nfl_position",)):
    base = list(base)
    tmp = df[base + list(factors)].copy()
    tmp["_r"] = demean(df, col, [base], min_cell)
    for _ in range(1 if len(factors) <= 1 else 3):
        for f in factors:
            tmp["_r"] = demean(tmp, "_r", [base + [f], base], min_cell)
    return tmp["_r"]


def burst_mask(P):
    """Plays that are real standing-start bursts in the player's role."""
    base = P.role & (P.s0 < 1.0)
    if GAME_BURST == "snap":
        return base & P.s10.notna() & P.d15.notna()
    return base & P.t2.notna() & P.t10.notna() & (P.t10 > P.t2)


def agility_column(P):
    """Per-play game agility value. With GAME_AGIL_WINDOW=prethrow, WR/TE/DB pass plays use the snap-to-throw window
    (second-highest frame, the same rule as AGIL_COL="second"); a throw sooner than PRE_THROW_MIN_FRAMES leaves no value."""
    v = P[AGIL_COL].copy()
    if GAME_AGIL_WINDOW == "prethrow" and "n_pre" in P:
        pass_play = P.pgm.isin(PRE_THROW_GROUPS) & P.n_pre.notna()
        pre_ok = P.n_pre >= PRE_THROW_MIN_FRAMES
        v = v.where(~pass_play, P.alat_2nd_pre.where(pre_ok))
    return v


def add_adjusted(frame):
    P = frame.copy()
    P["agil_play"] = agility_column(P)
    factors = ["route", "len_s", "align"]
    P["peak_c"], P["alat_c"], P["s10_c"], P["d15_c"], P["expl_c"] = (
        np.nan, np.nan, np.nan, np.nan, np.nan)
    open_play = P.n_after >= MIN_FRAMES_OPEN
    if open_play.any():
        P.loc[open_play, "peak_c"] = backfit(P[open_play], "peak", factors)
        P.loc[open_play, "alat_c"] = backfit(P[open_play], "agil_play", factors)
    ok = burst_mask(P)
    if ok.any() and GAME_BURST != "snap":
        # Same two numbers as the Combine E piece: yard 2-10 split (faster = higher) and mean acceleration over it.
        cells = [["nfl_position", "align", "route"], ["nfl_position", "align"], ["nfl_position"]]
        P["gsplit"] = -(P.t10 - P.t2)
        P["gmacc"] = (P.v10 - P.v2) / (P.t10 - P.t2)
        cols_ = ("gsplit",) if GAME_BURST == "clockfree_split" else ("gsplit", "gmacc")
        parts = []
        for col in cols_:
            c_ = demean(P[ok], col, cells)
            c_ = c_ / c_.groupby(P.loc[ok, "pgm"]).transform("std")
            P.loc[ok, col + "_c"] = c_
            parts.append(c_)
        # Per-play average, kept for the play-level models downstream. The player score takes each measure's best
        # effort (95th percentile) first and then averages, which is more repeatable than the q95 of the average.
        P.loc[ok, "expl_c"] = sum(parts) / len(parts)
    elif ok.any():
        cells = [["nfl_position", "align", "route"], ["nfl_position", "align"], ["nfl_position"]]
        P.loc[ok, "s10_c"] = demean(P[ok], "s10", cells)
        P.loc[ok, "d15_c"] = demean(P[ok], "d15", cells)
        # Put the two on one scale (divide by the spread within position group), then average.
        # This mirrors the Combine E piece, which averages distance at 1.5 s and speed at 1.0 s.
        sd_s = P.loc[ok].groupby("pgm").s10_c.transform("std")
        sd_d = P.loc[ok].groupby("pgm").d15_c.transform("std")
        P.loc[ok, "expl_c"] = (P.loc[ok, "s10_c"] / sd_s + P.loc[ok, "d15_c"] / sd_d) / 2
    return P


def q95(s):
    return s.quantile(0.95)


def game_components(P, min_plays=1, min_bursts=0, strict_join=False):
    """One row per player: top speed, agility, explosiveness and counts (P already adjusted).
    The defaults keep every player with at least one usable play."""
    burst = P[burst_mask(P)]
    gb = burst.groupby("nfl_id").agg(nb=("expl_c", "size"), g_expl=("expl_c", q95))
    bcols = [c for c in EXPL_PARTS if c in burst]
    if bcols:
        gb["g_expl"] = burst.groupby("nfl_id")[bcols].quantile(0.95).mean(axis=1)
    open_ = P[P.n_after >= MIN_FRAMES_OPEN]
    ga = open_.groupby("nfl_id").agg(
        npl=("peak_c", "size"), n_alat=("alat_c", "count"),
        g_top=("peak_c", q95), g_agil=("alat_c", q95),
    )
    G = ga.join(gb, how="inner" if strict_join else "left")
    G["nb"] = G.nb.fillna(0)
    G = G[(G.nb >= min_bursts) & (G.npl >= min_plays)].reset_index()
    G = G.merge(
        D[["nfl_id", "display_name", "nfl_position", "pgm", "draft_year", "draft_overall_pick",
           "combine_weight", "combine_height"]], on="nfl_id",
    )
    for c in ["g_top", "g_agil"]:
        G[c] = G[c] - G.groupby("nfl_position")[c].transform("mean")
    return G


PTa = add_adjusted(PT)
# Play-level, context-adjusted movement for every usable play (no snap floor). juice_transfer.py uses it for the
# mixed model and the snap-takeover test.
save(PTa[KEYS + ["season", "pgm", "nfl_position", "n_after", "role", "align", "route", "s0",
                 "peak_c", "alat_c", "expl_c"]], "play_level_movement.csv")
G_ALL = game_components(PTa)
print(f"Players with a game score: {len(G_ALL)} of {n_tracked}")
print(f"  with any burst (explosiveness):   {(G_ALL.nb > 0).sum()}")
print(f"  plays per player (median / min):  {G_ALL.npl.median():.0f} / {G_ALL.npl.min():.0f}")

# ## 7. Reliability, shrinkage and Game Juice+
#
# How repeatable is each game piece? Split each well-sampled player's plays in two random halves and correlate them.
# Each piece is then multiplied by `sb(n, r)`, the reliability of an average of n plays (Spearman-Brown). A player with
# few plays keeps only a small share of his z-score, so low-snap players sit near 100.
# The `confidence` column reports that share, so thin scores can be shown faded or filtered in any chart.
# Known limit: the 95th percentile of very few plays is lower on average than that of many plays; shrinkage reduces
# but does not fully remove this.

def split_half(P, ids, col, stat, reps=30, burst=False):
    sub = P[P.nfl_id.isin(ids)]
    if burst:
        sub = sub[burst_mask(sub)]
    else:
        sub = sub[sub.n_after >= MIN_FRAMES_OPEN]
    cols = list(col) if isinstance(col, (list, tuple)) else [col]   # several columns: stat of each, then average
    sub = sub[["nfl_id"] + cols].dropna().reset_index(drop=True)
    out = []
    for _ in range(reps):
        sub["h"] = rng.integers(0, 2, len(sub))
        g = sub.groupby(["nfl_id", "h"])[cols].agg(stat).mean(axis=1).unstack("h").dropna()
        if len(g) < 10:
            continue
        r = safe_spearman(g[0], g[1])
        if np.isfinite(r) and r > -0.99:
            out.append(2 * r / (1 + r))
    return float(np.nanmean(out)) if out else np.nan


G_REL = game_components(PTa, *RELIABILITY_MIN, strict_join=True)
if len(G_REL) < 60:
    print("Few well-sampled players; estimating reliabilities on all players instead.")
    G_REL = G_ALL
rel_ids = G_REL.nfl_id
GREL = {
    "top": split_half(PTa, rel_ids, "peak_c", q95),
    "expl": split_half(PTa, rel_ids, EXPL_PARTS or "expl_c", q95, burst=True),
    "agil": split_half(PTa, rel_ids, "alat_c", q95),
}
K_TYP = {
    "top": float(G_REL.npl.median()),
    "expl": float(G_REL.nb.replace(0, np.nan).median()),
    "agil": float(G_REL.n_alat.median()),
}
GREL1 = {k: single_obs_reliability(GREL[k], K_TYP[k]) for k in GREL}
GATE_G = bool(np.isfinite(GREL["agil"]) and GREL["agil"] >= 0.5)
print("Split-half reliability (full sample):", {k: round(v, 2) for k, v in GREL.items()})
print("Typical plays per well-sampled player:", {k: round(v) for k, v in K_TYP.items()})
print("Single-play reliability:", {k: round(v, 3) for k, v in GREL1.items()})
print("Game agility included:", GATE_G)

COMP_COL = {"top_z": ("g_top", "npl", "top"),
            "expl_z": ("g_expl", "nb", "expl"),
            "agil_z": ("g_agil", "n_alat", "agil")}
USE = ["top_z", "expl_z"] + (["agil_z"] if GATE_G else [])
_w = {z: max(float(np.nan_to_num(GREL[COMP_COL[z][2]])), 0.0) for z in USE}
_tot = sum(_w.values())
C_WEIGHTS = {z: (w / _tot if _tot > 0 else 1 / len(USE)) for z, w in _w.items()}
print("Game Juice+ component weights:", {k: round(v, 2) for k, v in C_WEIGHTS.items()})


def c_score(G, how="adjust"):
    """Game Juice+ from a component frame. how='adjust' removes body size and draft class (main score);
    how='zscore' only standardizes within position. `confidence` is the average shrink factor.

    Each piece is shrunk toward 100 by its own reliability at the player's play count. The final score is the
    reliability-weighted average of the shrunk pieces on the same 100 scale, with no second rescale, so it always
    lies inside the range of its own pieces (audit item 4)."""
    g = G.copy()
    for z, (col, _, _) in COMP_COL.items():
        g[z] = adjust(g, col) if how == "adjust" else g.groupby("pgm")[col].transform(zs)
    # Combine agility is only scored where its drills agree, so game agility is left out there too.
    g.loc[~g.pgm.map(gate).fillna(False).astype(bool), "agil_z"] = np.nan
    num, den, conf = 0, 0, 0
    for z in USE:
        _, ncol, key = COMP_COL[z]
        k = pd.Series(sb(g[ncol].fillna(0), GREL1[key]), index=g.index)
        shrunk = g[z] * k
        g[z + "_shrunk"] = shrunk
        num = num + shrunk.fillna(0) * C_WEIGHTS[z]
        den = den + shrunk.notna() * C_WEIGHTS[z]
        conf = conf + k.where(shrunk.notna(), 0) * C_WEIGHTS[z]
    g["raw"] = num / den.replace(0, np.nan)
    g["confidence"] = conf / den.replace(0, np.nan)
    g["game_juice"] = 100 + 10 * g["raw"]
    for z, name in [("top_z", "g_top"), ("expl_z", "g_expl"), ("agil_z", "g_agil")]:
        g[name + "_plus_unshrunk"] = 100 + 10 * g[z]
        g[name + "_plus"] = 100 + 10 * g[z + "_shrunk"] if z + "_shrunk" in g else np.nan
    return g


def piece_alpha(g, cols):
    """Cronbach-style alpha of the game pieces within each position group (is Game Juice+ one trait?)."""
    rows = []
    for grp, d in g.groupby("pgm"):
        d = d[[c for c in cols if d[c].notna().sum() >= 15]].dropna()
        if d.shape[1] < 2 or len(d) < 15:
            continue
        cm = d.corr(method="spearman").to_numpy()
        k = cm.shape[0]
        rb = cm[np.triu_indices(k, 1)].mean()
        rows.append({"pgm": grp, "n": len(d), "pieces": k, "mean_r": rb, "alpha": k * rb / (1 + (k - 1) * rb)})
    return pd.DataFrame(rows)


# Snap floor (audit item 4): a 95th percentile of a handful of plays sits well below average, so players under the
# floor are counted and reported but not given a game score.
G_FLOOR = G_ALL[G_ALL.npl >= MIN_GAME_PLAYS].copy()
GAME = c_score(G_FLOOR)
GAME = GAME[GAME.game_juice.notna()].copy()
print(f"Game Juice+ scored for {len(GAME)} players with {MIN_GAME_PLAYS}+ usable plays "
      f"({(G_ALL.npl < MIN_GAME_PLAYS).sum()} below the floor); median confidence {GAME.confidence.median():.2f}")
GAME_ALPHA = piece_alpha(GAME, ["top_z", "expl_z", "agil_z"])
print("Do the game pieces measure one trait? (alpha by group; 0.5 is the gate used for Combine drills)")
show(GAME_ALPHA)
save(GAME_ALPHA, "game_piece_alpha.csv")
# Bias check: mean unshrunk piece z by play count. Near 0 everywhere means the floor removed the small-sample drag.
_bias = GAME.assign(bucket=pd.cut(GAME.npl, [0, 99, 199, 599, 100000], labels=["50-99", "100-199", "200-599", "600+"]))
GAME_BIAS = _bias.groupby("bucket", observed=True)[["top_z", "expl_z", "agil_z", "game_juice"]].mean()
GAME_BIAS["n"] = _bias.groupby("bucket", observed=True).size()
print("Mean piece z and Game Juice+ by usable plays:")
show(GAME_BIAS)
save(GAME_BIAS.reset_index(), "game_bias_by_plays.csv")
save(GAME.drop(columns=["raw"]), "game_juice_scores.csv")

# Accounting table, final rows.
ACCOUNT_ROWS.append((f"  under the {MIN_GAME_PLAYS}-play floor (not scored)", int((G_ALL.npl < MIN_GAME_PLAYS).sum())))
ACCOUNT_ROWS.append((f"  scored (Game Juice+, {MIN_GAME_PLAYS}+ plays)", len(GAME)))
ACCOUNT = pd.DataFrame(ACCOUNT_ROWS, columns=["group", "players"])
ACCOUNT["share_of_tracked"] = (100 * ACCOUNT.players / n_tracked).round(1)
show(ACCOUNT)
save(ACCOUNT, "roster_accounting.csv")
print("Scored players by number of usable plays:")
print(pd.cut(GAME.npl, [0, 9, 49, 199, 100000], labels=["1-9", "10-49", "50-199", "200+"]).value_counts().sort_index().to_dict())
by_pos = D.groupby("pgm").nfl_id.nunique().to_frame("tracked").join(GAME.groupby("pgm").nfl_id.nunique().rename("scored"))
show(by_pos.fillna(0).astype(int), 0)
small = by_pos.index[by_pos.tracked < 15].tolist()
if small:
    print(f"Too few Combine-tracked players to score on their own (under 15): {small}")

# Season-to-season persistence of Game Juice+: the ceiling for any Combine link.
persist_rows, season_scores = [], []
for seas, sub in PT.groupby("season"):
    Gs = game_components(add_adjusted(sub), *SEASON_MIN, strict_join=True)
    if len(Gs) < 10:
        continue
    Gs = c_score(Gs, how="zscore")
    Gs["season"] = seas
    season_scores.append(Gs[["nfl_id", "season", "raw"]])
if season_scores:
    wide_s = pd.concat(season_scores).pivot(index="nfl_id", columns="season", values="raw")
    cols_s = sorted(wide_s.columns)
    for a_, b_ in zip(cols_s[:-1], cols_s[1:]):
        x = wide_s[[a_, b_]].dropna()
        if len(x) >= 10:
            persist_rows.append({"season_pair": f"{a_}->{b_}", "n_players": len(x),
                                 "spearman": safe_spearman(x[a_], x[b_])})
persist_df = pd.DataFrame(persist_rows)
show(persist_df)

# Score bands and stopwatch comparison scores (RAS-style is a percentile average of the official tests).
BAND_EDGES = [-np.inf, LOW_JUICE, BELOW_AVG, ELITE, np.inf]
BAND_LABELS = ["Low (<90)", "Below avg (90-100)", "Above avg (100-110)", "Elite (110+)"]

# --- comparison frame: Combine tracking score vs stopwatch scores -----------------
CMP = FINAL.merge(TESTS, on="nfl_id", how="left")


def ras_style(df):
    out = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("pgm"):
        pct = {c_: d[c_].rank(pct=True, ascending=(c_ not in LOWER_BETTER))
               for c_ in LOWER_BETTER + HIGHER_BETTER
               if c_ in d.columns and d[c_].notna().sum() >= 15}
        if pct:
            Pp = pd.DataFrame(pct)
            out.loc[d.index] = (10 * Pp.mean(axis=1)).where(Pp.notna().sum(axis=1) >= 4)
    return out


def adj_or_nan(df, col, sign=1):
    return sign * adjust(df, col) if col in df.columns else pd.Series(np.nan, index=df.index)


CMP["ras_style"] = ras_style(CMP)
CMP["ngs_adj"] = adj_or_nan(CMP, "ngs_athleticism_score")
CMP["ras_adj"] = adj_or_nan(CMP, "ras_style")
CMP["forty_adj"] = adj_or_nan(CMP, "forty", -1)
CMP["split_adj"] = adj_or_nan(CMP, "ten_yd_split", -1)
CMP["forty_neg"] = -CMP["forty"] if "forty" in CMP.columns else np.nan

# Fully held-out Combine formula: each draft class is scored by a formula fitted without it.
def gate_fn(ids):
    gt, rb_ = {g: False for g in FAM}, {g: np.nan for g in FAM}
    for g, wide in WIDE.items():
        w = wide[wide.index.isin(ids)]
        cors = []
        for i in range(w.shape[1]):
            for j in range(i + 1, w.shape[1]):
                both = w.iloc[:, [i, j]].dropna()
                if len(both) >= MIN_OVERLAP:
                    cors.append(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        rb = float(np.nanmean(cors)) if cors else np.nan
        k = float(w.notna().sum(axis=1).median()) if len(w) else 0
        alpha = k * rb / (1 + (k - 1) * rb) if np.isfinite(rb) and k >= 2 else np.nan
        gt[g], rb_[g] = bool(np.isfinite(alpha) and alpha >= GATE_ALPHA), rb
    return gt, rb_


def adjust_frozen(df, col, train):
    """Size model fit on the training classes only; the held-out class is standardized
    against itself."""
    out = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("pgm"):
        d = d.dropna(subset=[col, "combine_weight", "combine_height"])
        tr, te = d[train.loc[d.index]], d[~train.loc[d.index]]
        if len(tr) < 15 or len(te) < 3:
            continue

        def feats(x):
            return np.column_stack([x.combine_weight, x.combine_height,
                                    (x.combine_weight - tr.combine_weight.mean()) ** 2 / 100])
        if len(tr) >= 25:
            # Same size model as adjust() (plain least squares), so the held-out check tests the formula in use.
            p = LinearRegression().fit(feats(tr), tr[col]).predict(feats(te))
        else:
            p = tr[col].mean()
        r = te[col].values - p
        if np.std(r) > 0:
            out.loc[te.index] = (r - np.mean(r)) / np.std(r)
    return out


JF = pd.Series(np.nan, index=C.index)
CT = with_tests(C)
for cls in sorted(C.draft_year.dropna().unique()):
    train = C.draft_year != cls
    ids_tr = set(C.nfl_id[train])
    RE_f, RT_f, _ = retest_reliability(w2[w2.nfl_id.isin(ids_tr)])
    gate_f, rbar_f = gate_fn(ids_tr)
    z = {col: adjust_frozen(CT, col, train)
         for col in TRACKED_COLS + (OFFICIAL_COLS if BLEND_STOPWATCH else [])}
    E, T, A = combine_pieces(z, C, RE_f, RT_f, rbar_f, gate_f, blend=BLEND_STOPWATCH)
    raw = pd.concat([E, T, A], axis=1).mean(axis=1).where(E.notna() & T.notna())
    te = ~train & raw.notna()
    g_ = raw[te].groupby(C.pgm[te])
    JF[te] = 100 + 10 * (raw[te] - g_.transform("mean")) / g_.transform("std")
JF_MAP = dict(zip(C.nfl_id[JF.notna()], JF[JF.notna()]))

# ---------------------------------------------------------------------
# Analysis. Question: is the athleticism a player shows at the Combine, relative to his
# position group, the same as the athleticism he shows in NFL games, relative to his
# position group? And does Juice+ carry that over better than NGS, RAS-style and the 40?
# ---------------------------------------------------------------------

def hdr(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# One row per player with both scores, plus every comparison score.
M = FINAL[["nfl_id", "juice_final", "juice_tracked_only", "juice_5050", "juice_unadjusted", "E_plus", "T_plus", "A_plus",
           "E_trk_plus", "T_trk_plus", "A_trk_plus"]].merge(
    GAME[["nfl_id", "display_name", "nfl_position", "pgm", "draft_year", "draft_overall_pick", "game_juice",
          "g_top_plus", "g_expl_plus", "g_agil_plus", "g_top_plus_unshrunk", "g_expl_plus_unshrunk",
          "g_agil_plus_unshrunk", "npl", "nb", "confidence"]], on="nfl_id")
M = M.merge(CMP[["nfl_id", "combine_weight", "forty", "forty_neg", "ngs_athleticism_score", "ras_style",
                 "ngs_adj", "ras_adj", "forty_adj"]], on="nfl_id", how="left")
M["speed_score"] = M.combine_weight * 200 / M.forty ** 4
M["juice_heldout"] = M.nfl_id.map(JF_MAP)
M["g_above"] = M.game_juice >= BELOW_AVG

SCORES = [("Juice+", "juice_final"),
          ("Juice+ (held-out formula)", "juice_heldout"),
          ("NGS athleticism", "ngs_athleticism_score"),
          ("RAS-style", "ras_style"),
          ("Official 40", "forty_neg"),
          ("Speed score (weight x 40)", "speed_score"),
          ("NGS, size-adjusted", "ngs_adj"),
          ("RAS-style, size-adjusted", "ras_adj"),
          ("Official 40, size-adjusted", "forty_adj")]
NAME = dict((c, n) for n, c in SCORES)

# Everything is compared as a within-position percentile, so groups on different scales mix fairly.
for _, c in SCORES + [("game", "game_juice"), ("weight", "combine_weight")]:
    M["X_" + c] = M.groupby("pgm")[c].rank(pct=True)
# V: players who have every score, so each comparison uses exactly the same people.
V = M.dropna(subset=[c for _, c in SCORES]).copy()


def pr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def boot(x, y, reps=BOOT, seed=5):
    r_ = np.random.default_rng(seed)
    idx = r_.integers(0, len(x), (reps, len(x)))
    return np.array([pr(x[i], y[i]) for i in idx])


# ---------------------------------------------------------------------
hdr("1. WHO IS COVERED")
# ---------------------------------------------------------------------
print(f"Every player with Combine tracking ends in one row below. Game Juice+ needs {MIN_GAME_PLAYS}+ usable plays.")
show(ACCOUNT.set_index("group"), 1)
cov = (D.groupby("pgm").nfl_id.nunique().to_frame("tracked")
        .join(GAME.groupby("pgm").nfl_id.nunique().rename("game_scored"))
        .join(M.groupby("pgm").nfl_id.nunique().rename("scored_both"))
        .join(V.groupby("pgm").nfl_id.nunique().rename("have_all_scores")))
print("\nBy position group:")
show(cov.fillna(0).astype(int), 0)
print(f"\nPlayers with both Combine and Game Juice+: {len(M)}. "
      f"Players who also have NGS, RAS-style and a 40 (used for head-to-head): {len(V)}.")
print("Usable plays among scored players (median / min):", f"{GAME.npl.median():.0f} / {GAME.npl.min():.0f}")

# ---------------------------------------------------------------------
hdr("2. IS THE SCORE FREE OF BODY SIZE?")
# ---------------------------------------------------------------------
print("Correlation of each score with body weight, within position group (pooled).")
print("0 means size tells you nothing about the score. Far from 0 means a big player starts with a head start.")
size_rows = []
for name, c in SCORES + [("Game Juice+ (the target)", "game_juice")]:
    d = M.dropna(subset=[c, "combine_weight"])
    parts = [(safe_spearman(g[c], g.combine_weight), len(g)) for _, g in d.groupby("pgm") if len(g) >= 15]
    r_, lo, hi = pooled_fisher(parts)
    size_rows.append({"score": name, "corr_with_weight": r_, "ci_low": lo, "ci_high": hi, "n": len(d)})
SIZE = pd.DataFrame(size_rows).set_index("score")
show(SIZE)

print("\nDoes the Combine-to-game link hold for light, middle and heavy players of the SAME position?")
print("Players are split into weight thirds inside their own position group. Cells are correlations with Game Juice+.")
V["size_third"] = ""
for _, d_ in V.groupby("pgm"):
    V.loc[d_.index, "size_third"] = pd.qcut(d_.combine_weight.rank(method="first"), 3,
                                            labels=["light", "middle", "heavy"]).astype(str)
third_rows = {}
for name, c in SCORES:
    row = {t: pr(g["X_" + c], g.X_game_juice) for t, g in V.groupby("size_third")}
    row["spread (max-min)"] = max(row.values()) - min(row.values())
    third_rows[name] = row
THIRDS = pd.DataFrame(third_rows).T[["light", "middle", "heavy", "spread (max-min)"]]
show(THIRDS)
print("n per third:", V.size_third.value_counts().to_dict())

# ---------------------------------------------------------------------
hdr("3. DOES COMBINE ATHLETICISM CARRY OVER TO GAME ATHLETICISM?")
# ---------------------------------------------------------------------
print("Rank correlation between a player's Combine score and his Game Juice+, same 'within his position group' scale.")
carry = []
for name, c in SCORES:
    x, y = V["X_" + c].to_numpy(), V.X_game_juice.to_numpy()
    b = boot(x, y)
    carry.append({"score": name, "corr_with_game_juice": pr(x, y), "ci_low": np.percentile(b, 2.5),
                  "ci_high": np.percentile(b, 97.5), "n": len(V)})
CARRY = pd.DataFrame(carry).set_index("score")
show(CARRY, 3)

print("\nJuice+ on EVERYONE scored (no need for NGS or a 40), by position group:")
rows_g = []
for grp, d in list(M.groupby("pgm")) + [("ALL", M)]:
    if len(d) < 10:
        continue
    r_ = safe_spearman(d.juice_final, d.game_juice) if grp != "ALL" else pr(d.X_juice_final, d.X_game_juice)
    lo, hi = fisher_ci(r_, len(d))
    rows_g.append({"group": grp, "n": len(d), "corr": r_, "ci_low": lo, "ci_high": hi})
BYGROUP = pd.DataFrame(rows_g).set_index("group")
show(BYGROUP, 3)

print("\nSame, by how much game data backs the score (thin scores are pulled toward 100 by design):")
M["plays_bucket"] = pd.cut(M.npl, [0, 49, 199, 100000], labels=["1-49 plays", "50-199 plays", "200+ plays"])
bk = []
for b_, d in M.groupby("plays_bucket", observed=True):
    r_ = pr(d.X_juice_final, d.X_game_juice) if len(d) > 5 else np.nan
    bk.append({"plays": b_, "n": len(d), "corr": r_})
show(pd.DataFrame(bk).set_index("plays"), 3)

print("\nWhat a Combine Juice+ band says about the game (share above average, mean Game Juice+):")
M["band"] = pd.cut(M.juice_final, [-np.inf, LOW_JUICE, BELOW_AVG, ELITE, np.inf],
                   labels=["<90", "90-100", "100-110", "110+"], right=False)
BAND = M.groupby("band", observed=True).agg(n=("nfl_id", "size"), pct_above_avg=("g_above", lambda s: 100 * s.mean()),
                                            mean_game_juice=("game_juice", "mean"))
show(BAND, 1)

print("\nSeason-to-season repeatability of Game Juice+ (the ceiling for any Combine link):")
show(persist_df.set_index("season_pair") if len(persist_df) else pd.DataFrame())

# ---------------------------------------------------------------------
hdr("4. JUICE+ VS NGS, RAS-STYLE AND THE 40 (same players, paired bootstrap)")
# ---------------------------------------------------------------------
print("Edge = Juice+ correlation minus the other score's correlation, with a 95% interval.")
print("'Clear' means the interval stays above 0. Variants marked size-adjusted get the same size and class correction Juice+ gets.")
xj = V.X_juice_final.to_numpy()
yg = V.X_game_juice.to_numpy()
r_b = np.random.default_rng(9)
idx = r_b.integers(0, len(V), (BOOT, len(V)))
edge_rows = []
for name, c in SCORES[1:]:
    if c == "juice_heldout":
        continue
    xo = V["X_" + c].to_numpy()
    d_all = np.array([pr(xj[i], yg[i]) - pr(xo[i], yg[i]) for i in idx])
    est = pr(xj, yg) - pr(xo, yg)
    lo, hi = np.percentile(d_all, [2.5, 97.5])
    verdict = "clear edge" if lo > 0 else ("behind" if hi < 0 else "no clear difference")
    edge_rows.append({"Juice+ minus": name, "edge": est, "ci_low": lo, "ci_high": hi, "verdict": verdict})
EDGES = pd.DataFrame(edge_rows).set_index("Juice+ minus")
show(EDGES, 3)

print("\nBy position group, same players (rank correlation with Game Juice+):")
grp_rows = {}
for grp, d in V.groupby("pgm"):
    if len(d) >= 15:
        grp_rows[f"{grp} (n={len(d)})"] = {n: safe_spearman(d[c], d.game_juice) for n, c in SCORES[:1] + SCORES[2:5]}
show(pd.DataFrame(grp_rows).T)

# ---------------------------------------------------------------------
hdr("5. IS THIS JUST FITTING THE SAME PLAYERS TWICE?")
# ---------------------------------------------------------------------
print("The Combine formula (size model, reliabilities, agility gate) was tuned on these players.")
print("The held-out version scores each draft class with a formula built WITHOUT that class.")
MF = M.dropna(subset=["juice_heldout"])
cls_parts = []
for y_, d in MF.groupby("draft_year"):
    if len(d) >= 15:
        r_ = safe_spearman(d.juice_heldout, d.game_juice)
        cls_parts.append((r_, len(d)))
        print(f"  class {int(y_)}: corr {r_:+.2f} (n = {len(d)})")
fr, flo, fhi = pooled_fisher(cls_parts)
print(f"  pooled held-out: {fr:+.2f} [{flo:+.2f}, {fhi:+.2f}]   vs in-sample Juice+ on the same group: "
      f"{pooled_fisher([(safe_spearman(d.juice_final, d.game_juice), len(d)) for _, d in MF.groupby('draft_year') if len(d) >= 15])[0]:+.2f}")

# Position-blend weight search. Off by default: blend weights did not hold out of sample (audit, 'Stop doing').
BLEND_TAB = BLEND_EDGES = W_TAB = None
if RUN_BLEND_SEARCH:
    # ---------------------------------------------------------------------
    hdr("6. POSITION-SPECIFIC BLEND (tracked data vs official stopwatch tests)")
    # ---------------------------------------------------------------------
    print("Each position group gets its own mix of the tracked pieces and the official tests (0 = stopwatch only, 1 = tracked only).")
    print("The mix is chosen on two draft classes and scored on the third, so the result is out of sample.")
    print("Only a player's tracked pieces gate who is scored, so every version below scores the same people.")


    def official_parts(z):
        """The official tests that measure each trait, standardized and averaged."""
        out = {}
        for name in "ETA":
            cols = [(col, sg) for col, sg in OFFICIAL[name].items() if col in z]
            out[name] = (pd.concat([sg * z[col] for col, sg in cols], axis=1).mean(axis=1)
                         if cols else pd.Series(np.nan, index=C.index))
        return out


    def assemble(trk, off, w):
        """w is the weight on the tracked piece. A scalar, or a Series by player."""
        out = {}
        for k in "ETA":
            mix = w * trk[k] + (1 - w) * off[k]
            out[k] = mix.where(off[k].notna(), trk[k]).where(trk[k].notna())
        raw = pd.concat(list(out.values()), axis=1).mean(axis=1)
        return raw.where(out["E"].notna() & out["T"].notna())


    def scale_within(raw, keys):
        g = raw.groupby(keys)
        return 100 + 10 * (raw - g.transform("mean")) / g.transform("std").replace(0, np.nan)


    W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
    Z_ALL = {col: adjust(CT, col) for col in TRACKED_COLS + OFFICIAL_COLS}
    TRK_ALL = dict(zip("ETA", combine_pieces(Z_ALL, C, REL_E, REL_T, rbar, gate, blend=False)))
    OFF_ALL = official_parts(Z_ALL)
    RAW_W = {w: assemble(TRK_ALL, OFF_ALL, w) for w in W_GRID}
    GJ = C.nfl_id.map(dict(zip(GAME.nfl_id, GAME.game_juice)))


    def pick_w(train_mask):
        """For each position group, the weight whose Combine score best matches Game Juice+ on the training players."""
        chosen = {}
        for g_ in sorted(C.pgm.unique()):
            m = train_mask & (C.pgm == g_) & GJ.notna()
            scored = []
            for w in W_GRID:
                mm = m & RAW_W[w].notna()
                if mm.sum() >= 20:
                    r_ = safe_spearman(RAW_W[w][mm], GJ[mm])
                    if np.isfinite(r_):
                        scored.append((r_, -abs(w - 0.5), w))   # ties go to the weight nearest 50/50
            chosen[g_] = max(scored)[2] if scored else 0.5
        return chosen


    JB = pd.Series(np.nan, index=C.index)
    w_rows = []
    for cls in sorted(C.draft_year.dropna().unique()):
        train = C.draft_year != cls
        wmap = pick_w(train)
        w_rows.append({"held-out class": int(cls), **wmap})
        ids_tr = set(C.nfl_id[train])
        RE_f, RT_f, _ = retest_reliability(w2[w2.nfl_id.isin(ids_tr)])
        gate_f, rbar_f = gate_fn(ids_tr)
        z = {col: adjust_frozen(CT, col, train) for col in TRACKED_COLS + OFFICIAL_COLS}
        trk = dict(zip("ETA", combine_pieces(z, C, RE_f, RT_f, rbar_f, gate_f, blend=False)))
        raw = assemble(trk, official_parts(z), C.pgm.map(wmap).fillna(0.5))
        te = ~train & raw.notna()
        JB[te] = scale_within(raw[te], C.pgm[te])
    W_TAB = pd.DataFrame(w_rows).set_index("held-out class")
    print("\nWeight on the tracked pieces chosen for each fold (rows = class that was held out):")
    show(W_TAB)
    W_FINAL = pick_w(pd.Series(True, index=C.index))
    print("Weights chosen using all three classes (for scoring players going forward):",
          {k: v for k, v in W_FINAL.items() if (C.pgm == k).sum() >= 15})
    JBF = scale_within(assemble(TRK_ALL, OFF_ALL, C.pgm.map(W_FINAL).fillna(0.5)), C.pgm)

    M["juice_blend"] = M.nfl_id.map(dict(zip(C.nfl_id, JB)))
    M["juice_blend_final"] = M.nfl_id.map(dict(zip(C.nfl_id, JBF)))
    M["X_juice_blend"] = M.groupby("pgm").juice_blend.rank(pct=True)
    VB = M.dropna(subset=[c for _, c in SCORES] + ["juice_blend"]).copy()
    BL_SCORES = [("Juice+ blended (held-out)", "juice_blend"), ("Juice+ main version (held-out)", "juice_heldout"),
                 ("NGS athleticism", "ngs_athleticism_score"), ("RAS-style", "ras_style"), ("Official 40", "forty_neg")]
    yb = VB.X_game_juice.to_numpy()
    rows_b = []
    for name, c in BL_SCORES:
        x = VB["X_" + c].to_numpy()
        b = boot(x, yb)
        rows_b.append({"score": name, "corr_with_game_juice": pr(x, yb), "ci_low": np.percentile(b, 2.5),
                       "ci_high": np.percentile(b, 97.5), "n": len(VB)})
    BLEND_TAB = pd.DataFrame(rows_b).set_index("score")
    print(f"\nOut-of-sample carry-over, same {len(VB)} players:")
    show(BLEND_TAB, 3)

    xb = VB.X_juice_blend.to_numpy()
    idx_b = np.random.default_rng(21).integers(0, len(VB), (BOOT, len(VB)))
    be_rows = []
    for name, c in BL_SCORES[1:]:
        xo = VB["X_" + c].to_numpy()
        d_ = np.array([pr(xb[i], yb[i]) - pr(xo[i], yb[i]) for i in idx_b])
        lo, hi = np.percentile(d_, [2.5, 97.5])
        be_rows.append({"blended minus": name, "edge": pr(xb, yb) - pr(xo, yb), "ci_low": lo, "ci_high": hi,
                        "verdict": "clear edge" if lo > 0 else ("behind" if hi < 0 else "no clear difference")})
    BLEND_EDGES = pd.DataFrame(be_rows).set_index("blended minus")
    print("\nBlended Juice+ edge over each, paired bootstrap:")
    show(BLEND_EDGES, 3)

    print("\nBy position group (same players):")
    gb_rows = {}
    for grp, d in VB.groupby("pgm"):
        if len(d) >= 15:
            gb_rows[f"{grp} (n={len(d)})"] = {n: safe_spearman(d[c], d.game_juice) for n, c in BL_SCORES}
    show(pd.DataFrame(gb_rows).T)
    MB = M.dropna(subset=["juice_blend"])
    print(f"\nBlended Juice+ on everyone scored (n = {len(MB)}): {pr(MB.X_juice_blend, MB.X_game_juice):+.3f}; "
          f"correlation with weight {safe_spearman(MB.juice_blend, MB.combine_weight):+.2f}")
    print("Note: the size model and retest reliabilities for the weight search use all classes; "
          "only the weights themselves are chosen without the held-out class's game scores.")


# ---------------------------------------------------------------------
hdr("6b. VERSION COMPARISON (size link vs carry-over, side by side)")
# ---------------------------------------------------------------------
print(f"Main version for this run: {JUICE_VERSION}. Set JUICE_VERSION=tracked_only or blend_50_50 to change it.")
print("Size link = correlation with body weight inside each position group, pooled.")
print("These rows are in-sample except the position blend. Same players in every row.")
VERS = [("Tracked only", "juice_tracked_only"), ("50/50 with stopwatch tests", "juice_5050"),
        ("Unadjusted for size", "juice_unadjusted"),
        ("Official 40 (for reference)", "forty_neg"), ("NGS athleticism (for reference)", "ngs_athleticism_score")]
if RUN_BLEND_SEARCH:
    VERS.insert(2, ("Position blend (held-out)", "juice_blend"))
for _, c in VERS:
    M["X_" + c] = M.groupby("pgm")[c].rank(pct=True)
VV = M.dropna(subset=[c for _, c in VERS]).copy()
ver_rows = []
for name, c in VERS:
    parts = [(safe_spearman(g[c], g.combine_weight), len(g)) for _, g in VV.groupby("pgm") if len(g) >= 15]
    sr, slo, shi = pooled_fisher(parts)
    x, y = VV["X_" + c].to_numpy(), VV.X_game_juice.to_numpy()
    b = boot(x, y)
    thirds = []
    for _, g in VV.assign(t=VV.groupby("pgm").combine_weight.transform(lambda q: q.rank(method="first") // (len(q) / 3 + 1e-9))).groupby("t"):
        thirds.append(pr(g["X_" + c], g.X_game_juice))
    ver_rows.append({"version": name, "weight_link": sr, "wl_low": slo, "wl_high": shi,
                     "carry_over": pr(x, y), "co_low": np.percentile(b, 2.5), "co_high": np.percentile(b, 97.5),
                     "spread_across_sizes": max(thirds) - min(thirds), "n": len(VV)})
VERSIONS = pd.DataFrame(ver_rows).set_index("version")
show(VERSIONS, 3)

# ---------------------------------------------------------------------
hdr("7. PLAIN-ENGLISH READ")
# ---------------------------------------------------------------------
j = CARRY.loc["Juice+"]
print(f"- Carry-over: a player's Combine Juice+ and Game Juice+ have a rank correlation of {j.corr_with_game_juice:+.2f} "
      f"[{j.ci_low:+.2f}, {j.ci_high:+.2f}] on {int(j.n)} players with every score; "
      f"{BYGROUP.loc['ALL', 'corr']:+.2f} on all {int(BYGROUP.loc['ALL', 'n'])} scored players.")
sz = SIZE.loc["Juice+", "corr_with_weight"]
print(f"- Size: Juice+ vs weight is {sz:+.2f}; NGS {SIZE.loc['NGS athleticism', 'corr_with_weight']:+.2f}, "
      f"RAS-style {SIZE.loc['RAS-style', 'corr_with_weight']:+.2f}, 40 {SIZE.loc['Official 40', 'corr_with_weight']:+.2f}.")
sp = THIRDS["spread (max-min)"]
print(f"- Across light/middle/heavy players, the link moves by {sp['Juice+']:.2f} for Juice+, "
      f"{sp['NGS athleticism']:.2f} for NGS, {sp['RAS-style']:.2f} for RAS-style.")
for name, r_ in EDGES.iterrows():
    print(f"- vs {name}: {r_.verdict} ({r_.edge:+.3f}, 95% [{r_.ci_low:+.3f}, {r_.ci_high:+.3f}]).")
if RUN_BLEND_SEARCH:
    bt = BLEND_TAB.loc["Juice+ blended (held-out)"]
    print(f"- Position-specific blend (out of sample): {bt.corr_with_game_juice:+.2f} [{bt.ci_low:+.2f}, {bt.ci_high:+.2f}] "
          f"vs {BLEND_TAB.loc['Juice+ main version (held-out)', 'corr_with_game_juice']:+.2f} for the main version.")
    for name, r_ in BLEND_EDGES.iterrows():
        print(f"- blended vs {name}: {r_.verdict} ({r_.edge:+.3f}, 95% [{r_.ci_low:+.3f}, {r_.ci_high:+.3f}]).")
print("- These head-to-head edges are not the headline. juice_transfer.py reports what each tracked piece adds"
      " beyond the stopwatch, by position (audit item 3).")
print("- Caveat: Game Juice+ is size-adjusted by construction, so scores that are not size-adjusted start at a disadvantage."
      " The size-adjusted rows above show how much of any edge is just that correction.")

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
save(M.drop(columns=["plays_bucket", "band"]), "juice_player_table.csv")
save(ACCOUNT, "roster_accounting.csv")
save(CARRY.reset_index(), "carry_over.csv")
save(EDGES.reset_index(), "edges_vs_other_scores.csv")
save(THIRDS.reset_index(), "by_size_third.csv")
save(SIZE.reset_index(), "size_dependence.csv")
save(BYGROUP.reset_index(), "by_position_group.csv")
save(VERSIONS.reset_index(), "version_comparison.csv")
if RUN_BLEND_SEARCH:
    save(W_TAB.reset_index(), "blend_weights.csv")
    save(BLEND_TAB.reset_index(), "blend_carry_over.csv")
    save(BLEND_EDGES.reset_index(), "blend_edges.csv")
else:
    # Old blend tables came from earlier runs. Remove them so no stale number is read as current.
    for _stale in ("blend_weights.csv", "blend_carry_over.csv", "blend_edges.csv"):
        (OUT / _stale).unlink(missing_ok=True)

# Combine side in full (tracked features, pieces, official tests) for juice_physics.py and juice_transfer.py.
save(with_tests(FINAL).drop(columns=[c_ for c_ in ["raw"] if c_ in FINAL]), "combine_juice_scores.csv")
save(BURST_VALIDITY, "burst_validity.csv")
save(G_ALL[["nfl_id", "npl", "nb", "n_alat"]], "game_play_counts.csv")

# Run manifest: settings and headline counts for this run.
MANIFEST = {"run_id": RUN_ID, "git": _git, "code_sha": CODE_SHA, "utc": _stamp, "settings": SETTINGS,
            "counts": {"combine_tracked": int(n_tracked), "combine_scored": int(len(FINAL)),
                       "game_any_usable": int(len(G_ALL)), "game_scored": int(len(GAME)), "both": int(len(M)),
                       "head_to_head": int(len(V))},
            "reliability": {"combine_retest": RELI, "game_split_half": GREL},
            "game_weights": C_WEIGHTS}
(OUT / "run_manifest.json").write_text(json.dumps(MANIFEST, indent=2, default=float))
print(f"\nAll tables saved to: {OUT.resolve()}  (run {RUN_ID})")