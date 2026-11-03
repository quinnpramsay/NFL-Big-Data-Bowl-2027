"""
Juice+: does Combine movement predict NFL game movement?
========================================================

Combine Juice+ (from Combine tracking) vs Game Juice+ (from 2023-2025 regular-season
tracking). Only players with Combine tracking are used, so game data covers exactly the
positions the Combine data covers. Position groups: DB, DL, EDGE, OL, TE, WR.

Combine Juice+
  E explosiveness (distance at 1.5 s, speed at 1.0 s), T top-end speed (fastest 10 yards),
  A agility (position drills; only where the drills are reliable, not EDGE or TE).
  Each tracked piece is adjusted for size and draft class and shrunk by its retest reliability.
  It is then averaged with the official tests of the same trait (E: vertical, broad jump,
  10-yard split; T: 40; A: 3-cone, shuttle), adjusted the same way. The pieces are averaged
  and scaled to 100 +/- 10 within position group. BLEND_STOPWATCH = False gives tracked-only.

Game Juice+
  top speed, explosiveness and agility from game snaps, each as the player's best effort
  (95th percentile) after adjusting for position, alignment, route (coverage job for DBs)
  and play length. Plays in decided games (home win probability under 5% or over 95%) are
  dropped. Explosiveness averages speed at 1.0 s and distance at 1.5 s after a standing-start
  snap on role plays.
  Components are weighted by their split-half reliability, and each is shrunk toward
  average by the player's own number of plays, so players with few snaps count for less.
  Game agility is left out where the Combine agility check fails, so both sides match.

Checks
  Correlation by position, draft class, rookie season only; season-to-season persistence;
  outcome bands with risk ratios and AUC; Combine piece vs game piece; head-to-head vs NGS
  and RAS-style scores; tracked top speed vs the official 40; out-of-class R-squared and a
  fully held-out version of the formula; late-round flags and Combine Juice+ by round.

Run:
    pip install kagglehub python-dotenv pandas numpy scipy scikit-learn
    python3 juice.py

Credentials: KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json).
Optional environment variables:
    JUICE_DATA_DIR   folder holding the competition CSVs (skips Kaggle)
    JUICE_OUT_DIR    output folder (default juice_full_output)
    JUICE_CACHE_DIR  cache folder (default juice_cache; delete it to rebuild)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COMP = "nfl-big-data-bowl-2027"
DATA_DIR = os.getenv("JUICE_DATA_DIR")
OUT = Path(os.getenv("JUICE_OUT_DIR", "juice_full_output"))
CACHE = Path(os.getenv("JUICE_CACHE_DIR", "juice_cache"))
for _p in (OUT, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

if not DATA_DIR and not (
    os.getenv("KAGGLE_API_TOKEN")
    or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    or (Path.home() / ".kaggle" / "kaggle.json").exists()
):
    raise RuntimeError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN in .env, or put "
        "kaggle.json in ~/.kaggle/, or set JUICE_DATA_DIR to a local folder."
    )

SEASONS = (2023, 2024, 2025)
CHUNK_SIZE = int(os.getenv("JUICE_CHUNK_SIZE", 2_000_000))
MIN_PLAY_SECONDS = 3.0
MIN_FRAMES = int(round(MIN_PLAY_SECONDS * 10))   # drop dead plays
MIN_FRAMES_OPEN = 20                             # plays used for speed / agility

POOLS = {"all": (1, 0)}            # every tracked player with game data: (min plays, min standing snaps)
SEASON_POOLS = {"all": (1, 0)}
RELIABILITY_MIN = (200, 50)        # reliabilities are estimated on well-sampled players only
SEASON_MIN_PLAYS, SEASON_MIN_BURSTS = 100, 30   # persistence check, one season at a time

GATE_ALPHA, MIN_PLAYERS_DRILL, MIN_OVERLAP = 0.5, 20, 15
BLEND_STOPWATCH = True             # average tracked pieces with official tests of the same trait
GARBAGE_WP = 0.05                  # drop plays when home win probability is under 5% or over 95%
BOOT_REPS = int(os.getenv("JUICE_BOOT_REPS", 2000))
BELOW_AVG, ELITE = 100, 110
HIGH_PICK, LOW_PICK = 64, 140
HIGH_JUICE, LOW_JUICE = 110, 90

PGM = {
    "T": "OL", "G": "OL", "C": "OL", "DT": "DL", "NT": "DL",
    "DE": "EDGE", "OLB": "EDGE", "CB": "DB", "DB": "DB", "SS": "DB",
    "FS": "DB", "WR": "WR", "TE": "TE",
}
ALIAS = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
}
KEYS = ["game_id", "play_id", "nfl_id"]
rng = np.random.default_rng(42)



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_root = None


def get_file(name):
    """Path to a competition file, from JUICE_DATA_DIR or a Kaggle download."""
    global _root
    if DATA_DIR:
        return os.path.join(DATA_DIR, name)
    if _root is None:
        import kagglehub
        _root = kagglehub.competition_download(COMP)
    direct = os.path.join(_root, name)
    nested = os.path.join(_root, COMP, name)
    return direct if os.path.exists(direct) else nested


def load_csv(name, **kwargs):
    path = get_file(name)
    print(f"Using {name}: {path}")
    return pd.read_csv(path, **kwargs)


def save(df, name):
    df.to_csv(OUT / name, index=False)


def circ(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def zs(series):
    sd = series.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / sd


def safe_spearman(x, y):
    pair = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if len(pair) < 4 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan
    return float(spearmanr(pair.x, pair.y)[0])


def fisher_ci(r, n):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def sb(k, r):
    """Spearman-Brown: reliability of the mean of k observations."""
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
    sd = g.transform("std").replace(0, np.nan)
    return 100 + 10 * (df[col] - g.transform("mean")) / sd


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p_ = k / n
    den = 1 + z ** 2 / n
    ctr = (p_ + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


def adjust(df, col, use_class=True, size_adj=True):
    """Leave-one-out residual of `col` within each position group after removing
    body size and draft class, standardized to mean 0, sd 1."""
    out = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("pgm"):
        d = d.dropna(subset=[col, "combine_weight", "combine_height"])
        if len(d) < 15:
            continue
        y = d[col].to_numpy()
        feats = (
            pd.get_dummies(d.draft_year, prefix="yr", drop_first=True).astype(float)
            if use_class else pd.DataFrame(index=d.index)
        )
        if size_adj and len(d) >= 25:
            feats["w"] = d.combine_weight.to_numpy()
            feats["h"] = d.combine_height.to_numpy()
            feats["w2"] = ((d.combine_weight - d.combine_weight.mean()) ** 2 / 100).to_numpy()
        if feats.shape[1] == 0:
            pred = np.array([np.delete(y, i).mean() for i in range(len(y))])
        else:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
            pred = cross_val_predict(model, feats.to_numpy(), y, cv=LeaveOneOut())
        resid = y - pred
        if resid.std() > 0:
            out.loc[d.index] = (resid - resid.mean()) / resid.std()
    return out


def auc_ci(y, s, reps=1000, seed=3):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 10 or y.min() == y.max():
        return np.nan, np.nan, np.nan
    r_, boots = np.random.default_rng(seed), []
    for _ in range(reps):
        i = r_.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def pooled_fisher(rs_ns):
    """Pool correlations across groups by Fisher z weighted by n-3. Returns r, lo, hi."""
    rs_ns = [(r, n) for r, n in rs_ns if np.isfinite(r) and abs(r) < 1 and n > 3]
    if not rs_ns:
        return np.nan, np.nan, np.nan
    wsum = sum(n - 3 for _, n in rs_ns)
    r = float(np.tanh(sum(np.arctanh(r) * (n - 3) for r, n in rs_ns) / wsum))
    se = 1.96 / np.sqrt(wsum)
    return r, float(np.tanh(np.arctanh(r) - se)), float(np.tanh(np.arctanh(r) + se))


# ---------------------------------------------------------------------
# [1/6] Players, Combine results, games
# ---------------------------------------------------------------------

print("\n[1/6] Loading player, Combine and game metadata...")
players = load_csv("players.csv")
combine = load_csv("combine_results.csv")
games = load_csv("games.csv")

LOWER_BETTER = [c for c in ["forty", "ten_yd_split", "three_cone", "short_shuttle"] if c in combine.columns]
HIGHER_BETTER = [c for c in ["vertical", "broad_jump", "bench_reps"] if c in combine.columns]
TESTS = combine[["nfl_id"] + LOWER_BETTER + HIGHER_BETTER].copy()

pcols = [c for c in ["nfl_id", "display_name", "nfl_position", "draft_year",
                     "draft_round", "draft_overall_pick"] if c in players.columns]
D = players[pcols].copy()
D["pgm"] = D.nfl_position.map(PGM)
D = D.dropna(subset=["pgm"])
ccols = [c for c in ["nfl_id", "combine_weight", "combine_height",
                     "ngs_athleticism_score"] if c in combine.columns]
D = D.merge(combine[ccols], on="nfl_id", how="left")
print(f"Players in the six position groups: {len(D)}")

# ---------------------------------------------------------------------
# [2/6] Combine Juice+
# ---------------------------------------------------------------------

print("\n[2/6] Processing Combine tracking...")
ct = load_csv(
    "combine_tracking.csv",
    usecols=["nfl_id", "entity_type", "event_id", "drill_name",
             "drill_type", "time", "x", "y", "s", "a"],
)
ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.notna()].copy()
ct["nfl_id"] = ct.nfl_id.astype(int)
ct = ct[ct.nfl_id.isin(D.nfl_id)]
# Game data is limited to players, and so positions, that have Combine tracking.
D = D[D.nfl_id.isin(ct.nfl_id)].copy()
print(f"Players with Combine tracking: {len(D)}; by position: {D.nfl_position.value_counts().to_dict()}")
ct["drill_name"] = ct.drill_name.replace(ALIAS)
ct["time"] = pd.to_datetime(ct.time)
ct = ct.sort_values(["event_id", "time"])
ct["t"] = (ct.time - ct.groupby("event_id").time.transform("min")).dt.total_seconds()


def feats40(a):
    s, x, t = a.s.to_numpy(), a.x.to_numpy(), a.t.to_numpy()
    moving = np.where(s >= 1.0)[0]
    if len(moving) < 10:
        return None
    i0 = moving[0]
    s, x, t = s[i0:], x[i0:], t[i0:]
    progress = np.maximum.accumulate(np.abs(x - x[0])) + np.arange(len(x)) * 1e-9
    if progress[-1] < 36:
        return None
    elapsed = t - t[0]
    best = 0.0
    for start in np.arange(5.0, min(progress[-1] - 10, 30.0) + 1e-6, 2.5):
        mask = (progress >= start) & (progress <= start + 10)
        if mask.sum() >= 5:
            best = max(best, float(s[mask].mean()))
    smooth = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return pd.Series({
        "d15": np.interp(1.5, elapsed, progress),
        "v1": np.interp(1.0, elapsed, smooth),
        "fly10": best if best > 0 else np.nan,
    })


f40 = ct[ct.drill_name == "FORTY_YARD_DASH"]
a40 = f40.groupby("event_id").apply(feats40).dropna(subset=["fly10"])
a40["nfl_id"] = f40.groupby("event_id").nfl_id.first().reindex(a40.index).astype(int)
a40 = a40.sort_index()
a40["k"] = a40.groupby("nfl_id").cumcount()

w2 = a40[a40.k < 2].pivot(index="nfl_id", columns="k", values=["d15", "v1", "fly10"]).dropna()
w2.columns = [f"{f}_{k}" for f, k in w2.columns]
w2 = w2.reset_index().merge(D[["nfl_id", "pgm", "draft_year"]], on="nfl_id")


def retest_reliability(w):
    """Attempt-to-attempt Spearman WITHIN position after removing class means,
    pooled by Fisher z. Returns (explosiveness reliability, top-end reliability, dict)."""
    out = {}
    for feat in ["d15", "v1", "fly10"]:
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
    return (out["d15"] + out["v1"]) / 2, out["fly10"], out


REL_E, REL_T, RELI = retest_reliability(w2)
print(f"Combine retest reliability (within position): d15={RELI['d15']:.2f}, "
      f"v1={RELI['v1']:.2f}, fly10={RELI['fly10']:.2f}")

P40 = a40.groupby("nfl_id").agg(
    d15=("d15", "mean"), v1=("v1", "mean"), fly10=("fly10", "mean"),
    attempts=("fly10", "size"),
).reset_index()


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
TRACKED_COLS = ["d15", "v1", "fly10", "agil"]
OFFICIAL_COLS = sorted({c_ for d_ in OFFICIAL.values() for c_ in d_} & set(TESTS.columns))


def with_tests(df):
    """Same rows as df, plus the official Combine tests."""
    return df.join(TESTS.set_index("nfl_id"), on="nfl_id")


def combine_pieces(z, c, rel_e, rel_t, rbar_map, gate_map, shrink=True, blend=True):
    """E, T, A from standardized columns z. Tracked pieces are shrunk by their reliability;
    with blend=True each is then averaged with the official tests of the same trait."""
    E = pd.concat([z["d15"], z["v1"]], axis=1).mean(axis=1)
    T = z["fly10"]
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


S = build_scores(C, blend=BLEND_STOPWATCH)
S["juice_unadjusted"] = build_scores(C, size_adj=False, blend=BLEND_STOPWATCH)["juice"].values
# Tracked-only version, saved so outcome.py can show what the blend adds.
TRK = build_scores(C, blend=False)
S["juice_tracked_only"] = TRK["juice"].values
for comp in ["E", "T", "A"]:
    S[f"{comp}_trk_plus"] = TRK[f"{comp}_plus"].values
FINAL = S[S.E_plus.notna() & S.T_plus.notna()].copy()
FINAL["juice_final"] = FINAL["juice"]
print(f"Players with a Combine Juice+ score: {len(FINAL)}")
save(FINAL, "combine_juice_scores.csv")

# ---------------------------------------------------------------------
# [3/6] Game tracking pass (cached)
# ---------------------------------------------------------------------

print("\n[3/6] Processing NFL game tracking (cached after the first run)...")
REG = set(games.loc[games.season_type == "REG", "game_id"])


def summarize_chunk(df):
    cols = KEYS + ["n_after", "peak", "alat_max", "s0", "s10", "d15"]
    if df is None or len(df) == 0:
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
    out = df.groupby(KEYS).agg(
        n_after=("k", "size"), peak=("s", "max"), alat_max=("alat", "max"))
    out["s0"] = df.loc[df.k == 0].set_index(KEYS).s
    out["s10"] = df.loc[df.k == 10].set_index(KEYS).s
    # Distance covered in the first 1.5 s after the snap (same idea as the Combine d15).
    p0 = df.loc[df.k == 0].set_index(KEYS)[["x", "y"]]
    p15 = df.loc[df.k == 15].set_index(KEYS)[["x", "y"]]
    out["d15"] = np.hypot(p15.x - p0.x.reindex(p15.index), p15.y - p0.y.reindex(p15.index))
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


cache_file = CACHE / "game_play_table_v2.csv"   # v2 adds d15
if cache_file.exists():
    print(f"  Using cached play table: {cache_file}")
    PT = pd.read_csv(cache_file)
else:
    PT = game_pass()
    PT.to_csv(cache_file, index=False)
print(f"  Player-plays in table: {len(PT)}")

# ---------------------------------------------------------------------
# [4/6] Roles, context adjustment, Game Juice+ components
# ---------------------------------------------------------------------

print("\n[4/6] Building role fields and context-adjusted game metrics...")
pp = load_csv(
    "player_play.csv",
    usecols=KEYS + ["lined_up_position", "route_ran", "player_get_off", "coverage_assignment",
                    "pass_rushers_encountered", "pre_snap_home_team_win_probability"],
)
PT = (
    PT.merge(games[["game_id", "season"]], on="game_id")
      .merge(D[["nfl_id", "pgm", "nfl_position", "draft_year"]], on="nfl_id")
      .merge(pp, on=KEYS, how="left")
)
# Drop plays from decided games, where effort and snap counts change.
wp = PT.pre_snap_home_team_win_probability
PT = PT[~((wp <= GARBAGE_WP) | (wp >= 1 - GARBAGE_WP))].copy()
PT["role"] = (
    (PT.pgm.isin(["WR", "TE"]) & PT.route_ran.notna())
    | (PT.pgm.isin(["DL", "EDGE"]) & PT.player_get_off.notna())
    | ((PT.pgm == "DB") & PT.coverage_assignment.notna() & (PT.coverage_assignment != "NO_ASSIGNMENT"))
    | ((PT.pgm == "OL") & PT.pass_rushers_encountered.notna())
)
PT["align"] = np.where(PT.pgm.isin(["DL", "EDGE", "DB", "WR"]), PT.lined_up_position.fillna("UNK"), "ALL")
PT = PT[PT.n_after >= MIN_FRAMES].copy()
PT["len_s"] = np.minimum(PT.n_after // 10, 10)


def coverage_family(a):
    """Collapse the 19 coverage assignments into the jobs a DB actually has."""
    a = str(a)
    if a == "MAN":
        return "MAN"
    if a.startswith("DEEP_THIRD") or a == "DEEP_FREE":
        return "DEEP_THIRD"
    if a.startswith("QUARTERS"):
        return "QUARTERS"
    if a.startswith("DEEP_HALF"):
        return "DEEP_HALF"
    if a.startswith(("CURL_FLAT", "FLAT", "HOOK_CURL")) or a == "HOLE":
        return "SHORT_ZONE"
    return "OTHER"


# One "route" key for everyone: the route for WR/TE, the coverage job for DBs, nothing for the rest.
PT["route"] = np.where(PT.pgm.isin(["WR", "TE"]), PT.route_ran.fillna("NO_ROUTE"),
                       np.where(PT.pgm == "DB", PT.coverage_assignment.map(coverage_family), "NONE"))


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
    return P.role & (P.s0 < 1.0) & P.s10.notna() & P.d15.notna()


def add_adjusted(frame):
    P = frame.copy()
    factors = ["route", "len_s", "align"]
    P["peak_c"], P["alat_c"], P["s10_c"], P["d15_c"], P["expl_c"] = (
        np.nan, np.nan, np.nan, np.nan, np.nan)
    open_play = P.n_after >= MIN_FRAMES_OPEN
    if open_play.any():
        P.loc[open_play, "peak_c"] = backfit(P[open_play], "peak", factors)
        P.loc[open_play, "alat_c"] = backfit(P[open_play], "alat_max", factors)
    ok = burst_mask(P)
    if ok.any():
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


def game_components(P, min_plays, min_bursts, strict_join=False):
    """One row per player: top speed, agility, explosiveness and counts (P already adjusted)."""
    burst = P[burst_mask(P)]
    gb = burst.groupby("nfl_id").agg(nb=("expl_c", "size"), g_expl=("expl_c", q95))
    open_ = P[P.n_after >= MIN_FRAMES_OPEN]
    ga = open_.groupby("nfl_id").agg(
        npl=("peak_c", "size"), n_alat=("alat_c", "count"),
        g_top=("peak_c", q95), g_agil=("alat_c", q95),
    )
    G = ga.join(gb, how="inner" if strict_join else "left")
    G["nb"] = G.nb.fillna(0)
    G = G[(G.nb >= min_bursts) & (G.npl >= min_plays)].reset_index()
    G = G.merge(
        D[["nfl_id", "display_name", "nfl_position", "pgm", "draft_year",
           "combine_weight", "combine_height"]], on="nfl_id",
    )
    for c in ["g_top", "g_agil"]:
        G[c] = G[c] - G.groupby("nfl_position")[c].transform("mean")
    return G


PTa = add_adjusted(PT)
G_BY_POOL = {pool: game_components(PTa, mp, mb, strict_join=(mb > 0))
             for pool, (mp, mb) in POOLS.items()}
for pool, G in G_BY_POOL.items():
    print(f"  Pool '{pool}': {len(G)} players with game data")

# ---------------------------------------------------------------------
# [5/6] Reliabilities and Game Juice+
# ---------------------------------------------------------------------

print("\n[5/6] Estimating game reliabilities and building Game Juice+...")


def split_half(P, ids, col, stat, reps=30, burst=False):
    sub = P[P.nfl_id.isin(ids)]
    if burst:
        sub = sub[burst_mask(sub)]
    else:
        sub = sub[sub.n_after >= MIN_FRAMES_OPEN]
    sub = sub[["nfl_id", col]].dropna().reset_index(drop=True)
    out = []
    for _ in range(reps):
        sub["h"] = rng.integers(0, 2, len(sub))
        g = sub.groupby(["nfl_id", "h"])[col].agg(stat).unstack("h").dropna()
        if len(g) < 10:
            continue
        r = safe_spearman(g[0], g[1])
        if np.isfinite(r) and r > -0.99:
            out.append(2 * r / (1 + r))
    return float(np.nanmean(out)) if out else np.nan


G_REL = game_components(PTa, *RELIABILITY_MIN, strict_join=True)
if len(G_REL) < 60:
    print("  Few well-sampled players; estimating reliabilities on all players instead.")
    G_REL = G_BY_POOL["all"]
rel_ids = G_REL.nfl_id
GREL = {
    "top": split_half(PTa, rel_ids, "peak_c", q95),
    "expl": split_half(PTa, rel_ids, "expl_c", q95, burst=True),
    "agil": split_half(PTa, rel_ids, "alat_c", q95),
}
K_TYP = {
    "top": float(G_REL.npl.median()),
    "expl": float(G_REL.nb.replace(0, np.nan).median()),
    "agil": float(G_REL.n_alat.median()),
}
GREL1 = {k: single_obs_reliability(GREL[k], K_TYP[k]) for k in GREL}
GATE_G = bool(np.isfinite(GREL["agil"]) and GREL["agil"] >= 0.5)
print(f"  Split-half reliability (full sample): {GREL}")
print(f"  Typical observations per player:      {K_TYP}")
print(f"  Single-play reliability:              {GREL1}")
print(f"  Game agility included in Game Juice+: {GATE_G}")

COMP_COL = {"top_z": ("g_top", "npl", "top"),
            "expl_z": ("g_expl", "nb", "expl"),
            "agil_z": ("g_agil", "n_alat", "agil")}
USE = ["top_z", "expl_z"] + (["agil_z"] if GATE_G else [])
_w = {z: max(float(np.nan_to_num(GREL[COMP_COL[z][2]])), 0.0) for z in USE}
_tot = sum(_w.values())
C_WEIGHTS = {z: (w / _tot if _tot > 0 else 1 / len(USE)) for z, w in _w.items()}
print(f"  Game Juice+ component weights: {C_WEIGHTS}")


def c_score(G, how="adjust", use_class=True):
    """Game Juice+ from a component frame. how='adjust' removes size and
    class (the main score); how='zscore' only standardizes within position."""
    g = G.copy()
    for z, (col, _, _) in COMP_COL.items():
        if how == "adjust":
            g[z] = adjust(g, col, use_class=use_class)
        else:
            g[z] = g.groupby("pgm")[col].transform(zs)
    # Combine agility is only scored where its drills are reliable (not EDGE or TE), so leave
    # game agility out for those groups too. Weights renormalize per player.
    g.loc[~g.pgm.map(gate).fillna(False).astype(bool), "agil_z"] = np.nan
    num, den = 0, 0
    for z in USE:
        _, ncol, key = COMP_COL[z]
        shrunk = g[z] * sb(g[ncol].fillna(0), GREL1[key])
        num = num + shrunk.fillna(0) * C_WEIGHTS[z]
        den = den + shrunk.notna() * C_WEIGHTS[z]
    g["raw"] = num / den.replace(0, np.nan)
    g["game_juice"] = scale100(g, "raw")
    for z, name in [("top_z", "g_top_plus"), ("expl_z", "g_expl_plus"), ("agil_z", "g_agil_plus")]:
        g[name] = 100 + 10 * g[z]
    return g


GAME = {pool: c_score(G) for pool, G in G_BY_POOL.items()}

# Season-to-season persistence of Game Juice+ (the ceiling for any Combine link).
persist_rows, season_scores = [], []
for seas, sub in PT.groupby("season"):
    Gs = game_components(add_adjusted(sub), SEASON_MIN_PLAYS, SEASON_MIN_BURSTS, strict_join=True)
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

# ---------------------------------------------------------------------
# [6/6] Validation (everything below uses Game Juice+)
# ---------------------------------------------------------------------

print("\n[6/6] Comparing Combine Juice+ to Game Juice+ ...")

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


# --- fully held-out Combine formula (each class scored by a formula fit without it) ---
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
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(feats(tr), tr[col])
            p = mdl.predict(feats(te))
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
print(f"  Held-out Combine formula scored {len(JF_MAP)} players")


# --- rookie-season-only game score (one score per player from his draft-year season) ---
def rookie_corr(pool):
    mp, mb = SEASON_POOLS[pool]
    rk = PT[PT.draft_year.notna() & (PT.season == PT.draft_year)]
    if len(rk) == 0:
        return np.nan, 0
    G_rk = game_components(add_adjusted(rk), mp, mb, strict_join=(mb > 0))
    if len(G_rk) < 45:
        return np.nan, len(G_rk)
    Mr = FINAL[["nfl_id", "juice_final"]].merge(c_score(G_rk)[["nfl_id", "draft_year", "game_juice"]],
                                                on="nfl_id")
    parts = []
    for _, d in Mr.groupby("draft_year"):
        d = d.dropna()
        if len(d) >= 15:
            parts.append((safe_spearman(d.juice_final, d.game_juice), len(d)))
    return pooled_fisher(parts)[0], len(Mr)


# --- pool-level analysis ------------------------------------------------------------
def merged_for(pool):
    g = GAME[pool][["nfl_id", "display_name", "nfl_position", "pgm", "draft_year",
                    "game_juice", "g_top_plus", "g_expl_plus", "g_agil_plus", "npl", "nb"]]
    M = FINAL[["nfl_id", "juice_final", "E_plus", "T_plus", "A_plus"]].merge(g, on="nfl_id")
    M["g_below"] = M.game_juice < BELOW_AVG
    M["g_above"] = M.game_juice >= BELOW_AVG
    M["g_elite"] = M.game_juice >= ELITE
    M["c_band"] = pd.cut(M.juice_final, BAND_EDGES, labels=BAND_LABELS, right=False)
    M["c_p"] = M.groupby("pgm").juice_final.rank(pct=True)
    return M


def risk_ratio(g, o, reps=2000, seed=7):
    g, o = np.asarray(g, bool), np.asarray(o, bool)
    if g.sum() == 0 or o.mean() == 0:
        return np.nan, (np.nan, np.nan)
    r_ = np.random.default_rng(seed)
    est, boots = o[g].mean() / o.mean(), []
    for _ in range(reps):
        i = r_.integers(0, len(o), len(o))
        gi, oi = g[i], o[i]
        if gi.sum() and oi.mean() > 0:
            boots.append(oi[gi].mean() / oi.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(est), (float(lo), float(hi))


def risk_answer(label, group_mask, outcome):
    g, o = np.asarray(group_mask, bool), np.asarray(outcome, bool)
    n, k = int(g.sum()), int(o[g].sum())
    rr, (rlo, rhi) = risk_ratio(g, o)
    wlo, whi = wilson(k, n)
    return {"claim": label, "k": k, "n": n, "pct": 100 * k / n if n else np.nan,
            "wilson_lo": 100 * wlo, "wilson_hi": 100 * whi,
            "base_pct": 100 * o.mean(), "risk_ratio": rr, "rr_lo": rlo, "rr_hi": rhi}


def head_to_head(M):
    """Juice+ vs NGS and RAS-style scores on the same players, raw and size-adjusted."""
    if "ngs_athleticism_score" not in CMP.columns or CMP.ras_style.notna().sum() < 30:
        return None
    X = CMP[CMP.ngs_athleticism_score.notna() & CMP.ras_style.notna()].copy()
    weight_r = {g: {s_: spearmanr(d.combine_weight, d[c_], nan_policy="omit")[0] for s_, c_ in
                    [("Juice+", "juice_final"), ("NGS", "ngs_athleticism_score"), ("RAS", "ras_style")]}
                for g, d in X.groupby("pgm") if len(d) >= 15}
    V = (M[["nfl_id", "pgm", "juice_final", "game_juice", "g_top_plus", "T_plus"]]
         .merge(CMP[["nfl_id", "ngs_athleticism_score", "ras_style", "ngs_adj", "ras_adj",
                     "forty_adj", "split_adj", "forty_neg", "juice_unadjusted"]], on="nfl_id", how="left"))
    V = V.dropna(subset=["ngs_athleticism_score", "ras_style", "ngs_adj", "ras_adj", "juice_unadjusted"])
    if len(V) < 30:
        return None

    def pooled_rho(df, col, outcome="game_juice"):
        return spearmanr(df.groupby("pgm")[col].rank(pct=True),
                         df.groupby("pgm")[outcome].rank(pct=True), nan_policy="omit")[0]

    def corr_row(df, col, outcome):
        row = {"ALL": pooled_rho(df, col, outcome)}
        for g, d in df.groupby("pgm"):
            if len(d) >= 15:
                row[g] = spearmanr(d[col], d[outcome], nan_policy="omit")[0]
        return row

    sc = {"Juice+": "juice_final", "Juice+ (no size adj.)": "juice_unadjusted",
          "NGS": "ngs_athleticism_score", "NGS (size adj.)": "ngs_adj",
          "RAS-style": "ras_style", "RAS-style (size adj.)": "ras_adj"}
    ngs_tab = pd.DataFrame({k: corr_row(V, c_, "game_juice") for k, c_ in sc.items()}).T
    U = V.dropna(subset=["T_plus", "forty_adj", "split_adj", "forty_neg"])
    top_tab = (pd.DataFrame({k: corr_row(U, c_, "g_top_plus") for k, c_ in
                             {"Tracked fastest 10": "T_plus", "Official 40 (size adj.)": "forty_adj",
                              "10-yd split (size adj.)": "split_adj", "Official 40 (raw)": "forty_neg"}.items()}).T
               if len(U) >= 15 else pd.DataFrame())
    rng_b, boots = np.random.default_rng(1), {"NGS": [], "RAS": []}
    for _ in range(1000):
        b = V.iloc[rng_b.integers(0, len(V), len(V))]
        j_ = pooled_rho(b, "juice_final")
        boots["NGS"].append(j_ - pooled_rho(b, "ngs_athleticism_score"))
        boots["RAS"].append(j_ - pooled_rho(b, "ras_style"))
    edge = {k: (ngs_tab.loc["Juice+", "ALL"] - ngs_tab.loc[n_, "ALL"], *np.percentile(v, [2.5, 97.5]))
            for k, v, n_ in [("NGS", boots["NGS"], "NGS"), ("RAS", boots["RAS"], "RAS-style")]}
    # When the scores disagree
    for col, nm in [("juice_final", "j_p"), ("ngs_athleticism_score", "n_p"), ("ras_style", "r_p")]:
        X[nm] = 100 * X.groupby("pgm")[col].rank(pct=True)
    X["gap"] = X.j_p - (X.n_p + X.r_p) / 2
    Y = X.merge(M[["nfl_id", "game_juice"]], on="nfl_id")
    Y["g_p"] = 100 * Y.groupby("pgm").game_juice.rank(pct=True)
    disagree = {k: (int(m.sum()), float(Y.g_p[m].mean()) if m.sum() else np.nan) for k, m in
                [("Juice+ higher by 20+ pct pts", Y.gap >= 20), ("agree (within 20)", Y.gap.abs() < 20),
                 ("Juice+ lower by 20+ pct pts", Y.gap <= -20)]}
    return {"n": len(V), "ngs_tab": ngs_tab, "top_tab": top_tab, "edge": edge,
            "weight_r": weight_r, "disagree": disagree, "V": V}


def incremental_and_frozen(M):
    """Out-of-class R-squared (train on two classes, predict the third), with the in-sample
    Combine Juice+ and with the fully held-out formula."""
    keep = ["nfl_id", "draft_overall_pick", "combine_weight", "combine_height"] + [
        c_ for c_ in ["ngs_athleticism_score"] + LOWER_BETTER + HIGHER_BETTER if c_ in CMP.columns]
    IV = M[["nfl_id", "pgm", "draft_year", "juice_final", "game_juice", "npl"]].merge(
        CMP[keep], on="nfl_id", how="left")

    def pct_feature(col, lower_better=False):
        return IV.groupby("pgm")[col].rank(pct=True, ascending=not lower_better).fillna(0.5)

    trad = {c_: pct_feature(c_, c_ in LOWER_BETTER) for c_ in LOWER_BETTER + HIGHER_BETTER if c_ in IV}
    if "ngs_athleticism_score" in IV:
        trad["ngs"] = pct_feature("ngs_athleticism_score")
    trad["weight"] = IV.groupby("pgm").combine_weight.transform(zs).fillna(0)
    trad["height"] = IV.groupby("pgm").combine_height.transform(zs).fillna(0)
    X_trad = pd.DataFrame(trad)
    X_pick = pd.DataFrame({"log_pick": np.log(IV.draft_overall_pick.fillna(260))})
    X_juice = pd.DataFrame({"juice": (IV.juice_final - 100) / 10})
    X_jf = pd.DataFrame({"juice": (IV.nfl_id.map(JF_MAP).fillna(100) - 100) / 10})
    models = {"pick": X_pick, "trad": X_trad, "trad+pick": X_trad.join(X_pick),
              "juice": X_juice, "trad+pick+juice": X_trad.join(X_pick).join(X_juice),
              "juice_frozen": X_jf, "trad+pick+juice_frozen": X_trad.join(X_pick).join(X_jf)}
    yv, cls_ = IV.game_juice.values, IV.draft_year.values

    def out_of_class(Xm):
        pred, base = np.full(len(IV), np.nan), np.full(len(IV), np.nan)
        for c_ in np.unique(cls_):
            tr, te = cls_ != c_, cls_ == c_
            if tr.sum() < 20 or te.sum() == 0:
                continue
            m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(Xm.values[tr], yv[tr])
            pred[te], base[te] = m_.predict(Xm.values[te]), yv[tr].mean()
        return pred, base

    def r2(y_, p, b):
        ok = ~np.isnan(p)
        den = ((y_[ok] - b[ok]) ** 2).sum()
        return 1 - ((y_[ok] - p[ok]) ** 2).sum() / den if den > 0 else np.nan

    oof = {k: out_of_class(Xm) for k, Xm in models.items()}
    R2 = {k: r2(yv, *v) for k, v in oof.items()}

    def gain(full, seed):
        r_, dg = np.random.default_rng(seed), []
        for _ in range(2000):
            i = r_.integers(0, len(yv), len(yv))
            dg.append(r2(yv[i], oof[full][0][i], oof[full][1][i])
                      - r2(yv[i], oof["trad+pick"][0][i], oof["trad+pick"][1][i]))
        return (R2[full] - R2["trad+pick"], *np.nanpercentile(dg, [2.5, 97.5]))

    MF = M[["nfl_id", "pgm", "draft_year", "game_juice", "g_above"]].assign(
        juice_frozen=M.nfl_id.map(JF_MAP)).dropna(subset=["juice_frozen"])
    by_cls = {int(y): safe_spearman(d.juice_frozen, d.game_juice)
              for y, d in MF.groupby("draft_year") if len(d) >= 15}
    f_r, f_lo, f_hi = pooled_fisher(
        [(safe_spearman(d.juice_frozen, d.game_juice), len(d)) for _, d in MF.groupby("draft_year") if len(d) >= 15])
    f_auc = auc_ci(MF.g_above, MF.groupby("pgm").juice_frozen.rank(pct=True)) if len(MF) else (np.nan,) * 3
    return {"R2": R2, "gain": gain("trad+pick+juice", 5), "gain_frozen": gain("trad+pick+juice_frozen", 6),
            "frozen_by_class": by_cls, "frozen_r": (f_r, f_lo, f_hi), "frozen_auc": f_auc, "n_frozen": len(MF)}


def evaluate_pool(pool):
    R = {"pool": pool}
    M = merged_for(pool)
    R["M"] = M
    print(f"  Pool '{pool}': {len(M)} players with both scores")

    rows = []
    for grp, d in list(M.groupby("pgm")) + [("ALL", M)]:
        if len(d) < 10:
            continue
        r = safe_spearman(d.juice_final, d.game_juice)
        lo, hi = fisher_ci(r, len(d))
        rows.append({"player_pool": pool, "position_group": grp, "n_players": len(d),
                     "spearman": r, "ci95_low": lo, "ci95_high": hi})
    R["stick"] = pd.DataFrame(rows)
    R["class"] = pd.DataFrame([{"player_pool": pool, "draft_year": int(y), "n_players": len(d),
                                "spearman": safe_spearman(d.juice_final, d.game_juice)}
                               for y, d in M.groupby("draft_year") if len(d) >= 15])
    R["rookie"] = rookie_corr(pool)

    g_ = M.juice_final
    answers = [
        risk_answer("Combine < 100 -> below average in games", g_ < BELOW_AVG, M.g_below),
        risk_answer("Combine < 90 -> below average in games", g_ < LOW_JUICE, M.g_below),
        risk_answer("Combine >= 110 -> above average in games", g_ >= ELITE, M.g_above),
        risk_answer("Combine >= 110 -> elite (110+) in games", g_ >= ELITE, M.g_elite),
        risk_answer("Combine >= 100 -> above average in games", g_ >= BELOW_AVG, M.g_above),
    ]
    R["answers"] = pd.DataFrame(answers).assign(player_pool=pool)
    R["bands"] = pd.DataFrame({
        "player_pool": pool, "band": BAND_LABELS,
        "n": [int((M.c_band == b).sum()) for b in BAND_LABELS],
        "pct_below_avg": [100 * M.loc[M.c_band == b, "g_below"].mean() for b in BAND_LABELS],
        "pct_above_avg": [100 * M.loc[M.c_band == b, "g_above"].mean() for b in BAND_LABELS],
        "pct_elite": [100 * M.loc[M.c_band == b, "g_elite"].mean() for b in BAND_LABELS],
        "mean_game_juice": [M.loc[M.c_band == b, "game_juice"].mean() for b in BAND_LABELS],
    })
    a1, a2 = auc_ci(M.g_above, M.c_p), auc_ci(M.g_above, M.juice_final)
    R["auc"] = pd.DataFrame([
        {"player_pool": pool, "score": "percentile within position", "auc": a1[0], "ci95_low": a1[1], "ci95_high": a1[2]},
        {"player_pool": pool, "score": "raw Combine Juice+", "auc": a2[0], "ci95_low": a2[1], "ci95_high": a2[2]}])

    cmat = np.full((3, 3), np.nan)
    for i, cc in enumerate(["T_plus", "A_plus", "E_plus"]):
        for j, gc in enumerate(["g_top_plus", "g_agil_plus", "g_expl_plus"]):
            d = M[[cc, gc]].dropna()
            if len(d) >= 15:
                cmat[i, j] = safe_spearman(d[cc], d[gc])
    R["cmat"] = cmat
    R["h2h"] = head_to_head(M)
    R.update(incremental_and_frozen(M))
    return R


RES = {pool: evaluate_pool(pool) for pool in POOLS}

# --- Table 1 and Combine Juice+ by round (pool independent) ---------------------------
ROUND_MEANS = None
if "draft_round" in FINAL.columns:
    ROUND_MEANS = FINAL.groupby(
        pd.cut(FINAL.draft_round.fillna(8), [0, 1, 2, 3, 5, 7, 8],
               labels=["R1", "R2", "R3", "R4-5", "R6-7", "No pick"]), observed=False).juice_final.mean()
board = FINAL[((FINAL.draft_overall_pick >= LOW_PICK) | FINAL.draft_overall_pick.isna())
              & (FINAL.juice_final >= HIGH_JUICE)].copy()
for pool in POOLS:
    board = board.merge(GAME[pool][["nfl_id", "game_juice"]].rename(
        columns={"game_juice": f"game_juice_{pool}"}), on="nfl_id", how="left")
board = board.sort_values("juice_final", ascending=False)
TABLE1 = pd.DataFrame({
    "Player": board.display_name, "Pos": board.nfl_position, "Class": board.draft_year.astype(int),
    "Pick": board.draft_overall_pick.map(lambda p: "none recorded" if pd.isna(p) else f"{int(p)}"),
    "Combine Juice+": board.juice_final.round(1), "Top-end (T+)": board["T_plus"].round(1),
    **{f"Game Juice+ ({p})": board[f"game_juice_{p}"].map(
        lambda v: "too few snaps" if pd.isna(v) else f"{v:.1f}") for p in POOLS}})

# ---------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------

reliability_df = pd.DataFrame([{
    "combine_d15": RELI["d15"], "combine_v1": RELI["v1"], "combine_fly10": RELI["fly10"],
    "game_top_full": GREL["top"], "game_expl_full": GREL["expl"], "game_agil_full": GREL["agil"],
    "game_top_single_play": GREL1["top"], "game_expl_single_play": GREL1["expl"],
    "game_agil_single_play": GREL1["agil"], "agility_included": GATE_G,
    **{f"C_weight_{k}": v for k, v in C_WEIGHTS.items()},
}])
save(reliability_df, "reliabilities.csv")
for pool in POOLS:
    save(GAME[pool].drop(columns=["raw"], errors="ignore").assign(player_pool=pool),
         f"game_juice_scores_{pool}.csv")
player_table = FINAL[["nfl_id", "display_name", "nfl_position", "pgm", "draft_year",
                      "juice_final", "E_plus", "T_plus", "A_plus", "juice_tracked_only",
                      "E_trk_plus", "T_trk_plus", "A_trk_plus"]]
for pool in POOLS:
    player_table = player_table.merge(
        GAME[pool][["nfl_id", "game_juice"]].rename(columns={"game_juice": f"game_juice_{pool}"}),
        on="nfl_id", how="left")
save(player_table, "juice_player_table.csv")
save(pd.concat([RES[p]["stick"] for p in POOLS]), "combine_to_game_comparison.csv")
save(pd.concat([RES[p]["class"] for p in POOLS]), "class_stability.csv")
save(pd.DataFrame([{"player_pool": p, "pooled_spearman": RES[p]["rookie"][0],
                    "n_players": RES[p]["rookie"][1]} for p in POOLS]), "rookie_season_only.csv")
save(persist_df, "season_persistence.csv")
save(pd.concat([RES[p]["answers"] for p in POOLS]), "risk_ratios.csv")
save(pd.concat([RES[p]["bands"] for p in POOLS]), "band_outcomes.csv")
save(pd.concat([RES[p]["auc"] for p in POOLS]), "auc.csv")
save(pd.concat([pd.DataFrame(RES[p]["cmat"], index=["T", "A", "E"],
                             columns=["game_top", "game_agility", "game_explosiveness"])
                .reset_index().rename(columns={"index": "combine_piece"}).assign(player_pool=p)
                for p in POOLS]), "component_matrix.csv")
h2h_parts = []
for p in POOLS:
    h = RES[p]["h2h"]
    if h:
        save(h["ngs_tab"].reset_index().rename(columns={"index": "score"}).assign(player_pool=p),
             f"head_to_head_{p}.csv")
        if len(h["top_tab"]):
            save(h["top_tab"].reset_index().rename(columns={"index": "score"}).assign(player_pool=p),
                 f"top_end_piece_{p}.csv")
inc_rows = []
for p in POOLS:
    for k, v in RES[p]["R2"].items():
        inc_rows.append({"player_pool": p, "model": k, "out_of_class_r2": v})
    inc_rows.append({"player_pool": p, "model": "GAIN trad+pick+juice over trad+pick",
                     "out_of_class_r2": RES[p]["gain"][0], "ci95_low": RES[p]["gain"][1],
                     "ci95_high": RES[p]["gain"][2]})
    inc_rows.append({"player_pool": p, "model": "GAIN trad+pick+juice_frozen over trad+pick",
                     "out_of_class_r2": RES[p]["gain_frozen"][0], "ci95_low": RES[p]["gain_frozen"][1],
                     "ci95_high": RES[p]["gain_frozen"][2]})
save(pd.DataFrame(inc_rows), "incremental_r2.csv")
save(pd.DataFrame([{"player_pool": p, "n": RES[p]["n_frozen"], "pooled_spearman": RES[p]["frozen_r"][0],
                    "ci95_low": RES[p]["frozen_r"][1], "ci95_high": RES[p]["frozen_r"][2],
                    "auc": RES[p]["frozen_auc"][0], "auc_lo": RES[p]["frozen_auc"][1],
                    "auc_hi": RES[p]["frozen_auc"][2]} for p in POOLS]), "held_out_formula.csv")
save(TABLE1, "draft_board_table1.csv")
if ROUND_MEANS is not None:
    save(ROUND_MEANS.rename("mean_combine_juice").reset_index(), "combine_juice_by_round.csv")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def f2(x):
    return f"{x:.2f}" if x is not None and np.isfinite(x) else "n/a"


print("\n================ FULL RUN RESULTS  ================")
print("\nCoverage:")
print(f"  Players with a Combine Juice+ score: {len(FINAL)}")
for pool in POOLS:
    print(f"  Pool '{pool}': {len(GAME[pool])} with game scores, {len(RES[pool]['M'])} with both")
print("\nReliabilities:")
print(reliability_df.T.to_string(header=False))
print("\nGame Juice+ season-to-season persistence (ceiling for any Combine link):")
print(persist_df.to_string(index=False) if len(persist_df) else "  not enough data")
if ROUND_MEANS is not None:
    print("\nCombine Juice+ by draft round:", ROUND_MEANS.round(1).to_dict())

for pool in POOLS:
    R = RES[pool]
    print(f"\n---------------- Pool '{pool}' (n = {len(R['M'])}) ----------------")
    print("\nCombine vs Game Juice+ (Spearman):")
    print(R["stick"].round(3).to_string(index=False))
    print("\nBy draft class:")
    print(R["class"].round(3).to_string(index=False))
    print(f"\nRookie season only (pooled over classes): {f2(R['rookie'][0])} (n = {R['rookie'][1]})")
    print("\nBand claims (Wilson 95% interval on the rate, bootstrap risk ratio vs all players):")
    print(R["answers"].drop(columns="player_pool").round(2).to_string(index=False))
    print("\nOutcomes by Combine band:")
    print(R["bands"].drop(columns="player_pool").round(1).to_string(index=False))
    print("\nAUC for above-average game movers (0.5 = coin flip):")
    print(R["auc"].drop(columns="player_pool").round(3).to_string(index=False))
    print("\nCombine piece (rows T, A, E) vs game piece (cols top, agility, explosiveness):")
    print(pd.DataFrame(R["cmat"], index=["T", "A", "E"],
                       columns=["game_top", "game_agility", "game_expl"]).round(2).to_string())
    h = R["h2h"]
    if h:
        print(f"\nHead-to-head vs stopwatch scores (same {h['n']} players), rank correlation with Game Juice+:")
        print(h["ngs_tab"].round(2).to_string())
        if len(h["top_tab"]):
            print("\nTop-end piece vs game top-end:")
            print(h["top_tab"].round(2).to_string())
        print("\nJuice+ edge (bootstrap 95% CI):",
              {k: tuple(round(x, 2) for x in v) for k, v in h["edge"].items()})
        print("Weight correlations:", {g: {k: round(v, 2) for k, v in d.items()}
                                       for g, d in h["weight_r"].items()})
        print("When the scores disagree (mean game percentile):",
              {k: (n, round(m, 1) if np.isfinite(m) else m) for k, (n, m) in h["disagree"].items()})
    else:
        print("\nHead-to-head: NGS / stopwatch columns not available.")
    print("\nOut-of-class R2:", {k: round(v, 3) for k, v in R["R2"].items()})
    print("Gain of Juice+ over traditional + pick:", tuple(round(x, 3) for x in R["gain"]))
    print(f"Held-out formula (n = {R['n_frozen']}): by class "
          f"{ {k: round(v, 2) for k, v in R['frozen_by_class'].items()} }; pooled "
          f"{f2(R['frozen_r'][0])} [{f2(R['frozen_r'][1])}, {f2(R['frozen_r'][2])}]; "
          f"AUC {f2(R['frozen_auc'][0])} [{f2(R['frozen_auc'][1])}, {f2(R['frozen_auc'][2])}]")
    print("Gain with held-out formula:", tuple(round(x, 3) for x in R["gain_frozen"]))

print("\nTable 1. Late-round / undrafted players flagged by Combine Juice+ (110+):")
print(TABLE1.to_string(index=False))
print("\nThese are in-sample results except the held-out formula and out-of-class R2 blocks.")

print(f"\nAll output files saved to: {OUT.resolve()}")