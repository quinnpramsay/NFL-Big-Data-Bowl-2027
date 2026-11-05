"""Same physics on both sides: acceleration-speed (A-S) profiles from Combine and game tracking.

For each player, every frame is binned by speed. The two highest accelerations in each bin are kept, and a line is
fit through them on the bins at 3 m/s (3.28 yd/s) and up:  acc = a0 - (a0 / v0) * speed.
  a0 = theoretical max acceleration at zero speed ("first-step power", yd/s^2)
  v0 = theoretical max speed ("top gear", yd/s)
This is the in-situ method from soccer GPS research (Morin et al. 2021, J Biomech; Sci Rep 2022). It needs no standing
start, no snap clock and no route context, so the Combine and the game give the same two numbers.

Limit: the fit uses speeds of 3 m/s and up, so it is weak for OL, who rarely run fast in games.

The same pass also counts preseason plays for every Combine-tracked player (audit item 9), since the main pipeline
keeps regular season only.

Run after juice.py. Writes to JUICE_OUT_DIR (default juice_nb_output); caches the game pass in JUICE_CACHE_DIR.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import (OUT, CACHE, get_file, load_run_id, save_with_run, PGM,  # noqa: E402
                          grouped_centered_mean, group_codes)

SEASONS = (2023, 2024, 2025)
CHUNK = 2_000_000
BIN = 0.2 * 1.0936                 # 0.2 m/s bins, in yd/s
MIN_FIT_SPEED = 3.0 * 1.0936       # 3 m/s, in yd/s
TOP_K = 2                          # highest accelerations kept per speed bin
ACC_CAP = 12.0                     # yd/s^2; above this a value is a tracking glitch, not a player
SMOOTH = 5                         # frames in the centered rolling mean applied to speed before differencing
CACHE_FILE = CACHE / "asp_bins_v1.csv"
RUN_ID = load_run_id()


def frame_accel(df, keys, t_col="t"):
    """Forward acceleration from smoothed speed, within each run of frames (one play, or one drill attempt).
    Vectorized: a centered rolling mean of speed, then a centered difference, never crossing a group edge."""
    df = df.sort_values(keys + [t_col]).reset_index(drop=True)
    gc = group_codes(df, keys)
    sm = grouped_centered_mean(df.s.to_numpy(), gc, half=SMOOTH // 2)
    t = df[t_col].to_numpy(float)
    n = len(df)
    nxt, prv = np.r_[np.arange(1, n), n - 1], np.r_[0, np.arange(n - 1)]
    has_n, has_p = gc[nxt] == gc, gc[prv] == gc
    has_n[-1], has_p[0] = False, False
    i1, i0 = np.where(has_n, nxt, np.arange(n)), np.where(has_p, prv, np.arange(n))
    dt = t[i1] - t[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(dt > 0, (sm[i1] - sm[i0]) / dt, np.nan)
    df = df.assign(s_sm=sm, acc=acc)
    return df[np.isfinite(df.acc) & (df.acc.abs() < ACC_CAP)]


def top_bins(df, group_cols):
    """Two highest accelerations per player per speed bin, plus the top smoothed speed."""
    df = df.assign(bin=np.floor(df.s_sm / BIN).astype(int))
    df = df[df.acc > 0]
    top = (df.sort_values("acc", ascending=False)
             .groupby(group_cols + ["bin"], sort=False).head(TOP_K)[group_cols + ["bin", "acc"]])
    smax = df.groupby(group_cols).s_sm.max().rename("s_max").reset_index()
    return top, smax


def fit_asp(top, smax, group_cols):
    """Fit acc = a0 + slope * speed on bins from 3 m/s up to 95% of the top speed. One refit after removing
    points outside the 95% band, as in the soccer method."""
    rows = []
    sm = smax.set_index(group_cols).s_max
    for key, d in top.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        vmax = sm.get(key if len(key) > 1 else key[0], np.nan)
        speed = (d.bin.to_numpy() + 0.5) * BIN
        acc = d.acc.to_numpy()
        m = (speed >= MIN_FIT_SPEED) & (speed <= 0.95 * vmax)
        rec = dict(zip(group_cols, key))
        rec.update({"s_max": vmax, "n_points": int(m.sum()), "a0": np.nan, "v0": np.nan, "r2": np.nan})
        if m.sum() >= 6:
            x, y = speed[m], acc[m]
            for _ in range(2):
                slope, a0 = np.polyfit(x, y, 1)
                res = y - (a0 + slope * x)
                keep = np.abs(res) <= 1.96 * res.std(ddof=2) if len(x) > 6 else np.ones(len(x), bool)
                if keep.all():
                    break
                x, y = x[keep], y[keep]
            slope, a0 = np.polyfit(x, y, 1)
            pred = a0 + slope * x
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            if slope < 0 and a0 > 0:
                rec.update({"a0": a0, "v0": -a0 / slope, "r2": r2, "n_points": len(x)})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Who to profile: every Combine-tracked player in the competition's position groups
# ---------------------------------------------------------------------
players = pd.read_csv(get_file("players.csv"))
games = pd.read_csv(get_file("games.csv"))
players["pgm"] = players.nfl_position.map(PGM)
players = players.dropna(subset=["pgm"])
ct = pd.read_csv(get_file("combine_tracking.csv"),
                 usecols=["nfl_id", "entity_type", "event_id", "drill_name", "time", "s"])
ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.notna()].copy()
ct["nfl_id"] = ct.nfl_id.astype(int)
ct = ct[ct.nfl_id.isin(players.nfl_id)]
KEEP = set(ct.nfl_id)
print(f"Combine-tracked players in scope: {len(KEEP)}")

# ---------------------------------------------------------------------
# Combine side: all drills pooled, and the 40 alone
# ---------------------------------------------------------------------
ct["time"] = pd.to_datetime(ct.time)
ct["t"] = (ct.time - ct.groupby("event_id").time.transform("min")).dt.total_seconds()
cf = frame_accel(ct[["nfl_id", "event_id", "drill_name", "t", "s"]], ["event_id"])
cf["scope"] = "combine_all"
cf40 = cf[cf.drill_name == "FORTY_YARD_DASH"].assign(scope="combine_40")
comb = []
for part in (cf, cf40):
    top, smax = top_bins(part, ["nfl_id", "scope"])
    comb.append(fit_asp(top, smax, ["nfl_id", "scope"]))
COMBINE_ASP = pd.concat(comb, ignore_index=True)
COMBINE_TOP = pd.concat([top_bins(p, ["nfl_id", "scope"])[0] for p in (cf, cf40)], ignore_index=True)
print("Combine A-S fits:", COMBINE_ASP.groupby("scope").a0.apply(lambda s: f"{s.notna().sum()} fitted").to_dict())

# ---------------------------------------------------------------------
# Game side: one chunked pass over all three season files, regular season and preseason
# ---------------------------------------------------------------------
STYPE = dict(zip(games.game_id, games.season_type))
SEASON = dict(zip(games.game_id, games.season))
KEYS = ["game_id", "play_id", "nfl_id"]


def summarize(df):
    """Top accelerations per player, season and season type for one block of whole plays, plus play counts."""
    df = df[df.nfl_id.isin(KEEP)]
    if len(df) == 0:
        return None, None, None
    df = df.copy()
    # Game tracking is 10 Hz. ISO time strings sort correctly, so the frame index within a play is the clock.
    df = df.sort_values(KEYS + ["time"])
    df["t"] = df.groupby(KEYS, sort=False).cumcount() * 0.1
    df["stype"] = df.game_id.map(STYPE).fillna("UNK")
    df["season"] = df.game_id.map(SEASON)
    fr = frame_accel(df[KEYS + ["t", "s", "stype", "season"]], KEYS)
    top, smax = top_bins(fr, ["nfl_id", "season", "stype"])
    plays = df.groupby(["nfl_id", "season", "stype"]).play_id.size().rename("frames").reset_index()
    nplay = (df.drop_duplicates(KEYS).groupby(["nfl_id", "season", "stype"]).size().rename("plays").reset_index())
    return top, smax, plays.merge(nplay, on=["nfl_id", "season", "stype"])


def reduce_top(parts):
    t = pd.concat(parts, ignore_index=True)
    return t.sort_values("acc", ascending=False).groupby(["nfl_id", "season", "stype", "bin"]).head(TOP_K)


def game_pass():
    tops, smaxs, counts = [], [], []
    for yr in SEASONS:
        name = f"game_tracking_{yr}.csv"
        print(f"  {name} ...", flush=True)
        left = None
        for i, chunk in enumerate(pd.read_csv(get_file(name), usecols=KEYS + ["time", "s"], chunksize=CHUNK)):
            if left is not None:
                chunk = pd.concat([left, chunk], ignore_index=True)
            last = chunk.iloc[-1]
            same = (chunk.game_id == last.game_id) & (chunk.play_id == last.play_id)
            left, body = chunk[same], chunk[~same]
            t, s, c = summarize(body)
            if t is not None:
                tops.append(t), smaxs.append(s), counts.append(c)
            if len(tops) > 40:                    # keep memory flat
                tops = [reduce_top(tops)]
        t, s, c = summarize(left)
        if t is not None:
            tops.append(t), smaxs.append(s), counts.append(c)
    top = reduce_top(tops)
    smax = pd.concat(smaxs).groupby(["nfl_id", "season", "stype"]).s_max.max().reset_index()
    cnt = pd.concat(counts).groupby(["nfl_id", "season", "stype"])[["frames", "plays"]].sum().reset_index()
    return top, smax, cnt


if CACHE_FILE.exists():
    print(f"Using cached A-S bins: {CACHE_FILE}")
    _c = pd.read_csv(CACHE_FILE)
    GTOP = _c[_c.kind == "top"][["nfl_id", "season", "stype", "bin", "acc"]]
    GSMAX = _c[_c.kind == "smax"][["nfl_id", "season", "stype", "s_max"]]
    GCNT = pd.read_csv(CACHE / "game_play_counts_by_type_v1.csv")
else:
    GTOP, GSMAX, GCNT = game_pass()
    pd.concat([GTOP.assign(kind="top"), GSMAX.assign(kind="smax")]).to_csv(CACHE_FILE, index=False)
    GCNT.to_csv(CACHE / "game_play_counts_by_type_v1.csv", index=False)

# Regular season pooled over all seasons, and by season (for year 1 vs year 2).
REG = GTOP[GTOP.stype == "REG"]
REGM = GSMAX[GSMAX.stype == "REG"]
pooled_top = REG.sort_values("acc", ascending=False).groupby(["nfl_id", "bin"]).head(TOP_K).assign(scope="game_reg")
pooled_smax = REGM.groupby("nfl_id").s_max.max().reset_index().assign(scope="game_reg")
GAME_ASP = fit_asp(pooled_top, pooled_smax, ["nfl_id", "scope"])
GAME_ASP_SEASON = fit_asp(REG.assign(scope="game_reg_season"), REGM.assign(scope="game_reg_season"),
                          ["nfl_id", "season", "scope"])

ASP = pd.concat([COMBINE_ASP, GAME_ASP, GAME_ASP_SEASON], ignore_index=True)
ASP = ASP.merge(players[["nfl_id", "display_name", "nfl_position", "pgm", "draft_year"]], on="nfl_id", how="left")
save_with_run(ASP, "asp_profiles.csv", RUN_ID)
# Bin points, for the overlay chart (Combine line over game line).
BINS = pd.concat([COMBINE_TOP.assign(season=np.nan), pooled_top.drop(columns=[], errors="ignore")], ignore_index=True)
BINS["speed"] = (BINS.bin + 0.5) * BIN
save_with_run(BINS[["nfl_id", "scope", "season", "bin", "speed", "acc"]], "asp_points.csv", RUN_ID)

# ---------------------------------------------------------------------
# Quick read: does the Combine profile line up with the game profile?
# ---------------------------------------------------------------------
from scipy.stats import spearmanr  # noqa: E402

wide = ASP[ASP.scope.isin(["combine_all", "combine_40", "game_reg"])].pivot_table(
    index="nfl_id", columns="scope", values=["a0", "v0", "s_max"])
wide.columns = [f"{v}_{s}" for v, s in wide.columns]
wide = wide.join(players.set_index("nfl_id")[["pgm"]])
rows = []
for grp, d in list(wide.groupby("pgm")) + [("ALL (within-group ranks)", None)]:
    if d is None:
        d = wide.copy()
        for c in wide.columns.drop("pgm"):
            d[c] = wide.groupby("pgm")[c].rank(pct=True)
    rec = {"group": grp}
    for comb_scope in ("combine_all", "combine_40"):
        for v in ("a0", "v0", "s_max"):
            a, b = f"{v}_{comb_scope}", f"{v}_game_reg"
            if a in d and b in d:
                x = d[[a, b]].dropna()
                rec[f"{v}:{comb_scope}->game"] = spearmanr(x[a], x[b])[0] if len(x) >= 15 else np.nan
                rec[f"n_{v}_{comb_scope}"] = len(x)
    rows.append(rec)
LINK = pd.DataFrame(rows)
print("\nCombine A-S profile vs game A-S profile (Spearman within group):")
print(LINK[["group"] + [c for c in LINK.columns if ":" in c]].round(2).to_string(index=False))
save_with_run(LINK, "asp_combine_vs_game.csv", RUN_ID)

# Fit quality, so weak OL fits are visible.
q = ASP[ASP.scope == "game_reg"].groupby("pgm").agg(fitted=("a0", lambda s: s.notna().mean()),
                                                     median_r2=("r2", "median"), median_points=("n_points", "median"))
print("\nGame A-S fit quality by group:")
print(q.round(2).to_string())

# ---------------------------------------------------------------------
# Preseason (audit item 9): how many players with no regular-season snap appear in preseason?
# ---------------------------------------------------------------------
cnt = GCNT.pivot_table(index="nfl_id", columns="stype", values="plays", aggfunc="sum").fillna(0)
for col in ("REG", "PRE"):
    if col not in cnt:
        cnt[col] = 0
pre = pd.DataFrame({"nfl_id": sorted(KEEP)}).merge(cnt[["REG", "PRE"]].reset_index(), on="nfl_id", how="left").fillna(0)
pre = pre.merge(players[["nfl_id", "pgm"]], on="nfl_id")
no_reg = pre[pre.REG == 0]
PRE_SUMMARY = {
    "players_in_scope": int(len(pre)),
    "no_regular_season_play": int(len(no_reg)),
    "no_regular_season_but_preseason": int((no_reg.PRE > 0).sum()),
    "preseason_games_in_files": int(games[games.season_type == "PRE"].game_id.nunique()),
    "regular_games_in_files": int(games[games.season_type == "REG"].game_id.nunique()),
    "preseason_only_by_group": no_reg[no_reg.PRE > 0].pgm.value_counts().to_dict(),
}
print("\nPreseason check:", json.dumps(PRE_SUMMARY))
save_with_run(pre, "preseason_counts.csv", RUN_ID)
(OUT / "preseason_summary.json").write_text(json.dumps({"run_id": RUN_ID, **PRE_SUMMARY}, indent=2))
print(f"\nSaved A-S profiles and preseason counts to {OUT.resolve()} (run {RUN_ID})")
