"""Route running and separation: the football link the competition prompt asks for (WR, TE, DB).

Run after juice_casework.py, on the same run ID. Pre-registered checks R1-R4 are in PRIMARY_TESTS.md; everything
else in routes_report.txt is exploratory.

  A. Route mechanics per play, snap to throw only (raw game tracking, cached): top speed before the throw, sideways
     acceleration before the throw, the sharpest turn, speed through it (cut speed), braking into it, burst out of it.
  B. The same cut mechanics in the Combine position drills (Combine cut speed, a secondary Combine piece), with the
     attempt-to-attempt agreement that says how well the drills can measure it.
  C. Separation at the throw: is it a player trait (R1), what is a yard worth (completion, EPA), and does cut quality
     buy it (R2: player level and within player, play by play).
  D. The chain in yards and points (R3): Combine piece -> same piece in games -> separation -> EPA, joint bootstrap.
     R4: game agility vs separation (scored only when juice.py ran with GAME_AGIL_WINDOW=prethrow).
  E. Separation over expected (SOE): expected separation from route, coverage, alignment, cushion, down and time to
     throw (gradient boosting, out of fold by game). A weekly board, and how well SOE to date predicts the rest.
  F. DBs in man coverage: one-to-one matching on the first 1.5 s of the route, separation allowed, links.
  G. Matchups: receiver vs DB Combine agility on matched man-coverage plays (two-way bootstrap).
  H. Route example for the figures: one route, two receivers (rule fixed below, chosen as an illustration, not a test).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import CACHE, OUT, PGM, get_file, load_run_id, read_out, save_with_run, section_rng  # noqa: E402

RUN_ID = load_run_id()
B = 2000
KEYS = ["game_id", "play_id", "nfl_id"]
LINES, CHECKS = [], []
rng = section_rng("routes", "setup")
FEATS = ["alat_pre", "turn_max", "decel_in", "accel_out", "retain", "v_cut", "peak_pre"]
FEAT_NAME = {"alat_pre": "sideways accel before throw", "turn_max": "sharpest turn", "decel_in": "braking into cut",
             "accel_out": "burst out of cut", "retain": "speed kept through cut", "v_cut": "cut speed",
             "peak_pre": "top speed before throw"}
ROUTE_CACHE = CACHE / "route_mech_v1.csv"
CUTS_CACHE = CACHE / "combine_cut_attempts_v1.csv"
POS_K = (0, 5, 10, 15)             # frames after the snap whose positions are kept for man-coverage matching
MAN_MATCH_MAX = 6.0                # yd: mean DB-receiver distance over those frames for a man match to count
DB_MIN_SNAPS, WR_MIN_ROUTES = 20, 50
R4_BASELINE = 0.139   # this script's R4 statistic on run ffa63e (all-frames game agility); scratchpad version +0.13


def out(text=""):
    LINES.append(str(text))
    print(text)


def hdr(text):
    """Section header; reseeds the generator from the section label (see juice_common.section_rng)."""
    global rng
    rng = section_rng("routes", text.split()[0])
    out("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def save(df, name):
    save_with_run(df, name, RUN_ID)


def fmt(r, lo, hi, nd=3):
    return f"{r:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def check(test_id, claim, est, lo, hi, n, passed, note=""):
    CHECKS.append({"test": test_id, "claim": claim, "estimate": est, "ci_low": lo, "ci_high": hi, "n": n,
                   "passed": bool(passed), "note": note})


def circ(d):
    return (d + 180.0) % 360.0 - 180.0


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_spearman(x, y, reps=B):
    """Spearman with a percentile bootstrap over players."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    vals = [spearman(x[i], y[i]) for i in rng.integers(0, n, (reps, n))]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return spearman(x, y), lo, hi, n


def within_slope(df, x, y, cluster="nfl_id", min_n=20, reps=400):
    """Slope of y on x after removing each player's mean of both (player fixed effects), with a cluster bootstrap
    over players. Done on per-player sums, so a bootstrap draw costs one matrix product."""
    t = df[[cluster, x, y]].dropna()
    t = t[t.groupby(cluster)[cluster].transform("size") >= min_n]
    if t[cluster].nunique() < 10:
        return np.nan, np.nan, np.nan, len(t), t[cluster].nunique()
    xc = t[x] - t.groupby(cluster)[x].transform("mean")
    yc = t[y] - t.groupby(cluster)[y].transform("mean")
    sxy = (xc * yc).groupby(t[cluster]).sum().to_numpy()
    sxx = (xc ** 2).groupby(t[cluster]).sum().to_numpy()
    k = len(sxy)
    W = rng.multinomial(k, np.full(k, 1 / k), size=reps)
    bs = (W @ sxy) / (W @ sxx)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(sxy.sum() / sxx.sum()), lo, hi, len(t), k


def cluster_ols_slope(df, x, y, cluster="nfl_id", reps=1000):
    """OLS slope of y on x with a cluster bootstrap over players (per-player sums)."""
    t = df[[cluster, x, y]].dropna()
    g = t.groupby(cluster)
    S = pd.DataFrame({"n": g.size(), "sx": g[x].sum(), "sy": g[y].sum(),
                      "sxx": (t[x] ** 2).groupby(t[cluster]).sum(),
                      "sxy": (t[x] * t[y]).groupby(t[cluster]).sum()}).to_numpy()

    def slope(tot):
        n, sx, sy, sxx, sxy = tot.T
        return (sxy - sx * sy / n) / (sxx - sx ** 2 / n)

    est = float(slope(S.sum(axis=0)[None, :])[0])
    k = len(S)
    W = rng.multinomial(k, np.full(k, 1 / k), size=reps)
    lo, hi = np.percentile(slope(W @ S), [2.5, 97.5])
    return est, lo, hi, len(t)


def split_half(df, col, cluster="nfl_id", reps=50):
    """Split-half reliability of a player mean (random halves of plays, Spearman-Brown), averaged over reps."""
    t = df[[cluster, col]].dropna().reset_index(drop=True)
    vals = []
    for _ in range(reps):
        h = rng.integers(0, 2, len(t))
        m = t.assign(h=h).groupby([cluster, "h"])[col].mean().unstack("h").dropna()
        r = spearman(m[0], m[1])
        if np.isfinite(r):
            vals.append(2 * r / (1 + r))
    return float(np.mean(vals)) if vals else np.nan


def adjust_cells(df, col, cells, min_n=40):
    """col minus the mean of the finest cell with min_n+ plays (cells listed coarse to fine)."""
    mu = df.groupby(list(cells[0]))[col].transform("mean")
    for c in cells[1:]:
        m = df.groupby(list(c))[col].transform("mean")
        n = df.groupby(list(c))[col].transform("count")
        mu = m.where(n >= min_n, mu)
    return df[col] - mu


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
players = pd.read_csv(get_file("players.csv"))
players["pgm"] = players.nfl_position.map(PGM)
games = pd.read_csv(get_file("games.csv"))[["game_id", "season", "week", "season_type"]]
REG = set(games.loc[games.season_type == "REG", "game_id"])
P = read_out("player_outcomes.csv", RUN_ID)
PP_COLS = KEYS + ["route_ran", "coverage_assignment", "pass_result", "play_nullified_by_penalty",
                  "separation_at_pass_forward", "target", "expected_points_added", "team_coverage_man_zone",
                  "team_coverage_type", "receiver_alignment", "lined_up_position", "cushion", "down",
                  "yards_to_go", "play_direction"]
pp = pd.read_csv(get_file("player_play.csv"), usecols=PP_COLS)


# =====================================================================
hdr("A. ROUTE MECHANICS PER PLAY, SNAP TO THROW (raw game tracking)")
# =====================================================================
def feats(g):
    ev = g.event.to_numpy()
    snap = np.flatnonzero(ev == "ball_snap")
    thr = np.flatnonzero(np.isin(ev, ["pass_forward", "pass_shovel"]))
    if len(snap) == 0 or len(thr) == 0 or thr[0] <= snap[0] + 5:
        return None
    i0, i1 = snap[0], thr[0]
    s = g.s.to_numpy()[i0:i1 + 1]
    d = g.dir.to_numpy()[i0:i1 + 1]
    x, y = g.x.to_numpy(), g.y.to_numpy()
    n = len(s)
    sm = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
    om = np.r_[0, circ(np.diff(d))]
    om_s = pd.Series(om).rolling(3, center=True, min_periods=1).mean().to_numpy()
    alat = np.where(sm >= 2, sm * np.abs(om_s) * np.pi / 180 / 0.1, np.nan)
    a_sorted = np.sort(alat[np.isfinite(alat)])[::-1]
    o = {"t_throw": (n - 1) / 10, "peak_pre": float(np.nanmax(sm)),
         "alat_pre": float(a_sorted[1]) if len(a_sorted) >= 2 else np.nan,
         "xt": x[i1], "yt": y[i1]}
    for k in POS_K:
        j = i0 + min(k, n - 1)
        o[f"x{k}"], o[f"y{k}"] = x[j], y[j]
    head = np.unwrap(np.deg2rad(d))
    idx = np.arange(n)
    turn = np.abs(np.rad2deg(head[np.clip(idx + 3, 0, n - 1)] - head[np.clip(idx - 2, 0, n - 1)]))
    turn = np.where(sm >= 2, turn, np.nan)
    if np.isfinite(turn).any():
        k = int(np.nanargmax(turn))
        o["turn_max"] = float(turn[k])
        j = max(0, k - 10) + int(np.argmax(sm[max(0, k - 10):k + 1]))
        kc = j + int(np.argmin(sm[j:k + 1]))
        v_entry, v_cut = sm[j], sm[kc]
        o.update(v_entry=float(v_entry), v_cut=float(v_cut),
                 retain=float(v_cut / v_entry) if v_entry > 0 else np.nan,
                 decel_in=float((v_entry - v_cut) / max((kc - j) / 10, 0.1)))
        b = min(n, kc + 9)
        if b - kc >= 3:
            m = kc + int(np.argmax(sm[kc:b]))
            o["accel_out"] = float((sm[m] - v_cut) / max((m - kc) / 10, 0.1))
    return o


def build_route_mech():
    ids = set(players.loc[players.pgm.isin(["WR", "TE", "DB"]), "nfl_id"])
    w = pp[pp.nfl_id.isin(ids) & pp.game_id.isin(REG) & pp.pass_result.notna()
           & (pp.play_nullified_by_penalty.astype(str) != "Y")
           & (pp.route_ran.notna() | (pp.coverage_assignment.notna() & (pp.coverage_assignment != "NO_ASSIGNMENT")))]
    want = pd.MultiIndex.from_frame(w[KEYS])
    rows = []
    for yr in sorted(games.season.unique()):
        name = f"game_tracking_{yr}.csv"
        print(f"  {name} ...", flush=True)
        left = None
        for ch in pd.read_csv(get_file(name), usecols=KEYS + ["time", "s", "dir", "x", "y", "event"],
                              chunksize=2_000_000):
            ch = ch[ch.nfl_id.isin(ids) & ch.game_id.isin(REG)]
            if left is not None:
                ch = pd.concat([left, ch], ignore_index=True)
            if not len(ch):
                left = None
                continue
            last = ch.iloc[-1]
            same = (ch.game_id == last.game_id) & (ch.play_id == last.play_id) & (ch.nfl_id == last.nfl_id)
            left, body = ch[same], ch[~same]
            body = body[pd.MultiIndex.from_frame(body[KEYS]).isin(want)]
            for key, g in body.sort_values(KEYS + ["time"]).groupby(KEYS, sort=False):
                f = feats(g)
                if f is not None:
                    rows.append({**dict(zip(KEYS, key)), **f})
        if left is not None and len(left):
            left = left[pd.MultiIndex.from_frame(left[KEYS]).isin(want)]
            for key, g in left.sort_values(KEYS + ["time"]).groupby(KEYS, sort=False):
                f = feats(g)
                if f is not None:
                    rows.append({**dict(zip(KEYS, key)), **f})
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


if ROUTE_CACHE.exists():
    print(f"Using cached route mechanics: {ROUTE_CACHE}")
    MECH = pd.read_csv(ROUTE_CACHE)
else:
    MECH = build_route_mech()
    MECH.to_csv(ROUTE_CACHE, index=False)
out(f"Player-plays with pre-throw tracking (WR/TE routes, DB coverage jobs): {len(MECH):,}")
out("Per play, frames from the snap to the throw only (3-frame smoothed speed, s >= 2 yd/s for turns):")
for f in FEATS:
    out(f"  {f:10s} {FEAT_NAME[f]}")


# =====================================================================
hdr("B. COMBINE CUT SPEED: THE SAME CUT MECHANICS IN THE POSITION DRILLS")
# =====================================================================
def cut_feats(s_raw, d):
    n = len(s_raw)
    sm = pd.Series(s_raw).rolling(3, center=True, min_periods=1).mean().to_numpy()
    om = np.r_[0, circ(np.diff(d))]
    om_s = pd.Series(om).rolling(3, center=True, min_periods=1).mean().to_numpy()
    alat = np.where(sm >= 2, sm * np.abs(om_s) * np.pi / 180 / 0.1, np.nan)
    a_sorted = np.sort(alat[np.isfinite(alat)])[::-1]
    o = {"alat_pre": float(a_sorted[1]) if len(a_sorted) >= 2 else np.nan, "peak_pre": float(np.nanmax(sm))}
    head = np.unwrap(np.deg2rad(d))
    idx = np.arange(n)
    turn = np.where(sm >= 2, np.abs(np.rad2deg(head[np.clip(idx + 3, 0, n - 1)] - head[np.clip(idx - 2, 0, n - 1)])),
                    np.nan)
    if np.isfinite(turn).any():
        k = int(np.nanargmax(turn))
        j = max(0, k - 10) + int(np.argmax(sm[max(0, k - 10):k + 1]))
        kc = j + int(np.argmin(sm[j:k + 1]))
        v_entry, v_cut = sm[j], sm[kc]
        o.update(turn_max=float(turn[k]), v_cut=float(v_cut),
                 retain=float(v_cut / v_entry) if v_entry > 0 else np.nan,
                 decel_in=float((v_entry - v_cut) / max((kc - j) / 10, 0.1)))
        b = min(n, kc + 9)
        if b - kc >= 3:
            m = kc + int(np.argmax(sm[kc:b]))
            o["accel_out"] = float((sm[m] - v_cut) / max((m - kc) / 10, 0.1))
    return o


if CUTS_CACHE.exists():
    print(f"Using cached Combine cut attempts: {CUTS_CACHE}")
    CA = pd.read_csv(CUTS_CACHE)
else:
    ct = pd.read_csv(get_file("combine_tracking.csv"),
                     usecols=["nfl_id", "entity_type", "event_id", "drill_type", "drill_name", "time", "s", "dir"])
    ct = ct[(ct.entity_type == "PLAYER") & ct.nfl_id.notna()
            & ct.drill_type.isin(["SKILL_DRILLS_WR", "SKILL_DRILLS_DB", "SKILL_DRILLS_TE"])]
    ct["nfl_id"] = ct.nfl_id.astype(int)
    ct = ct.sort_values(["event_id", "time"])
    rows = []
    for ev, g in ct.groupby("event_id", sort=True):
        s, dr = g.s.to_numpy(), g.dir.to_numpy()
        mv = np.flatnonzero(s >= 1.0)
        if len(mv) < 10:
            continue
        rows.append({"event_id": ev, "nfl_id": int(g.nfl_id.iloc[0]), "drill_name": g.drill_name.iloc[0],
                     **cut_feats(s[mv[0]:mv[-1] + 1], dr[mv[0]:mv[-1] + 1])})
    CA = pd.DataFrame(rows)
    CA.to_csv(CUTS_CACHE, index=False)
CA = CA.merge(players[["nfl_id", "pgm"]], on="nfl_id")
cnt = CA.groupby("drill_name").nfl_id.nunique()
CA = CA[CA.drill_name.isin(cnt[cnt >= 20].index)].copy()
pd_ = CA.groupby(["nfl_id", "pgm", "drill_name"])[FEATS].mean().reset_index()
for f in FEATS:
    pd_[f] = pd_.groupby("drill_name")[f].transform(lambda v: (v - v.mean()) / v.std())
CUTS = pd_.groupby(["nfl_id", "pgm"])[FEATS].mean().add_prefix("c_").reset_index()
CUTS["n_drills"] = pd_.groupby(["nfl_id", "pgm"]).size().to_numpy()
save(CUTS, "combine_cuts.csv")
# Attempt-level agreement: odd vs even attempts of the same drill, each z-scored within drill.
CA["k"] = CA.sort_values("event_id").groupby(["nfl_id", "drill_name"]).cumcount() % 2
pk = CA.groupby(["nfl_id", "pgm", "drill_name", "k"])[FEATS].mean().reset_index()
for f in FEATS:
    pk[f] = pk.groupby(["drill_name", "k"])[f].transform(lambda v: (v - v.mean()) / v.std())
H = pk.groupby(["nfl_id", "pgm", "k"])[FEATS].mean().unstack("k")
CUT_REL = []
out(f"Drill attempts: {len(CA):,}; players with a Combine cut composite: {len(CUTS)}")
out("Odd vs even attempts (Spearman of the player composites), by group:")
for grp in ("WR", "DB", "TE"):
    hh = H[H.index.get_level_values("pgm") == grp]
    if len(hh) < 15:
        continue
    rel = {f: spearman(*hh[f].dropna().to_numpy().T) for f in FEATS}
    CUT_REL += [{"pgm": grp, "feature": f, "odd_even_r": r, "players": len(hh)} for f, r in rel.items()]
    out(f"  {grp} (n={len(hh)}): " + ", ".join(f"{f} {r:+.2f}" for f, r in rel.items()))
save(pd.DataFrame(CUT_REL), "combine_cut_reliability.csv")


# =====================================================================
hdr("C. SEPARATION AT THE THROW (WR and TE routes): TRAIT, VALUE, AND WHAT BUYS IT")
# =====================================================================
d = (MECH.merge(pp, on=KEYS).merge(players[["nfl_id", "display_name", "pgm"]], on="nfl_id")
     .merge(games[["game_id", "season", "week"]], on="game_id"))
d = d[d.pgm.isin(["WR", "TE"]) & d.route_ran.notna() & d.separation_at_pass_forward.notna()
      & (d.route_ran != "SCREEN")].copy()
d["cov"] = d.team_coverage_man_zone.fillna("UNK")
d["tbin"] = np.clip((d.t_throw * 2).round() / 2, 1.5, 4.5)
d["sep"] = d.separation_at_pass_forward
CELLS = (("pgm", "route_ran"), ("pgm", "route_ran", "cov"), ("pgm", "route_ran", "cov", "tbin"))
for c in ["sep"] + FEATS:
    d[c + "_a"] = adjust_cells(d, c, CELLS)
    d[c + "_z"] = d[c + "_a"] / d[c + "_a"].std()
d["is_target"] = d.target.astype(str).str.lower().isin(["true", "1", "1.0"])
out(f"WR/TE route plays with separation and pre-throw tracking: {len(d):,} ({d.nfl_id.nunique()} players). "
    f"Separation and every feature are adjusted for route x man/zone x time to throw (finest cell with 40+ plays).")

wr = d[d.pgm == "WR"]
n_routes = wr.groupby("nfl_id").size()
wr50 = wr[wr.nfl_id.isin(n_routes[n_routes >= WR_MIN_ROUTES].index)]
rel_sep = split_half(wr50, "sep_a")
n_r1 = wr50.nfl_id.nunique()
out(f"\nR1  Separation as a player trait: split-half reliability {rel_sep:+.2f} "
    f"({n_r1} WRs with {WR_MIN_ROUTES}+ routes, median {int(n_routes[n_routes >= WR_MIN_ROUTES].median())} routes)")
check("R1", "WR separation at the throw is a stable player trait (split-half)", rel_sep, np.nan, np.nan, n_r1,
      rel_sep >= 0.5)

tg = d[d.is_target].copy()
tg["comp"] = (tg.pass_result == "C").astype(float)
tg["epa"] = tg.expected_points_added
comp_b = cluster_ols_slope(tg, "sep_a", "comp")
epa_b = cluster_ols_slope(tg.dropna(subset=["epa"]), "sep_a", "epa")
EPA_PER_YD = epa_b[0]
out(f"\nWhat a yard is worth, {len(tg):,} targets (cluster bootstrap over players):")
out(f"  completion rate per yard of separation  {fmt(100 * comp_b[0], 100 * comp_b[1], 100 * comp_b[2], 1)} points")
out(f"  EPA per target per yard                 {fmt(*epa_b[:3])}")
VALUE = pd.DataFrame([{"outcome": "completion", "per_yard": comp_b[0], "ci_low": comp_b[1], "ci_high": comp_b[2],
                       "targets": comp_b[3]},
                      {"outcome": "EPA per target", "per_yard": epa_b[0], "ci_low": epa_b[1], "ci_high": epa_b[2],
                       "targets": epa_b[3]}])
save(VALUE, "separation_value.csv")

# Player means (WR and TE), joined to the Combine pieces.
PM = (d.groupby(["nfl_id", "display_name", "pgm"])
      .agg(routes=("sep_a", "size"), sep=("sep_a", "mean"), sep_raw=("sep", "mean"),
           **{f: (f + "_a", "mean") for f in FEATS}).reset_index())
PM = PM.merge(P[["nfl_id", "draft_year", "A_trk_plus", "T_trk_plus", "E_trk_plus", "neg_forty", "juice_final",
                 "game_juice", "g_agil_plus", "g_top_plus"]], on="nfl_id", how="left")
PM = PM.merge(CUTS.drop(columns="pgm"), on="nfl_id", how="left")
save(PM, "route_players.csv")

LINKS = []
out("\nGame movement -> separation. Player level: Spearman of player means (WRs with 50+ routes).")
out("Play level: yards of separation per 1 SD of the feature on that play, within player (fixed effects).")
for grp, gl in (("WR", ["WR"]), ("TE", ["TE"]), ("WR+TE", ["WR", "TE"])):
    dd = d[d.pgm.isin(gl)]
    pm = PM[PM.pgm.isin(gl) & (PM.routes >= WR_MIN_ROUTES)]
    out(f"  {grp}:")
    for f in FEATS:
        pl_ = boot_spearman(pm[f], pm.sep)
        wi = within_slope(dd, f + "_z", "sep_a")
        LINKS.append({"group": grp, "feature": f, "player_r": pl_[0], "player_lo": pl_[1], "player_hi": pl_[2],
                      "players": pl_[3], "play_yd_per_sd": wi[0], "play_lo": wi[1], "play_hi": wi[2],
                      "plays": wi[3], "play_players": wi[4]})
        out(f"    {FEAT_NAME[f]:28s} player {fmt(*pl_[:3], 2)} n={pl_[3]:3d} | play {fmt(*wi[:3])} yd/SD "
            f"({wi[3]:,} plays, {wi[4]} players)")
LINKS = pd.DataFrame(LINKS)
save(LINKS, "route_links.csv")
r2 = LINKS[(LINKS.group == "WR") & (LINKS.feature == "v_cut")].iloc[0]
check("R2", "WR cut speed buys separation (player level and within player)", r2.player_r, r2.player_lo,
      r2.player_hi, int(r2.players), r2.player_lo > 0 and r2.play_lo > 0,
      f"within player {r2.play_yd_per_sd:+.3f} yd/SD [{r2.play_lo:+.3f}, {r2.play_hi:+.3f}], {int(r2.plays):,} plays")


# =====================================================================
hdr("D. THE CHAIN IN YARDS AND POINTS: COMBINE -> GAME MOVEMENT -> SEPARATION -> EPA (WR)")
# =====================================================================
out("a = SD of the game piece per SD of the Combine piece; b = yards of separation per SD of the game piece;")
out("a x b = yards of separation per SD at the Combine. Joint percentile bootstrap over players (4,000 draws).")
out(f"EPA per 100 targets = a x b x {EPA_PER_YD:.3f} EPA per yard x 100 (the EPA-per-yard interval is not propagated).")
CH_SPECS = [("40-yard dash", "neg_forty", "peak_pre", True), ("Combine drill cut speed", "c_v_cut", "v_cut", True),
            ("tracked top speed (T)", "T_trk_plus", "peak_pre", False),
            ("Juice+ Combine agility (A)", "A_trk_plus", "alat_pre", False),
            ("Juice+ Combine agility (A)", "A_trk_plus", "v_cut", False),
            ("Combine drill sideways accel", "c_alat_pre", "alat_pre", False)]
wp = PM[(PM.pgm == "WR") & (PM.routes >= WR_MIN_ROUTES)]


def zs(v):
    return (v - v.mean()) / v.std()


def chain_est(t, x, m):
    zx, zm = zs(t[x]), zs(t[m])
    a = np.polyfit(zx, zm, 1)[0]
    b = np.polyfit(zm, t.sep, 1)[0]
    c = np.polyfit(zx, t.sep, 1)[0]
    return a, b, a * b, c


CHAIN = []
for label, x, m, pre in CH_SPECS:
    t = wp[[x, m, "sep"]].dropna().reset_index(drop=True)
    if len(t) < 20:
        continue
    a, b, ab, c = chain_est(t, x, m)
    bs = np.array([chain_est(t.iloc[i].reset_index(drop=True), x, m) for i in rng.integers(0, len(t), (4000, len(t)))])
    lo_ab, hi_ab = np.percentile(bs[:, 2], [2.5, 97.5])
    rho1, rho2 = spearman(t[x], t[m]), spearman(t[m], t.sep)
    rho_d = boot_spearman(t[x], t.sep)
    prod = rho1 * rho2
    n_need = ((1.96 + 0.84) / np.arctanh(abs(prod))) ** 2 + 3 if abs(prod) > 0.01 else np.inf
    CHAIN.append({"combine_piece": label, "combine_col": x, "game_piece": FEAT_NAME[m], "game_col": m,
                  "preregistered": pre, "n": len(t), "a_sd_per_sd": a, "a_lo": np.percentile(bs[:, 0], 2.5),
                  "a_hi": np.percentile(bs[:, 0], 97.5), "b_yd_per_sd": b, "b_lo": np.percentile(bs[:, 1], 2.5),
                  "b_hi": np.percentile(bs[:, 1], 97.5), "ab_yd": ab, "ab_lo": lo_ab, "ab_hi": hi_ab,
                  "epa_per_100": 100 * ab * EPA_PER_YD, "epa_lo": 100 * lo_ab * EPA_PER_YD,
                  "epa_hi": 100 * hi_ab * EPA_PER_YD, "direct_yd": c, "direct_rho": rho_d[0],
                  "direct_rho_lo": rho_d[1], "direct_rho_hi": rho_d[2], "rho_link1": rho1, "rho_link2": rho2,
                  "players_needed_direct": n_need})
    out(f"\n  {label} -> {FEAT_NAME[m]} -> separation{'  (R3)' if pre else '  (exploratory)'}, n={len(t)}")
    out(f"    a {a:+.2f} SD/SD   b {b:+.3f} yd/SD   a x b {fmt(ab, lo_ab, hi_ab)} yd   "
        f"= {fmt(100 * ab * EPA_PER_YD, 100 * lo_ab * EPA_PER_YD, 100 * hi_ab * EPA_PER_YD, 2)} EPA per 100 targets")
    need = (f"which needs about {n_need:,.0f} players to detect directly (80% power)" if np.isfinite(n_need)
            else "too close to 0 to detect directly at any sample size")
    out(f"    direct, one jump: Spearman {fmt(*rho_d[:3], 2)}; links multiply to {prod:+.2f}, {need}")
CHAIN = pd.DataFrame(CHAIN)
save(CHAIN, "route_chain.csv")
for _, r in CHAIN[CHAIN.preregistered].iterrows():
    check("R3", f"Chain {r.combine_piece} -> {r.game_piece} -> separation (yd per Combine SD)", r.ab_yd, r.ab_lo,
          r.ab_hi, int(r.n), r.ab_lo > 0, f"{r.epa_per_100:+.2f} EPA per 100 targets")
sd_sep = wp.sep.std()
out(f"\nFor scale: WRs' route-adjusted separation varies by {sd_sep:.2f} yd (1 SD of player means), "
    f"worth {100 * sd_sep * EPA_PER_YD:.1f} EPA per 100 targets.")

r4 = boot_spearman(wp.g_agil_plus, wp.sep)
WINDOW = json.loads((OUT / "run_manifest.json").read_text())["settings"].get("GAME_AGIL_WINDOW", "all")
out(f"\nR4  Game agility (GAME_AGIL_WINDOW={WINDOW}) vs separation: {fmt(*r4[:3], 2)} n={r4[3]}; "
    f"run ffa63e (all frames after the snap), same statistic: {R4_BASELINE:+.3f}")
if WINDOW != "prethrow":
    out("    R4 not scored: the pre-throw window was reverted (run 947468 gave +0.114, not shown).")
else:
    check("R4", f"Pre-throw game agility links to separation better than the all-frames version ({R4_BASELINE:+.3f})",
          r4[0], r4[1], r4[2], r4[3], r4[0] > R4_BASELINE, f"descriptive: point estimate vs {R4_BASELINE:+.3f} (run ffa63e)")


# =====================================================================
hdr("E. SEPARATION OVER EXPECTED (SOE): A WEEKLY NUMBER")
# =====================================================================
CAT = ["route_ran", "team_coverage_man_zone", "team_coverage_type", "receiver_alignment", "lined_up_position", "pgm"]
NUM = ["cushion", "t_throw", "down", "yards_to_go"]
X = d[CAT + NUM].copy()
for c in CAT:
    X[c] = X[c].astype("category").cat.codes
cat_mask = [True] * len(CAT) + [False] * len(NUM)
exp = np.full(len(d), np.nan)
for tr, te in GroupKFold(n_splits=5).split(X, d.sep, d.game_id):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=80,
                                      categorical_features=cat_mask, random_state=0)
    m.fit(X.iloc[tr], d.sep.iloc[tr])
    exp[te] = m.predict(X.iloc[te])
d["sep_exp"] = exp
d["soe"] = d.sep - d.sep_exp
r2_oof = 1 - ((d.sep - d.sep_exp) ** 2).sum() / ((d.sep - d.sep.mean()) ** 2).sum()
r2_cells = 1 - (d.sep_a ** 2).sum() / ((d.sep - d.sep.mean()) ** 2).sum()
out(f"Expected separation from {', '.join(CAT + NUM)}.")
out(f"Out-of-fold R^2 (5 folds by game): {r2_oof:.3f}; the route x coverage x time cells above (in sample): "
    f"{r2_cells:.3f}")
wr50 = d[(d.pgm == "WR") & d.nfl_id.isin(n_routes[n_routes >= WR_MIN_ROUTES].index)]
out(f"Split-half reliability of SOE, WRs with {WR_MIN_ROUTES}+ routes: {split_half(wr50, 'soe'):+.2f}")
# Week-to-week: SOE to date (earlier seasons + this season through week k) vs SOE for the rest of the season.
STAB = []
for k in (4, 8, 12):
    rows = []
    for season in sorted(d.season.unique()):
        before = d[(d.season < season) | ((d.season == season) & (d.week <= k))]
        after = d[(d.season == season) & (d.week > k)]
        b_ = before.groupby("nfl_id").agg(soe_to_date=("soe", "mean"), n_to_date=("soe", "size"))
        a_ = after.groupby("nfl_id").agg(soe_rest=("soe", "mean"), n_rest=("soe", "size"))
        j = b_[b_.n_to_date >= 30].join(a_[a_.n_rest >= 30], how="inner").reset_index()
        rows.append(j.assign(season=season))
    j = pd.concat(rows)
    j = j.merge(players[["nfl_id", "pgm"]], on="nfl_id")
    j = j[j.pgm == "WR"]
    ids = j.nfl_id.unique()
    by = {i: g for i, g in j.groupby("nfl_id")}
    est = spearman(j.soe_to_date, j.soe_rest)
    bs = []
    for _ in range(B // 2):
        s_ = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
        bs.append(spearman(s_.soe_to_date, s_.soe_rest))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    STAB.append({"through_week": k, "rho": est, "ci_low": lo, "ci_high": hi, "player_seasons": len(j),
                 "players": len(ids)})
    out(f"  WR SOE through week {k:2d} vs rest of season: {fmt(est, lo, hi, 2)} ({len(j)} player-seasons, "
        f"30+ routes on each side)")
save(pd.DataFrame(STAB), "soe_stability.csv")
WEEK = (d.groupby(["nfl_id", "display_name", "pgm", "season", "week"])
        .agg(routes=("soe", "size"), soe=("soe", "mean"), sep=("sep", "mean"), sep_exp=("sep_exp", "mean"),
             cut_speed=("v_cut", "mean"), top_speed_pre=("peak_pre", "mean")).reset_index()
        .sort_values(["nfl_id", "season", "week"]))
g_ = WEEK.groupby(["nfl_id", "season"])
WEEK["routes_to_date"] = g_.routes.cumsum()
WEEK["soe_to_date"] = (WEEK.soe * WEEK.routes).groupby([WEEK.nfl_id, WEEK.season]).cumsum() / WEEK.routes_to_date
save(WEEK, "soe_weekly.csv")
SEAS = (d.groupby(["nfl_id", "display_name", "pgm", "season"])
        .agg(routes=("soe", "size"), soe=("soe", "mean"), sep=("sep", "mean"), cut_speed_adj=("v_cut_a", "mean"),
             top_speed_pre_adj=("peak_pre_a", "mean")).reset_index())
save(SEAS, "soe_players.csv")
top = SEAS[(SEAS.pgm == "WR") & (SEAS.routes >= 150)].sort_values("soe", ascending=False)
out("\nTop 5 WR seasons by SOE (150+ routes):")
for _, r in top.head(5).iterrows():
    out(f"  {r.display_name:22s} {int(r.season)}  {r.soe:+.2f} yd over expected on {int(r.routes)} routes")


# =====================================================================
hdr("F. DBs IN MAN COVERAGE (exploratory)")
# =====================================================================
dm = MECH.merge(pp[KEYS + ["coverage_assignment"]], on=KEYS).merge(players[["nfl_id", "pgm"]], on="nfl_id")
dbs = dm[(dm.pgm == "DB") & (dm.coverage_assignment == "MAN")]
recs = d[["game_id", "play_id", "nfl_id", "route_ran", "pgm", "tbin", "is_target", "pass_result",
          "expected_points_added", "separation_at_pass_forward"] + [f"{a}{k}" for k in POS_K for a in "xy"]
         + ["xt", "yt"]]
rec_by = {k: g for k, g in recs.groupby(["game_id", "play_id"])}
M = []
for (gid, pid), gd in dbs.groupby(["game_id", "play_id"]):
    gr = rec_by.get((gid, pid))
    if gr is None:
        continue
    cost = np.zeros((len(gd), len(gr)))
    for k in POS_K:
        cost += np.hypot(gd[f"x{k}"].to_numpy()[:, None] - gr[f"x{k}"].to_numpy()[None, :],
                         gd[f"y{k}"].to_numpy()[:, None] - gr[f"y{k}"].to_numpy()[None, :])
    cost /= len(POS_K)
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        if cost[i, j] <= MAN_MATCH_MAX:
            a, b = gd.iloc[i], gr.iloc[j]
            M.append({"game_id": gid, "play_id": pid, "nfl_id": a.nfl_id, "nfl_id_r": b.nfl_id, "match_dist": cost[i, j],
                      "sep": float(np.hypot(a.xt - b.xt, a.yt - b.yt)), "route": b.route_ran, "pgm_r": b.pgm,
                      "tbin": b.tbin, "target_r": b.is_target, "pass_result": b.pass_result,
                      "epa": b.expected_points_added, "official_sep_r": b.separation_at_pass_forward,
                      **{f: a[f] for f in FEATS}})
M = pd.DataFrame(M)
out(f"MAN-assigned DB plays: {len(dbs):,}; matched one-to-one to a tracked receiver within {MAN_MATCH_MAX:.0f} yd "
    f"(mean distance over the first 1.5 s): {len(M):,} plays, {M.nfl_id.nunique() if len(M) else 0} DBs")
if len(M) > 200:
    agree = spearman(M.sep, M.official_sep_r)
    out(f"Agreement of DB-receiver distance at the throw with the receiver's official separation: {agree:.2f} "
        f"(the scratchpad nearest-at-snap match gave 0.48)")
    M["sep_a"] = adjust_cells(M.assign(c="all"), "sep", (("c",), ("pgm_r", "route"), ("pgm_r", "route", "tbin")))
    for f in FEATS:
        M[f + "_z"] = adjust_cells(M.assign(c="all"), f, (("c",), ("pgm_r", "route")))
        M[f + "_z"] = M[f + "_z"] / M[f + "_z"].std()
    nsn = M.groupby("nfl_id").size()
    m20 = M[M.nfl_id.isin(nsn[nsn >= DB_MIN_SNAPS].index)]
    out(f"DBs with {DB_MIN_SNAPS}+ matched man snaps: {m20.nfl_id.nunique()}; split-half reliability of separation "
        f"allowed {split_half(m20, 'sep_a'):+.2f}")
    t_ = M[M.target_r.astype(bool)]
    if len(t_) > 100:
        cb = cluster_ols_slope(t_.assign(comp=(t_.pass_result == "C").astype(float)), "sep_a", "comp")
        eb = cluster_ols_slope(t_.dropna(subset=["epa"]), "sep_a", "epa")
        out(f"Targets on the matched receiver: {len(t_)}; per yard allowed: completion {100 * cb[0]:+.1f} points "
            f"[{100 * cb[1]:+.1f}, {100 * cb[2]:+.1f}], EPA {fmt(*eb[:3])}")
    DBL = []
    gpm = (m20.groupby("nfl_id").agg(snaps=("sep_a", "size"), sep=("sep_a", "mean"),
                                     **{f: (f + "_z", "mean") for f in FEATS}).reset_index()
           .merge(P[["nfl_id", "A_trk_plus", "neg_forty", "T_trk_plus", "g_agil_plus"]], on="nfl_id", how="left")
           .merge(CUTS.drop(columns="pgm"), on="nfl_id", how="left"))
    out("DB movement -> separation allowed (negative = tighter). Player level (Spearman) | within DB (yd per SD):")
    for f in FEATS:
        pl_ = boot_spearman(gpm[f], gpm.sep)
        wi = within_slope(M, f + "_z", "sep_a", min_n=15)
        DBL.append({"feature": f, "player_r": pl_[0], "player_lo": pl_[1], "player_hi": pl_[2], "players": pl_[3],
                    "play_yd_per_sd": wi[0], "play_lo": wi[1], "play_hi": wi[2], "plays": wi[3]})
        out(f"  {FEAT_NAME[f]:28s} {fmt(*pl_[:3], 2)} n={pl_[3]} | {fmt(*wi[:3])} ({wi[3]:,} plays)")
    out("Combine -> DB separation allowed (Spearman, DBs with 20+ matched snaps):")
    for c in ["A_trk_plus", "neg_forty", "c_alat_pre", "c_v_cut"]:
        r_ = boot_spearman(gpm[c], gpm.sep)
        DBL.append({"feature": "combine:" + c, "player_r": r_[0], "player_lo": r_[1], "player_hi": r_[2],
                    "players": r_[3]})
        out(f"  {c:11s} {fmt(*r_[:3], 2)} n={r_[3]}")
    save(pd.DataFrame(DBL), "db_man_links.csv")
    save(gpm, "db_man_players.csv")


# =====================================================================
hdr("G. MATCHUPS: RECEIVER VS DB COMBINE AGILITY ON MATCHED MAN PLAYS (exploratory)")
# =====================================================================
if len(M) > 200:
    cs = P.set_index("nfl_id")[["A_trk_plus"]] / 10          # plus scale: 10 points = 1 SD
    mm = M.join(cs, on="nfl_id").join(cs.add_suffix("_r"), on="nfl_id_r").dropna(subset=["A_trk_plus",
                                                                                          "A_trk_plus_r", "sep_a"])
    out(f"Plays with Combine agility for both players: {len(mm):,} ({mm.nfl_id.nunique()} DBs, "
        f"{mm.nfl_id_r.nunique()} receivers)")
    if len(mm) > 200:
        Xm = np.column_stack([np.ones(len(mm)), mm.A_trk_plus, mm.A_trk_plus_r])
        yv = mm.sep_a.to_numpy()
        bm = np.linalg.lstsq(Xm, yv, rcond=None)[0]
        dbs_u, rcs_u = mm.nfl_id.unique(), mm.nfl_id_r.unique()
        di = pd.Index(dbs_u).get_indexer(mm.nfl_id)
        rj = pd.Index(rcs_u).get_indexer(mm.nfl_id_r)
        bs = []
        for _ in range(B // 2):     # two-way (pigeonhole) bootstrap: resample DBs and receivers independently
            w = (np.bincount(rng.integers(0, len(dbs_u), len(dbs_u)), minlength=len(dbs_u))[di]
                 * np.bincount(rng.integers(0, len(rcs_u), len(rcs_u)), minlength=len(rcs_u))[rj])
            sw = np.sqrt(w)[:, None]
            bs.append(np.linalg.lstsq(Xm * sw, yv * sw[:, 0], rcond=None)[0])
        bs = np.array(bs)
        MU = pd.DataFrame([{"player": "DB", "yd_per_sd": bm[1], "ci_low": np.percentile(bs[:, 1], 2.5),
                            "ci_high": np.percentile(bs[:, 1], 97.5)},
                           {"player": "receiver", "yd_per_sd": bm[2], "ci_low": np.percentile(bs[:, 2], 2.5),
                            "ci_high": np.percentile(bs[:, 2], 97.5)}]).assign(plays=len(mm))
        save(MU, "matchups.csv")
        for _, r in MU.iterrows():
            out(f"  separation per 1 SD of {r.player} Combine agility: {fmt(r.yd_per_sd, r.ci_low, r.ci_high)} yd")


# =====================================================================
hdr("H. ROUTE EXAMPLE FOR THE FIGURES (illustration, not a test)")
# =====================================================================
# Rule (fixed before looking at the plays): among WRs with 50+ routes, the players at the 90th and 10th percentile of
# mean route-adjusted cut speed. Route = the break route (out, in, slant, corner, post) both ran most often. For each,
# among his plays on that route with a real cut (sharpest turn 45+ degrees), the play closest to his own typical play:
# smallest sum of standardized distances from his median cut speed and his median separation on that route.
cand = PM[(PM.pgm == "WR") & (PM.routes >= WR_MIN_ROUTES)].sort_values("v_cut").reset_index(drop=True)
hi_p = cand.iloc[int(round(0.9 * (len(cand) - 1)))]
lo_p = cand.iloc[int(round(0.1 * (len(cand) - 1)))]
BREAKS = ["OUT", "IN", "SLANT", "CORNER", "POST"]
both = d[d.nfl_id.isin([hi_p.nfl_id, lo_p.nfl_id]) & d.route_ran.isin(BREAKS) & (d.turn_max >= 45)]
rc = both.groupby(["route_ran", "nfl_id"]).size().unstack().dropna()
route = rc.min(axis=1).idxmax() if len(rc) else None
EX = []
if route is not None:
    for lab, pr in (("high cut speed", hi_p), ("low cut speed", lo_p)):
        r_ = both[(both.nfl_id == pr.nfl_id) & (both.route_ran == route)]
        dist = ((r_.v_cut - r_.v_cut.median()).abs() / r_.v_cut.std()
                + (r_.sep - r_.sep.median()).abs() / r_.sep.std())
        pick = r_.loc[dist.idxmin()]
        EX.append({"label": lab, "game_id": int(pick.game_id), "play_id": int(pick.play_id), "nfl_id": int(pr.nfl_id),
                   "display_name": pr.display_name, "route": route, "sep": pick.sep, "v_cut": pick.v_cut,
                   "player_v_cut_adj": pr.v_cut, "play_direction": pick.play_direction,
                   "season": int(pick.season)})
    want = {(e["game_id"], e["play_id"], e["nfl_id"]): e for e in EX}
    frames = []
    for yr in sorted({e["season"] for e in EX}):
        for ch in pd.read_csv(get_file(f"game_tracking_{yr}.csv"), usecols=KEYS + ["time", "x", "y", "s", "event"],
                              chunksize=2_000_000):
            ch = ch[ch.game_id.isin({k[0] for k in want})]
            ch = ch[[(a, b, c) in want for a, b, c in zip(ch.game_id, ch.play_id, ch.nfl_id)]]
            if len(ch):
                frames.append(ch)
    F = pd.concat(frames).sort_values(KEYS + ["time"])
    rows = []
    for key, g in F.groupby(KEYS):
        e = want[key]
        ev = g.event.to_numpy()
        i0 = np.flatnonzero(ev == "ball_snap")[0]
        i1 = np.flatnonzero(np.isin(ev, ["pass_forward", "pass_shovel"]))[0]
        g = g.iloc[i0:i1 + 1]
        sgn = 1 if e["play_direction"] == "right" else -1
        xr = sgn * (g.x.to_numpy() - g.x.iloc[0])
        yr_ = sgn * (g.y.to_numpy() - g.y.iloc[0])
        for k, (xx, yy, ss) in enumerate(zip(xr, yr_, g.s.to_numpy())):
            rows.append({**{c: e[c] for c in ("label", "display_name", "route", "sep", "v_cut", "player_v_cut_adj")},
                         "frame": k, "x": xx, "y": yy, "s": ss})
    save(pd.DataFrame(rows), "route_example.csv")
    for e in EX:
        out(f"  {e['label']}: {e['display_name']}, {route}, {e['season']} game {e['game_id']} play {e['play_id']}: "
            f"cut speed {e['v_cut']:.1f} yd/s, separation {e['sep']:.1f} yd")

# =====================================================================
hdr("I. PRE-REGISTERED ROUTE CHECKS (from PRIMARY_TESTS.md)")
# =====================================================================
CH = pd.DataFrame(CHECKS)
for _, r in CH.iterrows():
    ci = "" if not np.isfinite(r.ci_low) else f" [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
    out(f"  {r.test}  {'PASS' if r.passed else 'not shown'}  {r.claim}: {r.estimate:+.3f}{ci} n={r.n}"
        + (f"  ({r.note})" if r.note else ""))
save(CH, "route_checks.csv")
(OUT / "routes_report.txt").write_text(f"Run {RUN_ID}\n" + "\n".join(LINES) + "\n")
print(f"\nWrote {OUT / 'routes_report.txt'}")
