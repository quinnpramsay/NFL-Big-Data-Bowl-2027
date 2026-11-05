"""Where does the juice leak? Combine trait -> same trait in games -> football result.

Runs after juice.py and juice_physics.py, on the same run ID. Implements the analysis items of the November 3 audit:

  item 1  The chain, by position: link 1 (Combine -> game movement), link 2 (game movement -> football result),
          and the direct Combine -> result link, with play-level outcomes that fit each position.
  item 2  Decision rule: does the new tracked burst reach about 0.35 with NFL get-off? If so, lead with pass rush.
  item 3  What each tracked piece adds beyond the stopwatch, by position (replaces the pooled head-to-head),
          and out-of-class R^2 for the pooled scores.
  item 5  Mixed models: play outcome = context + player effect, player effect = beta x Combine trait + noise.
  item 6  "A Combine is worth N snaps" (snap takeover) and "unused juice" (does year 2 catch up?).
  item 9  Tiers, who reached the field, late-round flag, 600+ snap group, snaps confound, size adjustment.

PRIMARY tests are the ones written down in PRIMARY_TESTS.md before this run. Everything else is EXPLORATORY.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import OUT, get_file, load_run_id, read_out, save_with_run  # noqa: E402

RUN_ID = load_run_id()
B = 1000
rng = np.random.default_rng(2027)
LINES = []
GROUPS = ["DB", "DL", "EDGE", "OL", "TE", "WR"]


def out(text=""):
    LINES.append(str(text))
    print(text)


def hdr(text):
    out("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def save(df, name):
    save_with_run(df, name, RUN_ID)


# ---------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------

def pct(df, col, by="pgm"):
    return df.groupby(by)[col].rank(pct=True)


def residualize(v, controls):
    """OLS residual of v on an intercept plus the control columns (2-D array or None)."""
    if controls is None or controls.shape[1] == 0:
        return v - v.mean()
    X = np.column_stack([np.ones(len(v)), controls])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def partial_r(x, y, C):
    rx, ry = residualize(x, C), residualize(y, C)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_partial(x, y, C, reps=B):
    n = len(x)
    idx = rng.integers(0, n, (reps, n))
    vals = [partial_r(x[i], y[i], None if C is None else C[i]) for i in idx]
    return np.nanpercentile(vals, [2.5, 97.5])


def fisher_ci(r, n):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def group_dummies(g):
    d = pd.get_dummies(pd.Series(g), drop_first=True).astype(float).to_numpy()
    return d if d.shape[1] else None


def stack(*cols):
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    return np.column_stack([c if np.ndim(c) == 2 else np.asarray(c)[:, None] for c in cols])


def fmt(r, lo, hi):
    return f"{r:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ---------------------------------------------------------------------
# Load one run's tables
# ---------------------------------------------------------------------
CJ = read_out("combine_juice_scores.csv", RUN_ID)
GJ = read_out("game_juice_scores.csv", RUN_ID)
PC = read_out("game_play_counts.csv", RUN_ID)
PLM = read_out("play_level_movement.csv", RUN_ID)
MAN = json.loads((OUT / "run_manifest.json").read_text())
MIN_GAME_PLAYS = MAN["settings"]["MIN_GAME_PLAYS"]
try:
    ASP = read_out("asp_profiles.csv", RUN_ID)
except (FileNotFoundError, SystemExit) as e:
    print(f"A-S profiles not used ({e}). Run juice_physics.py after juice.py to include them.")
    ASP = None

LOWER = ["forty", "ten_yd_split", "three_cone", "short_shuttle"]
HIGHER = ["vertical", "broad_jump", "bench_reps"]


def ras_style(df):
    out_ = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("pgm"):
        p = {c: d[c].rank(pct=True, ascending=(c not in LOWER)) for c in LOWER + HIGHER
             if c in d and d[c].notna().sum() >= 15}
        if p:
            P_ = pd.DataFrame(p)
            out_.loc[d.index] = (10 * P_.mean(axis=1)).where(P_.notna().sum(axis=1) >= 4)
    return out_


P = CJ.copy()
P["ras_style"] = ras_style(P)
P["neg_forty"] = -P.forty
P["pick"] = P.draft_overall_pick.where(P.draft_overall_pick > 0, 260).fillna(260)
P["late"] = P.pick >= 140
gcols = ["nfl_id", "game_juice", "g_top_plus", "g_expl_plus", "g_agil_plus", "g_top", "g_expl", "g_agil",
         "npl", "nb", "confidence"]
P = P.merge(GJ[gcols], on="nfl_id", how="left").merge(
    PC.rename(columns={"npl": "npl_any"})[["nfl_id", "npl_any"]], on="nfl_id", how="left")
P["npl_any"] = P.npl_any.fillna(0)
P["log_snaps"] = np.log1p(P.npl)
if ASP is not None:
    for scope, tag in (("combine_all", "c"), ("combine_40", "c40"), ("game_reg", "g")):
        a = ASP[ASP.scope == scope].set_index("nfl_id")
        P[f"a0_{tag}"] = P.nfl_id.map(a.a0)
        P[f"v0_{tag}"] = P.nfl_id.map(a.v0)
# Tracked-only pieces, Combine side. Names used in tables.
PIECES = {"E": ("Burst (E, yard 2-10 split + accel)", "E_trk_plus", "g_expl_plus"),
          "T": ("Speed (T, fastest 10 yd)", "T_trk_plus", "g_top_plus"),
          "A": ("Agility (A, position drills)", "A_trk_plus", "g_agil_plus")}
out(f"Run {RUN_ID}. Combine-scored players: {len(P)}; with Game Juice+ ({MIN_GAME_PLAYS}+ plays): "
    f"{P.game_juice.notna().sum()}")

# ---------------------------------------------------------------------
# Football outcomes from player_play (regular season, plays not wiped out by penalty)
# ---------------------------------------------------------------------
games = pd.read_csv(get_file("games.csv"))[["game_id", "season", "season_type", "week"]]
PP_COLS = ["game_id", "play_id", "nfl_id", "lined_up_position", "route_ran", "target", "rec_yards",
           "yards_after_catch", "expected_yards_after_catch", "separation_at_pass_forward", "player_get_off",
           "time_to_pressure", "pressure_allowed", "peak_pressure_probability_allowed", "time_to_pressure_allowed",
           "dropback_duration", "pass_result", "coverage_assignment", "expected_points_added",
           "play_nullified_by_penalty"]
pp = pd.read_csv(get_file("player_play.csv"), usecols=PP_COLS)
pp = pp.merge(games, on="game_id")
pp = pp[(pp.season_type == "REG") & (pp.play_nullified_by_penalty.astype(str) != "Y")]
pp = pp.merge(P[["nfl_id", "pgm", "nfl_position"]], on="nfl_id")


def tf(s):
    return s.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1.0": 1.0, "0.0": 0.0, "1": 1.0, "0": 0.0})


def demean_cells(df, col, cells):
    """Subtract the mean of each context cell (falls back to the next coarser cell under 50 plays)."""
    mu = df.groupby(cells[-1])[col].transform("mean")
    for c in reversed(cells[:-1]):
        m, n = df.groupby(c)[col].transform("mean"), df.groupby(c)[col].transform("count")
        mu = m.where(n >= 50, mu)
    return df[col] - mu


route = pp.route_ran.notna() & pp.pgm.isin(["WR", "TE"])
catch = route & pp.yards_after_catch.notna() & pp.expected_yards_after_catch.notna()
rush = pp.pgm.isin(["DL", "EDGE"]) & pp.player_get_off.notna()
pro = (pp.pgm == "OL") & pp.pressure_allowed.notna()
cover = ((pp.pgm == "DB") & pp.coverage_assignment.notna() & (pp.coverage_assignment != "NO_ASSIGNMENT")
         & pp.pass_result.notna())

pp["sep"] = np.where(route, pp.separation_at_pass_forward, np.nan)
pp["sep_c"] = np.nan
pp.loc[route & pp.sep.notna(), "sep_c"] = demean_cells(pp[route & pp.sep.notna()], "sep",
                                                       [["nfl_position", "route_ran", "season"], ["nfl_position", "route_ran"], ["nfl_position"]])
pp["yac_oe"] = np.where(catch, pp.yards_after_catch - pp.expected_yards_after_catch, np.nan)
pp["rec_yds_route"] = np.where(route, pp.rec_yards.fillna(0), np.nan)
pp["get_off"] = np.where(rush, -pp.player_get_off, np.nan)          # faster = higher
pp["get_off_c"] = np.nan
pp.loc[rush, "get_off_c"] = demean_cells(pp[rush], "get_off", [["nfl_position", "lined_up_position", "season"],
                                                                 ["nfl_position", "lined_up_position"], ["nfl_position"]])
pp["pressure"] = np.where(rush & pp.pass_result.notna(), pp.time_to_pressure.notna().astype(float), np.nan)
pp["ttp"] = np.where(rush, -pp.time_to_pressure, np.nan)            # quicker pressure = higher
pp["ppp_held"] = np.where(pro, -pp.peak_pressure_probability_allowed, np.nan)   # lower pressure prob = higher
pp["ppp_held_c"] = np.nan
_ok = pro & pp.ppp_held.notna() & pp.dropback_duration.notna()
if _ok.any():
    # Longer dropbacks give a rusher more time, so take dropback length out first (linear, within position).
    for pos, d in pp[_ok].groupby("nfl_position"):
        X = np.column_stack([np.ones(len(d)), d.dropback_duration])
        pp.loc[d.index, "ppp_held_c"] = d.ppp_held - X @ np.linalg.lstsq(X, d.ppp_held, rcond=None)[0]
pp["ttp_allowed"] = np.where(pro, pp.time_to_pressure_allowed, np.nan)  # later pressure = higher
pp["held"] = np.where(pro, 1 - tf(pp.pressure_allowed), np.nan)
pp["cov_epa"] = np.where(cover, -pp.expected_points_added, np.nan)

# Player-level means with a low floor (audit item 1: lower floors, model by snap where possible).
OUTCOMES = {
    "WR": [("sep_c", "Separation at throw (route-adjusted)", 30), ("yac_oe", "YAC over expected", 10),
           ("rec_yds_route", "Yards per route", 50)],
    "TE": [("sep_c", "Separation at throw (route-adjusted)", 30), ("yac_oe", "YAC over expected", 10),
           ("rec_yds_route", "Yards per route", 50)],
    "DL": [("get_off_c", "Get-off (alignment-adjusted, faster = higher)", 30), ("pressure", "Pressure rate", 30),
           ("ttp", "Time to pressure (quicker = higher)", 5)],
    "EDGE": [("get_off_c", "Get-off (alignment-adjusted, faster = higher)", 30), ("pressure", "Pressure rate", 30),
             ("ttp", "Time to pressure (quicker = higher)", 5)],
    "OL": [("ppp_held_c", "Peak pressure prob. allowed (flipped, dropback-adjusted)", 30),
           ("ttp_allowed", "Time to pressure allowed", 5), ("held", "Pressure held rate", 30)],
    "DB": [("cov_epa", "EPA allowed in coverage (flipped)", 50)],
}
for grp, outs in OUTCOMES.items():
    for col, _, floor in outs:
        s = pp[pp.pgm == grp].groupby("nfl_id")[col].agg(["mean", "count"])
        P.loc[P.pgm == grp, "o_" + col] = P.loc[P.pgm == grp, "nfl_id"].map(s["mean"].where(s["count"] >= floor))
        P.loc[P.pgm == grp, "n_" + col] = P.loc[P.pgm == grp, "nfl_id"].map(s["count"])

# =====================================================================
hdr("A. WHAT TRACKING ADDS BEYOND THE STOPWATCH, BY POSITION (audit item 3)")
# =====================================================================
out("Partial correlation of each tracked piece with Game Juice+, on within-group percentile ranks, holding the")
out("controls fixed. 95% bootstrap intervals (1,000 resamples of players). Pooled rows add group dummies.")
CONTROL_SETS = [("the 40", ["neg_forty"]),
                ("40 + RAS-style", ["neg_forty", "ras_style"]),
                ("40 + RAS-style + snaps + pick", ["neg_forty", "ras_style", "log_snaps", "pick"])]
ADD_ROWS = []
SETS = [(g, [g]) for g in GROUPS] + [("WR + DB", ["WR", "DB"]), ("ALL", GROUPS)]


def adds(piece_col, target, groups, controls):
    cols = [piece_col, target] + controls
    d = P[P.pgm.isin(groups)].dropna(subset=cols).copy()
    if len(d) < 25:
        return None
    for c in cols:
        d["X_" + c] = d.groupby("pgm")[c].rank(pct=True)
    x, y = d["X_" + piece_col].to_numpy(), d["X_" + target].to_numpy()
    C = stack(*[d["X_" + c].to_numpy() for c in controls], group_dummies(d.pgm) if len(groups) > 1 else None)
    r = partial_r(x, y, C)
    lo, hi = boot_partial(x, y, C)
    return r, lo, hi, len(d)


for key, (label, pcol, gpiece) in PIECES.items():
    for set_name, groups in SETS:
        for cname, controls in CONTROL_SETS:
            for target, tname in (("game_juice", "Game Juice+"), (gpiece, "matching game piece")):
                res = adds(pcol, target, groups, controls)
                if res:
                    ADD_ROWS.append({"piece": label, "group": set_name, "holding": cname, "target": tname,
                                     "partial_r": res[0], "ci_low": res[1], "ci_high": res[2], "n": res[3]})
ADDS = pd.DataFrame(ADD_ROWS)
save(ADDS, "tracking_adds.csv")
headline = ADDS[(ADDS.target == "Game Juice+") & (ADDS.holding == "the 40")]
tab = headline.assign(cell=[fmt(r, l, h) + f" n={n}" for r, l, h, n in
                            zip(headline.partial_r, headline.ci_low, headline.ci_high, headline.n)])
out("\nHolding the 40, target Game Juice+:")
out(tab.pivot(index="group", columns="piece", values="cell").to_string())
out("\nPooled (ALL), every control set:")
allrows = ADDS[(ADDS.group == "ALL") & (ADDS.target == "Game Juice+")]
for _, r in allrows.iterrows():
    out(f"  {r.piece:32s} holding {r.holding:32s} {fmt(r.partial_r, r.ci_low, r.ci_high)}  n={r.n}")

# By draft class, agility holding the 40 (does it repeat in each class?)
out("\nAgility holding the 40, by draft class (all groups with an agility score):")
for yr, d in P.groupby("draft_year"):
    sub = P.index.isin(d.index)
    tmp = P[sub]
    cols = ["A_trk_plus", "game_juice", "neg_forty"]
    tmp = tmp.dropna(subset=cols)
    if len(tmp) < 25:
        continue
    X = {c: tmp.groupby("pgm")[c].rank(pct=True).to_numpy() for c in cols}
    C = stack(X["neg_forty"], group_dummies(tmp.pgm))
    r = partial_r(X["A_trk_plus"], X["game_juice"], C)
    lo, hi = boot_partial(X["A_trk_plus"], X["game_juice"], C)
    out(f"  {int(yr)}: {fmt(r, lo, hi)}  n={len(tmp)}")

# Out-of-class R^2 for the pooled scores (why the head-to-head is not the headline).
out("\nOut-of-class R^2 predicting Game Juice+ (within-group ranks; each class predicted by a model fit on the others):")
MODELS = [("Juice+ 50/50", ["juice_5050"]), ("40 + tracked Juice+", ["neg_forty", "juice_tracked_only"]),
          ("Official 40", ["neg_forty"]), ("Tracked Juice+", ["juice_tracked_only"]),
          ("40 + tracked agility", ["neg_forty", "A_trk_plus"]),
          ("RAS-style", ["ras_style"]), ("Pick", ["pick"]), ("NGS", ["ngs_athleticism_score"])]
need = sorted({c for _, cs in MODELS for c in cs if c != "A_trk_plus"} | {"game_juice"})
R2D = P.dropna(subset=need).copy()
for c in need + ["A_trk_plus"]:
    R2D["X_" + c] = R2D.groupby("pgm")[c].rank(pct=True)
R2D["X_A_trk_plus"] = R2D["X_A_trk_plus"].fillna(0.5)   # groups without an agility score get the middle rank
PRED = {}
for name, cs in MODELS:
    pred = pd.Series(np.nan, index=R2D.index)
    for yr in R2D.draft_year.unique():
        tr, te = R2D.draft_year != yr, R2D.draft_year == yr
        X = np.column_stack([np.ones(tr.sum())] + [R2D.loc[tr, "X_" + c] for c in cs])
        beta = np.linalg.lstsq(X, R2D.loc[tr, "X_game_juice"], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum())] + [R2D.loc[te, "X_" + c] for c in cs]) @ beta
    PRED[name] = pred.to_numpy()
y = R2D.X_game_juice.to_numpy()


def r2(p, y_):
    return 1 - ((y_ - p) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()


idx = rng.integers(0, len(y), (B, len(y)))
R2_ROWS = []
for name, _ in MODELS:
    vals = [r2(PRED[name][i], y[i]) for i in idx]
    R2_ROWS.append({"model": name, "r2": r2(PRED[name], y), "ci_low": np.percentile(vals, 2.5),
                    "ci_high": np.percentile(vals, 97.5), "n": len(y)})
R2T = pd.DataFrame(R2_ROWS).sort_values("r2", ascending=False)
out(R2T.round(3).to_string(index=False))
for a, b_ in (("Juice+ 50/50", "Official 40"), ("40 + tracked Juice+", "Official 40"),
              ("40 + tracked agility", "Official 40")):
    d_ = [r2(PRED[a][i], y[i]) - r2(PRED[b_][i], y[i]) for i in idx]
    out(f"  {a} minus {b_}: {r2(PRED[a], y) - r2(PRED[b_], y):+.3f} "
        f"[{np.percentile(d_, 2.5):+.3f}, {np.percentile(d_, 97.5):+.3f}]")
save(R2T, "out_of_class_r2.csv")

# =====================================================================
hdr("B. THE CHAIN: COMBINE TRAIT -> GAME TRAIT -> FOOTBALL RESULT (audit item 1)")
# =====================================================================
out("Spearman within group. Link 1 = Combine piece vs the same game piece. Link 2 = game piece vs outcome.")
out("Direct = Combine piece vs outcome. 95% Fisher intervals. Outcome floors are low on purpose (see OUTCOMES).")
CHAIN = []
TRAITS = [("E", "E_trk_plus", "g_expl_plus"), ("T", "T_trk_plus", "g_top_plus"), ("A", "A_trk_plus", "g_agil_plus")]
if ASP is not None:
    TRAITS += [("a0 (A-S)", "a0_c", "a0_g"), ("v0 (A-S)", "v0_c", "v0_g")]
TRAITS += [("40 (stopwatch)", "neg_forty", "g_top_plus")]


def sp(d, a, b_):
    x = d[[a, b_]].dropna()
    if len(x) < 12:
        return np.nan, np.nan, np.nan, len(x)
    r = stats.spearmanr(x[a], x[b_])[0]
    return (r, *fisher_ci(r, len(x)), len(x))


for grp, outs in OUTCOMES.items():
    d = P[P.pgm == grp]
    for tname, ccol, gcol in TRAITS:
        if ccol not in d or gcol not in d:
            continue
        l1 = sp(d, ccol, gcol)
        for ocol, olabel, _ in outs:
            l2, dr = sp(d, gcol, "o_" + ocol), sp(d, ccol, "o_" + ocol)
            CHAIN.append({"group": grp, "trait": tname, "outcome": olabel,
                          "link1_combine_to_game": l1[0], "l1_lo": l1[1], "l1_hi": l1[2], "l1_n": l1[3],
                          "link2_game_to_result": l2[0], "l2_lo": l2[1], "l2_hi": l2[2], "l2_n": l2[3],
                          "direct_combine_to_result": dr[0], "d_lo": dr[1], "d_hi": dr[2], "d_n": dr[3]})
CHAIN = pd.DataFrame(CHAIN)


def leak(r):
    """Where the chain breaks: the first link whose interval includes 0."""
    if not (r.l1_lo > 0 or r.l1_hi < 0):
        return "link 1 (Combine -> game)"
    if not (r.l2_lo > 0 or r.l2_hi < 0):
        return "link 2 (game -> result)"
    return "holds"


CHAIN["breaks_at"] = CHAIN.apply(leak, axis=1)
save(CHAIN, "chain_links.csv")
for grp in OUTCOMES:
    out(f"\n{grp}")
    d = CHAIN[CHAIN.group == grp]
    for _, r in d.iterrows():
        out(f"  {r.trait:16s} -> {r.outcome[:44]:44s} L1 {r.link1_combine_to_game:+.2f} (n={r.l1_n:3d})  "
            f"L2 {r.link2_game_to_result:+.2f} (n={r.l2_n:3d})  direct {r.direct_combine_to_result:+.2f}  "
            f"[{r.breaks_at}]")

# Decision rule from the audit (item 2): tracked burst vs NFL get-off.
hdr("C. DECISION RULE: TRACKED BURST VS NFL GET-OFF (audit item 2)")
rush_d = P[P.pgm.isin(["DL", "EDGE"])].dropna(subset=["E_trk_plus", "o_get_off_c"]).copy()
for c in ["E_trk_plus", "o_get_off_c", "pick", "neg_forty", "split"]:
    rush_d["X_" + c] = rush_d.groupby("pgm")[c].rank(pct=True)
xb, yb = rush_d.X_E_trk_plus.to_numpy(), rush_d.X_o_get_off_c.to_numpy()
C_g = group_dummies(rush_d.pgm)
r_b = partial_r(xb, yb, C_g)
lo_b, hi_b = boot_partial(xb, yb, C_g)
r_bp = partial_r(xb, yb, stack(rush_d.X_pick.to_numpy(), C_g))
fd = rush_d.dropna(subset=["neg_forty"])
r_40 = partial_r(fd.groupby("pgm").neg_forty.rank(pct=True).to_numpy(),
                 fd.groupby("pgm").o_get_off_c.rank(pct=True).to_numpy(), group_dummies(fd.pgm))
out(f"Tracked burst (yard 2-10) vs NFL get-off, DL + EDGE: {fmt(r_b, lo_b, hi_b)}  n={len(rush_d)}")
out(f"  after removing draft pick: {r_bp:+.3f};  the official 40 on the same players: {r_40:+.3f}")
LEAD = "pass rush (DL/EDGE first step)" if r_b >= 0.35 else "WR and DB agility"
out(f"Audit rule: lead with pass rush if this reaches about 0.35, otherwise WR and DB agility -> {LEAD}")
DECISION = {"burst_vs_getoff": r_b, "ci": [lo_b, hi_b], "n": int(len(rush_d)), "after_pick": r_bp,
            "forty_vs_getoff": r_40, "lead_story": LEAD}

# =====================================================================
hdr("D. MIXED MODELS: PLAY OUTCOME = CONTEXT + PLAYER EFFECT (audit item 5)")
# =====================================================================
out("Each play's outcome (already context-adjusted) = beta x Combine trait + player random effect + noise.")
out("beta is per 1 SD of the trait within position. 'transfer' = beta / SD of the player effect in a model with no")
out("trait, a correlation-like number: the share of true player-level spread the Combine trait lines up with.")
out("Every player with at least one play counts, in proportion to how much he shows. No snap floor.")

for c in ["E_trk_plus", "T_trk_plus", "A_trk_plus", "neg_forty", "a0_c", "v0_c"]:
    if c in P:
        P["z_" + c] = P.groupby("pgm")[c].transform(lambda s: (s - s.mean()) / s.std())
PLM2 = PLM.merge(P[["nfl_id"] + [c for c in P.columns if c.startswith("z_")]], on="nfl_id")
burst_ok = PLM2.role.astype(str).str.lower().eq("true") & (PLM2.s0 < 1.0)
PLM2["expl_play"] = PLM2.expl_c.where(burst_ok)
PLM2["peak_play"] = PLM2.peak_c.where(PLM2.n_after >= 20)
PLM2["alat_play"] = PLM2.alat_c.where(PLM2.n_after >= 20)
pp2 = pp.merge(P[["nfl_id"] + [c for c in P.columns if c.startswith("z_")]], on="nfl_id")

MM_SPECS = []
for grp in GROUPS:
    MM_SPECS += [(grp, PLM2, "peak_play", "game top speed", "z_T_trk_plus"),
                 (grp, PLM2, "peak_play", "game top speed", "z_neg_forty"),
                 (grp, PLM2, "expl_play", "game burst", "z_E_trk_plus"),
                 (grp, PLM2, "alat_play", "game agility", "z_A_trk_plus")]
    for ocol, olabel, _ in OUTCOMES[grp]:
        if ocol in ("ttp", "ttp_allowed"):
            continue                              # too few plays per player for a play-level model
        for trait in ("z_E_trk_plus", "z_T_trk_plus", "z_A_trk_plus", "z_neg_forty"):
            MM_SPECS.append((grp, pp2, ocol, olabel, trait))
    if ASP is not None:
        MM_SPECS += [(grp, PLM2, "expl_play", "game burst", "z_a0_c"), (grp, PLM2, "peak_play", "game top speed", "z_v0_c")]


def _tau2(ybar, v, X):
    """Between-player variance by Paule-Mandel: the tau^2 at which the weighted residual sum of squares of the
    player means equals its degrees of freedom. Bisection, so it never fails to converge."""
    k, p_ = X.shape

    def q(t2):
        w = 1 / (t2 + v)
        W = np.sqrt(w)
        beta = np.linalg.lstsq(X * W[:, None], ybar * W, rcond=None)[0]
        return (w * (ybar - X @ beta) ** 2).sum() - (k - p_)

    lo, hi = 0.0, max(float(np.var(ybar)) * 10, 1e-9)
    if q(lo) <= 0:
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if q(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def mixed(df, ycol, xcol):
    """Random-intercept model with a player-level trait, fit in closed form:
        y_play = mu + beta * x_player + u_player + e,  u ~ N(0, tau^2), e ~ N(0, sigma^2).
    sigma^2 is the pooled within-player variance, tau^2 comes from Paule-Mandel, and beta is the GLS estimate
    on the player means with weights 1 / (tau^2 + sigma^2 / n). This is what lme4 or MixedLM fit, without the
    convergence failures MixedLM showed here when the player share of variance is small."""
    d = df[[ycol, xcol, "nfl_id"]].dropna()
    if d.nfl_id.nunique() < 15 or len(d) < 200:
        return None
    g = d.groupby("nfl_id")
    n, ybar, x = g[ycol].size().to_numpy(float), g[ycol].mean().to_numpy(), g[xcol].first().to_numpy()
    ss = (g[ycol].var(ddof=1).fillna(0) * (g[ycol].size() - 1)).sum()
    sigma2 = float(ss / max((n - 1).sum(), 1))
    v = sigma2 / n
    X0, X1 = np.ones((len(n), 1)), np.column_stack([np.ones(len(n)), x])
    tau2_0 = _tau2(ybar, v, X0)          # no trait: the player-level spread to explain
    tau2_1 = _tau2(ybar, v, X1)
    w = 1 / (tau2_1 + v)
    XtWX = X1.T @ (X1 * w[:, None])
    beta = np.linalg.solve(XtWX, X1.T @ (w * ybar))
    se = float(np.sqrt(np.linalg.inv(XtWX)[1, 1]))
    sd_u = float(np.sqrt(tau2_0))
    b_ = float(beta[1])
    if sd_u <= 0:
        return {"beta": b_, "beta_lo": b_ - 1.96 * se, "beta_hi": b_ + 1.96 * se,
                "p": float(2 * stats.norm.sf(abs(b_ / se))), "sd_player_effect": 0.0, "transfer": np.nan,
                "transfer_lo": np.nan, "transfer_hi": np.nan, "players": len(n), "plays": int(n.sum()), "icc": 0.0,
                "share_explained": np.nan}
    return {"beta": b_, "beta_lo": b_ - 1.96 * se, "beta_hi": b_ + 1.96 * se,
            "p": float(2 * stats.norm.sf(abs(b_ / se))), "sd_player_effect": sd_u,
            "transfer": b_ / sd_u, "transfer_lo": (b_ - 1.96 * se) / sd_u, "transfer_hi": (b_ + 1.96 * se) / sd_u,
            "players": len(n), "plays": int(n.sum()), "icc": tau2_0 / (tau2_0 + sigma2),
            "share_explained": 1 - tau2_1 / tau2_0}


MM_ROWS = []
for grp, frame, ycol, ylabel, xcol in MM_SPECS:
    if xcol not in frame:
        continue
    res = mixed(frame[frame.pgm == grp], ycol, xcol)
    if res:
        MM_ROWS.append({"group": grp, "outcome": ylabel, "trait": xcol.replace("z_", ""), **res})
MM = pd.DataFrame(MM_ROWS)
# When players differ very little on an outcome (ICC under 0.01), the player-effect SD is tiny and the transfer ratio
# is unstable. Those rows are flagged, not read as findings.
MM["stable"] = MM.icc >= 0.01
save(MM, "mixed_models.csv")
for grp in GROUPS:
    d = MM[MM.group == grp]
    if len(d):
        out(f"\n{grp}")
        for _, r in d.iterrows():
            out(f"  {r.outcome[:44]:44s} ~ {r.trait:12s} transfer {fmt(r.transfer, r.transfer_lo, r.transfer_hi)}"
                f"  players={r.players:3d} plays={r.plays:6d} icc={r.icc:.3f}" + ("" if r.stable else "  UNSTABLE (icc<0.01)"))

# =====================================================================
hdr("E. A COMBINE IS WORTH N SNAPS (audit item 6)")
# =====================================================================
out("Rookie season only. Target = the 95th percentile of a player's rookie-season plays after play 100 (fixed for")
out("every k). The early read is the 95th percentile of his first k plays, the same best-effort statistic.")
out("Predictors: the first k plays (game only), the Combine (matched tracked piece + the 40), or both.")
out("Leave-one-class-out R^2. Takeover = smallest k where game-only beats Combine-only.")
out("Target and early read are size-free within position (weight and height removed), like Game Juice+.")
K_GRID = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100]
TARGET_FROM, MIN_TARGET = 100, 50
TAKE_SPECS = [("peak_play", "game top speed", "T_trk_plus"), ("alat_play", "game agility", "A_trk_plus"),
              ("expl_play", "game burst", "E_trk_plus")]
pl_draft = PLM2.merge(P[["nfl_id", "draft_year", "T_trk_plus", "A_trk_plus", "E_trk_plus", "neg_forty"]],
                      on="nfl_id")
pl_draft = pl_draft[pl_draft.season == pl_draft.draft_year].sort_values(["nfl_id", "game_id", "play_id"])
TAKE_ROWS = []


def size_free(df, col):
    """Residual of col on weight and height within position: the same size-free terms as Game Juice+ and the
    size-adjusted Combine pieces. Without this the size-adjusted Combine predictor is graded on a size-loaded target."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.dropna(subset=[col, "combine_weight", "combine_height"]).groupby("pgm"):
        X = np.column_stack([np.ones(len(g)), g.combine_weight, g.combine_height])
        out.loc[g.index] = g[col] - X @ np.linalg.lstsq(X, g[col], rcond=None)[0]
    return out


def loco_r2(df, xcols, ycol="target"):
    pred = pd.Series(np.nan, index=df.index)
    for yr in df.draft_year.unique():
        tr, te = df.draft_year != yr, df.draft_year == yr
        if tr.sum() < 10 or te.sum() < 3:
            continue
        X = np.column_stack([np.ones(tr.sum())] + [df.loc[tr, c] for c in xcols])
        beta = np.linalg.lstsq(X, df.loc[tr, ycol], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum())] + [df.loc[te, c] for c in xcols]) @ beta
    ok = pred.notna()
    yv = df.loc[ok, ycol]
    return 1 - ((yv - pred[ok]) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), int(ok.sum())


for mcol, mlabel, ccol in TAKE_SPECS:
    d = pl_draft.dropna(subset=[mcol])
    d = d.assign(k=d.groupby("nfl_id").cumcount())
    # Best effort, as in Game Juice+: the 95th percentile of the later plays is the target.
    tgt = d[d.k >= TARGET_FROM].groupby("nfl_id")[mcol].agg([("q95", lambda s: s.quantile(0.95)), "count"])
    tgt = tgt[tgt["count"] >= MIN_TARGET]["q95"].rename("target")
    base = P.set_index("nfl_id").loc[tgt.index, ["pgm", "draft_year", ccol, "neg_forty", "combine_weight",
                                                 "combine_height"]].join(tgt).dropna()
    if len(base) < 30:
        out(f"  {mlabel}: only {len(base)} rookies with {TARGET_FROM + MIN_TARGET}+ plays; skipped")
        continue
    # Size-free within position (this also centers it), so the models compare players inside a group.
    base["target"] = size_free(base, "target")
    for c in (ccol, "neg_forty"):
        base[c] = base.groupby("pgm")[c].transform(lambda s: (s - s.mean()) / s.std())
    r2_c, n_c = loco_r2(base, [ccol, "neg_forty"])
    for k in K_GRID:
        early = d[d.k < k].groupby("nfl_id")[mcol].quantile(0.95).rename("early")
        b2 = base.join(early).dropna()
        b2["early"] = size_free(b2, "early")
        r2_g, _ = loco_r2(b2, ["early"])
        r2_b, _ = loco_r2(b2, ["early", ccol, "neg_forty"])
        TAKE_ROWS.append({"metric": mlabel, "k": k, "r2_game_only": r2_g, "r2_combine_only": r2_c,
                          "r2_both": r2_b, "combine_adds": r2_b - r2_g, "players": len(b2)})
TAKE = pd.DataFrame(TAKE_ROWS)
save(TAKE, "snap_takeover.csv")
TAKEOVER = {}
for mlabel, d in TAKE.groupby("metric", sort=False):
    over = d[d.r2_game_only >= d.r2_combine_only]
    k_star = int(over.k.min()) if len(over) else None
    TAKEOVER[mlabel] = k_star
    out(f"\n{mlabel} (n={d.players.iloc[0]}): Combine-only R^2 {d.r2_combine_only.iloc[0]:+.3f}; "
        f"game data takes over at k = {k_star if k_star else '>' + str(K_GRID[-1])} plays")
    out(d[["k", "r2_game_only", "r2_both", "combine_adds"]].round(3).to_string(index=False))

# Unused juice: does year 2 move toward the Combine read?
hdr("F. UNUSED JUICE: DO PLAYERS MOVING BELOW THEIR COMBINE CATCH UP IN YEAR 2? (audit item 6)")
out("Season means of each play metric (50+ plays per season), size-free and standardized within group and season.")
out("Test: year-2 score on year-1 score plus the Combine piece. A positive Combine coefficient means the Combine")
out("predicts year 2 beyond year 1 (catch-up). Regression to the mean is handled by keeping year 1 in the model.")
UJ_ROWS = []
for mcol, mlabel, ccol in TAKE_SPECS:
    s = PLM2.dropna(subset=[mcol]).groupby(["nfl_id", "season"])[mcol].agg(["mean", "count"]).reset_index()
    s = s[s["count"] >= 50].merge(P[["nfl_id", "pgm", "draft_year", ccol, "combine_weight", "combine_height"]],
                                  on="nfl_id")
    # Size-free within position and season, matching the size-adjusted Combine piece.
    s["sf"] = pd.concat([size_free(g, "mean") for _, g in s.groupby("season")])
    s["z"] = s.groupby(["pgm", "season"])["sf"].transform(lambda v: (v - v.mean()) / v.std())
    y1 = s[s.season == s.draft_year].set_index("nfl_id")
    y2 = s[s.season == s.draft_year + 1].set_index("nfl_id")
    d = y1[["pgm", "z", ccol]].rename(columns={"z": "y1"}).join(y2[["z"]].rename(columns={"z": "y2"})).dropna()
    if len(d) < 25:
        out(f"  {mlabel}: {len(d)} players with year 1 and year 2; skipped")
        continue
    d["c"] = d.groupby("pgm")[ccol].transform(lambda v: (v - v.mean()) / v.std())
    d["gap"] = d.c - d.y1
    X = np.column_stack([np.ones(len(d)), d.y1, d.c])
    beta = np.linalg.lstsq(X, d.y2, rcond=None)[0]
    bs = []
    for i in rng.integers(0, len(d), (B, len(d))):
        bs.append(np.linalg.lstsq(X[i], d.y2.to_numpy()[i], rcond=None)[0][2])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    flagged = d[d.gap >= 1.0]
    UJ_ROWS.append({"metric": mlabel, "n": len(d), "beta_combine": beta[2], "ci_low": lo, "ci_high": hi,
                    "beta_year1": beta[1], "flagged_unused": len(flagged),
                    "flagged_mean_change": (flagged.y2 - flagged.y1).mean() if len(flagged) else np.nan,
                    "others_mean_change": (d.y2 - d.y1)[d.gap < 1.0].mean()})
    out(f"  {mlabel}: Combine coefficient {fmt(beta[2], lo, hi)} (year-1 coef {beta[1]:+.2f}), n={len(d)}; "
        f"{len(flagged)} flagged as unused juice")
UJ = pd.DataFrame(UJ_ROWS)
save(UJ, "unused_juice.csv")

# =====================================================================
hdr("G. SMALLER CHECKS (audit item 9)")
# =====================================================================
G9 = {}
S9 = P.dropna(subset=["juice_final"]).copy()
S9["tier"] = pd.cut(S9.juice_final, [-np.inf, 90, 110, np.inf], labels=["<90", "90-110", "110+"], right=False)
hasg = S9.game_juice.notna()
# Game Juice+ is the average of its shrunk pieces with no second rescale, so its spread is narrower than 10 points.
# "Elite in games" is therefore the top 16% within position (what +1 SD meant before), and above average is the top half.
S9["g_pct"] = S9.groupby("pgm").game_juice.rank(pct=True)
out(f"Game Juice+ SD within position (pooled): {S9.groupby('pgm').game_juice.std().mean():.1f} points")
out("Tiers (Combine Juice+) among players with Game Juice+ (elite = top 16% in games within position):")
TIER_ROWS = []
for t, d in S9[hasg].groupby("tier", observed=True):
    k_el, k_ab, n = int((d.g_pct > 0.84).sum()), int((d.g_pct > 0.5).sum()), len(d)
    lo_e, hi_e = wilson(k_el, n)
    lo_a, hi_a = wilson(k_ab, n)
    TIER_ROWS.append({"tier": t, "n": n, "elite_in_games": k_el, "elite_share": k_el / n, "elite_lo": lo_e,
                      "elite_hi": hi_e, "above_avg": k_ab, "above_share": k_ab / n, "above_lo": lo_a, "above_hi": hi_a})
    out(f"  {t:7s} n={n:3d}  elite in games {k_el}/{n} = {k_el / n:.1%} [{lo_e:.1%}, {hi_e:.1%}]   "
        f"above average {k_ab / n:.1%} [{lo_a:.1%}, {hi_a:.1%}]")
TIERS = pd.DataFrame(TIER_ROWS)
mid = S9[hasg & (S9.tier == "90-110")]
r_mid = stats.spearmanr(mid.juice_final, mid.game_juice)[0]
out(f"  Inside 90-110: Spearman {fmt(r_mid, *fisher_ci(r_mid, len(mid)))} n={len(mid)}")
# Cubic test: does a curve beat a line? (If not, tiers are just an honest way to read a linear link.)
import statsmodels.api as sm  # noqa: E402

dd = S9[hasg]
xz = (dd.juice_final - 100) / 10
lin = sm.OLS(dd.game_juice, sm.add_constant(np.column_stack([xz]))).fit()
cub = sm.OLS(dd.game_juice, sm.add_constant(np.column_stack([xz, xz ** 2, xz ** 3]))).fit()
f_p = float(cub.compare_f_test(lin)[1])
out(f"  Cubic vs straight line: F-test p = {f_p:.3f} ({'curve' if f_p < 0.05 else 'no evidence of a threshold'})")
G9["tiers_inside_90_110"] = r_mid
G9["cubic_p"] = f_p

out("\nWho reached the field (Combine-scored players):")
RF_ROWS = []
for t, d in S9.groupby("tier", observed=True):
    n, k_any, k_sc = len(d), int((d.npl_any > 0).sum()), int(d.game_juice.notna().sum())
    RF_ROWS.append({"tier": t, "n": n, "any_usable_play": k_any, "share_any": k_any / n,
                    "any_lo": wilson(k_any, n)[0], "any_hi": wilson(k_any, n)[1],
                    f"scored_{MIN_GAME_PLAYS}plus": k_sc, "share_scored": k_sc / n})
    out(f"  {t:7s} n={n:3d}  any usable snap {k_any / n:.1%}  |  {MIN_GAME_PLAYS}+ plays {k_sc / n:.1%}")
RF = pd.DataFrame(RF_ROWS)
out(f"  Mean Combine Juice+: {S9[hasg].juice_final.mean():.1f} with a game score, "
    f"{S9[~hasg].juice_final.mean():.1f} without")
rf_r = stats.spearmanr(S9.juice_final, S9.game_juice.notna())[0]
out(f"  Combine Juice+ vs reaching {MIN_GAME_PLAYS}+ plays (point-biserial, Spearman): {rf_r:+.3f}")
G9["reached_field_corr"] = rf_r
pre_f = OUT / "preseason_summary.json"
if pre_f.exists():
    pre = json.loads(pre_f.read_text())
    out(f"  Preseason: {pre['no_regular_season_but_preseason']} of {pre['no_regular_season_play']} players with no "
        f"regular-season play appear in preseason tracking ({pre['preseason_games_in_files']} preseason games in the files).")
    G9["preseason"] = pre

out("\nLate-round flag (pick 140+ or undrafted):")
L = S9[hasg & S9.late]
hi_l, lo_l = L[L.juice_final >= 110], L[L.juice_final < 110]
k1, k2 = int((hi_l.g_pct > 0.5).sum()), int((lo_l.g_pct > 0.5).sum())
out(f"  110+: {k1}/{len(hi_l)} in the top half in games; under 110: {k2}/{len(lo_l)} "
    f"({k1 / max(len(hi_l), 1):.0%} vs {k2 / max(len(lo_l), 1):.0%})")
out("  Players: " + ", ".join(f"{r.display_name} ({r.nfl_position}, {r.juice_final:.0f} -> {r.game_juice:.0f})"
                              for _, r in hi_l.sort_values("game_juice", ascending=False).iterrows()))
G9["late_round"] = {"hi_above": k1, "hi_n": len(hi_l), "lo_above": k2, "lo_n": len(lo_l)}

out("\nCarry-over by usable plays (within-group ranks, Pearson):")
for lab, (a, b_) in {"50-199": (50, 199), "200-599": (200, 599), "600+": (600, 1e9)}.items():
    d = S9[hasg & S9.npl.between(a, b_)]
    if len(d) > 10:
        r = np.corrcoef(d.groupby("pgm").juice_final.rank(pct=True), d.groupby("pgm").game_juice.rank(pct=True))[0, 1]
        out(f"  {lab:8s} n={len(d):3d}  {fmt(r, *fisher_ci(r, len(d)))}")

out("\nSnaps confound (Spearman within group, pooled ranks):")
SC = S9[hasg].copy()
for c in ["game_juice", "juice_final", "neg_forty", "g_top_plus", "g_expl_plus", "g_agil_plus", "npl", "pick"]:
    SC["X_" + c] = SC.groupby("pgm")[c].rank(pct=True)
for c in ["game_juice", "juice_final", "neg_forty", "g_top_plus", "g_expl_plus", "g_agil_plus"]:
    d = SC.dropna(subset=["X_" + c])
    out(f"  {c:12s} vs snaps {np.corrcoef(d['X_' + c], d.X_npl)[0, 1]:+.2f}   vs pick {np.corrcoef(d['X_' + c], d.X_pick)[0, 1]:+.2f}")
for c in ["juice_final", "neg_forty"]:
    d = SC.dropna(subset=["X_" + c])
    raw = np.corrcoef(d["X_" + c], d.X_game_juice)[0, 1]
    part = partial_r(d["X_" + c].to_numpy(), d.X_game_juice.to_numpy(), stack(d.X_npl.to_numpy()))
    out(f"  {c:12s} with Game Juice+: {raw:+.3f} -> {part:+.3f} holding snaps")

out("\nSize adjustment (Spearman with Game Juice+):")
for grp, d in list(S9[hasg].groupby("pgm")) + [("ALL", S9[hasg])]:
    if grp == "ALL":
        a = np.corrcoef(d.groupby("pgm").juice_final.rank(pct=True), d.groupby("pgm").game_juice.rank(pct=True))[0, 1]
        b_ = np.corrcoef(d.groupby("pgm").juice_unadjusted.rank(pct=True), d.groupby("pgm").game_juice.rank(pct=True))[0, 1]
    else:
        a, b_ = stats.spearmanr(d.juice_final, d.game_juice)[0], stats.spearmanr(d.juice_unadjusted, d.game_juice)[0]
    out(f"  {grp:5s} n={len(d):3d}  size-adjusted {a:+.2f}   unadjusted {b_:+.2f}")

save(TIERS, "tiers.csv")
save(RF, "reached_field.csv")

# =====================================================================
hdr("H. PRIMARY TESTS (from PRIMARY_TESTS.md)")
# =====================================================================


def get_add(piece, group, holding="the 40", target="Game Juice+"):
    r = ADDS[(ADDS.piece == PIECES[piece][0]) & (ADDS.group == group) & (ADDS.holding == holding)
             & (ADDS.target == target)]
    return r.iloc[0] if len(r) else None


PRIMARY = []
a1 = get_add("A", "ALL", "40 + RAS-style")
if a1 is not None:
    PRIMARY.append(("P1", "Tracked agility adds to Game Juice+ beyond the 40 and RAS-style (pooled)",
                    a1.partial_r, a1.ci_low, a1.ci_high, int(a1.n)))
PRIMARY.append(("P2", "Tracked burst (yard 2-10) vs NFL get-off, DL + EDGE", r_b, lo_b, hi_b, int(len(rush_d))))
wr = MM[(MM.group == "WR") & (MM.outcome.str.startswith("Separation")) & (MM.trait == "A_trk_plus")]
if len(wr):
    w = wr.iloc[0]
    PRIMARY.append(("P3", "WR: tracked agility -> separation at throw (mixed model transfer)",
                    w.transfer, w.transfer_lo, w.transfer_hi, int(w.players)))
ts = TAKE[(TAKE.metric == "game top speed")]
if len(ts):
    PRIMARY.append(("P4", f"Snap takeover for top speed: k* = {TAKEOVER.get('game top speed')}",
                    np.nan, np.nan, np.nan, int(ts.players.iloc[0])))
if ASP is not None:
    lk = read_out("asp_combine_vs_game.csv", RUN_ID)
    row = lk[lk.group == "ALL (within-group ranks)"]
    if len(row) and "v0:combine_all->game" in row:
        r = float(row["v0:combine_all->game"].iloc[0])
        n = int(row["n_v0_combine_all"].iloc[0])
        PRIMARY.append(("P5", "A-S top gear v0: Combine vs games (pooled ranks)", r, *fisher_ci(r, n), n))
PRIM = pd.DataFrame(PRIMARY, columns=["test", "claim", "estimate", "ci_low", "ci_high", "n"])
PRIM["verdict"] = np.where(PRIM.ci_low > 0, "supported", np.where(PRIM.ci_high < 0, "reversed",
                           np.where(PRIM.estimate.isna(), "see table", "not shown")))
out(PRIM.round(3).to_string(index=False))
save(PRIM, "primary_tests.csv")

# Player and play-level outcome tables, for juice_predictive.py (same definitions, same run).
save(P.drop(columns=[c for c in P.columns if c.startswith("X_")]), "player_outcomes.csv")
OUT_PLAY_COLS = ["sep_c", "yac_oe", "rec_yds_route", "get_off_c", "pressure", "ttp", "ppp_held_c", "ttp_allowed",
                 "held", "cov_epa", "expected_points_added"]
save(pp[["game_id", "play_id", "nfl_id", "pgm", "season"] + OUT_PLAY_COLS], "play_level_outcomes.csv")

SUMMARY = {"run_id": RUN_ID, "decision": DECISION, "takeover_k": TAKEOVER, "smaller_checks": G9}
(OUT / "transfer_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
(OUT / "transfer_report.txt").write_text("\n".join(LINES))
print(f"\nSaved transfer_report.txt and tables to {OUT.resolve()} (run {RUN_ID})")
