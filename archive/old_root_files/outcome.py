import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Folders: the competition data and the output folder of juice.py.
COMP = "nfl-big-data-bowl-2027"
OUT = Path(os.getenv("JUICE_OUT_DIR", "juice_full_output"))
root = os.getenv("JUICE_DATA_DIR")
if not root:
    import kagglehub
    root = kagglehub.competition_download(COMP)
root = Path(root)
if (root / COMP).is_dir():
    root = root / COMP

B = 1000
rng = np.random.default_rng(0)
lines = []


def out(text=""):
    lines.append(str(text))
    print(text)


# Small tables: Juice+ outputs, Combine results, player info, career outcomes, games.
J = pd.read_csv(OUT / "juice_player_table.csv").drop(columns=["game_juice_all"], errors="ignore")
J_NEED = ["juice_tracked_only", "E_trk_plus", "T_trk_plus", "A_trk_plus"]
if any(c not in J.columns for c in J_NEED):
    raise SystemExit("juice_player_table.csv is from an older run. Rerun the new juice.py first.")
G = pd.read_csv(OUT / "game_juice_scores_all.csv")[["nfl_id", "game_juice", "g_top_plus", "g_expl_plus"]]
CR = pd.read_csv(root / "combine_results.csv")[
    ["nfl_id", "combine_weight", "forty", "ten_yd_split", "vertical", "broad_jump", "three_cone",
     "short_shuttle", "ngs_athleticism_score"]]
PL = pd.read_csv(root / "players.csv")[["nfl_id", "draft_overall_pick"]]
CS = pd.read_csv(root / "player_career_successes.csv")
games = pd.read_csv(root / "games.csv")[["game_id", "season", "season_type"]]
D = (J.merge(G, on="nfl_id", how="left").merge(CR, on="nfl_id", how="left")
      .merge(PL, on="nfl_id", how="left").merge(CS, on="nfl_id", how="left"))

# Data check 1: are speeds in each season stored with decimals? (Whole numbers only would mean rounding.)
out("DATA CHECK. Share of game-tracking speed values that are whole numbers (first 500k rows):")
for yr in (2023, 2024, 2025):
    s = pd.read_csv(root / f"game_tracking_{yr}.csv", usecols=["s"], nrows=500000).s
    out(f"  {yr}: {np.mean(s % 1 == 0):.3f}")

# Career outcome: share of available games started (17 per season since the draft).
D["seasons"] = 2026 - D.draft_year
D["start_share"] = D.career_games_started / (17 * D.seasons)
D["honors"] = (D.ap_all_pro_1st_team + D.ap_all_pro_2nd_team + D.pro_bowl_original_ballot) > 0

# Play-level outcomes from player_play, regular season only, penalty-nullified plays removed.
# The merge with D keeps only players with Combine tracking, so positions match the Combine data.
PP_COLS = ["game_id", "nfl_id", "team_abbr", "expected_points_added", "play_nullified_by_penalty", "target",
           "pass_result", "coverage_assignment", "player_get_off", "pressure_allowed", "sack",
           "tackle_for_loss", "time_to_pressure", "time_to_qb_hurry"]
pp = pd.read_csv(root / "player_play.csv", usecols=PP_COLS)
pp = pp.merge(D[["nfl_id", "pgm"]], on="nfl_id").merge(games, on="game_id")
pp = pp[(pp.season_type == "REG") & (pp.play_nullified_by_penalty.astype(str) != "Y")].copy()


def tf(s):
    """True/False, 1.0/0.0 or 'True'/'False' to 1.0/0.0; anything else to NaN."""
    return s.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1.0": 1.0, "0.0": 0.0, "1": 1.0, "0": 0.0})


# EPA is the offense's EPA, so it is flipped for defenders. Team codes are used as given.
defense = pp.pgm.isin(["DL", "EDGE", "DB"])
pp["epa"] = pp.expected_points_added.where(~defense, -pp.expected_points_added)
pp["side"] = np.where(defense, "D", "O")
pp["team"] = pp.team_abbr
pp["epa_adj"] = pp.epa - pp.groupby(["team", "season", "side"]).epa.transform("mean")
rush = pp.pgm.isin(["DL", "EDGE"]) & pp.player_get_off.notna() & pp.pass_result.notna()
cover = ((pp.pgm == "DB") & pp.coverage_assignment.notna() & (pp.coverage_assignment != "NO_ASSIGNMENT")
         & pp.pass_result.notna())
pass_pro = (pp.pgm == "OL") & pp.pressure_allowed.notna()
targeted = pp.pgm.isin(["WR", "TE"]) & (tf(pp.target) == 1)
pp["pressure"] = pp.time_to_pressure.notna().astype(float)
pp["disrupt"] = ((tf(pp.sack) == 1) | (tf(pp.tackle_for_loss) == 1)
                 | pp.time_to_pressure.notna() | pp.time_to_qb_hurry.notna()).astype(float)
pp["fast_get_off"] = -pp.player_get_off
pp["pressure_held"] = 1 - tf(pp.pressure_allowed)


def per_player(mask, col, min_n):
    s = pp.loc[mask].groupby("nfl_id")[col].agg(["mean", "count"])
    return s["mean"].where(s["count"] >= min_n)


everyone = pp.epa.notna()
OC = pd.DataFrame({
    "epa_field": per_player(everyone, "epa", 200),
    "epa_field_adj": per_player(everyone, "epa_adj", 200),
    "epa_target": per_player(targeted, "epa", 25),
    "epa_coverage": per_player(cover, "epa", 100),
    "pressure_rate": per_player(rush, "pressure", 100),
    "disrupt_rate": per_player(rush, "disrupt", 100),
    "pressure_held": per_player(pass_pro, "pressure_held", 100),
    "get_off_speed": per_player(rush, "fast_get_off", 100),
})
D = D.join(OC, on="nfl_id")

# Data check 2: does each outcome look sensible? (rates and counts so definitions can be eyeballed)
out("\nDATA CHECK. player_play rows kept: " + str(len(pp)) + "; rows per player: "
    + str(pp.groupby("nfl_id").size().describe().round(0).to_dict()))
out(f"  pass-rush pressure rate (DL/EDGE): {pp.loc[rush, 'pressure'].mean():.3f}"
    f"   disruption rate: {pp.loc[rush, 'disrupt'].mean():.3f}"
    f"   OL pressure-held rate: {pp.loc[pass_pro, 'pressure_held'].mean():.3f}"
    f"   mean EPA on targeted plays (WR/TE): {pp.loc[targeted, 'epa'].mean():+.3f}")
out("  players with each outcome: " + str(OC.notna().sum().to_dict()))
out(f"  mean get-off time on pass-rush snaps (seconds, lower is faster): {pp.loc[rush, 'player_get_off'].mean():.3f}")

# Draft pick, best pick = highest percentile within position group. A pick of 0 or missing = undrafted.
D["pick"] = D.draft_overall_pick.where(D.draft_overall_pick > 0, 260)
D["pick_p"] = D.groupby("pgm").pick.rank(pct=True, ascending=False)

# Stopwatch comparison scores.
D["neg_forty"] = -D.forty
D["speed_score"] = D.combine_weight * 200 / D.forty ** 4


SCORES = [
    ("Juice+ (as built)", "juice_final"),
    ("Juice+ tracked-only", "juice_tracked_only"),
    ("E piece (as built)", "E_plus"),
    ("E piece tracked-only", "E_trk_plus"),
    ("T piece (as built)", "T_plus"),
    ("T piece tracked-only", "T_trk_plus"),
    ("A piece (as built)", "A_plus"),
    ("A piece tracked-only", "A_trk_plus"),
    ("NGS athleticism score", "ngs_athleticism_score"),
    ("Official 40 (faster = better)", "neg_forty"),
    ("Speed score (weight x 40)", "speed_score"),
    ("Draft pick", "pick_p"),
    ("Game Juice+ (same players)", "game_juice"),
]
GAME_COLS = {"game_juice", "g_top_plus", "g_expl_plus"}
OUTCOMES = [
    ("Career: share of games started (within class)", "start_share"),
    ("EPA on field (good for player's side)", "epa_field"),
    ("EPA on field, team-season adjusted", "epa_field_adj"),
    ("WR/TE: EPA per target", "epa_target"),
    ("DB: EPA allowed in coverage (flipped)", "epa_coverage"),
    ("DL/EDGE: pressure rate on pass rush", "pressure_rate"),
    ("DL/EDGE: disruption rate on pass rush", "disrupt_rate"),
    ("OL: pressure held rate", "pressure_held"),
    ("DL/EDGE: NFL get-off speed (flipped seconds)", "get_off_speed"),
    ("Game Juice+", "game_juice"),
    ("Game explosiveness", "g_expl_plus"),
    ("Game top speed", "g_top_plus"),
]

# Everything is compared as within-position percentiles, so groups with different scales mix fairly.
for _, c in SCORES + OUTCOMES:
    D["X_" + c] = D.groupby("pgm")[c].rank(pct=True)
D["X_start_share"] = D.groupby(["pgm", "draft_year"]).start_share.rank(pct=True)


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def resid(v, p):
    X = np.c_[np.ones(len(p)), p]
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def ci(vals):
    return np.nanpercentile(vals, [2.5, 97.5])


def table_row(xc, yc, with_partial):
    m = D[["X_" + xc, "X_" + yc, "pick_p"]].notna().all(axis=1)
    x, y, p = (D.loc[m, "X_" + xc].to_numpy(), D.loc[m, "X_" + yc].to_numpy(), D.loc[m, "pick_p"].to_numpy())
    if len(x) < 30:
        return None
    idx = rng.integers(0, len(x), (B, len(x)))
    r = corr(x, y)
    lo, hi = ci([corr(x[i], y[i]) for i in idx])
    res = {"n": len(x), "rho": r, "lo": lo, "hi": hi}
    if with_partial:
        res["partial"] = corr(resid(x, p), resid(y, p))
        res["plo"], res["phi"] = ci([corr(resid(x[i], p[i]), resid(y[i], p[i])) for i in idx])
    return res


def edge(a, b, yc):
    m = D[["X_" + a, "X_" + b, "X_" + yc]].notna().all(axis=1)
    xa, xb, y = (D.loc[m, "X_" + a].to_numpy(), D.loc[m, "X_" + b].to_numpy(), D.loc[m, "X_" + yc].to_numpy())
    if len(y) < 30:
        return None
    idx = rng.integers(0, len(y), (B, len(y)))
    d = corr(xa, y) - corr(xb, y)
    return d, *ci([corr(xa[i], y[i]) - corr(xb[i], y[i]) for i in idx]), len(y)


EDGES = [("juice_final", "ngs_athleticism_score", "Juice+ minus NGS athleticism"),
         ("juice_final", "neg_forty", "Juice+ minus official 40"),
         ("juice_final", "speed_score", "Juice+ minus speed score"),
         ("juice_final", "juice_tracked_only", "Juice+ as built minus tracked-only")]
rows = []
for olabel, oc in OUTCOMES:
    d = D[D["X_" + oc].notna()]
    out(f"\n==== {olabel}  (n = {len(d)}; groups: {d.pgm.value_counts().to_dict()})")
    out(f"{'score':38s} {'n':>4s}  corr [95% CI]               | after removing draft pick [95% CI]")
    is_game_outcome = oc in GAME_COLS
    for slabel, sc in SCORES:
        if sc == oc or (is_game_outcome and sc in GAME_COLS):
            continue
        res = table_row(sc, oc, with_partial=(sc != "pick_p"))
        if res is None:
            continue
        part = (f"{res['partial']:+.3f} [{res['plo']:+.3f}, {res['phi']:+.3f}]" if "partial" in res else "n/a")
        out(f"{slabel:38s} {res['n']:4d}  {res['rho']:+.3f} [{res['lo']:+.3f}, {res['hi']:+.3f}]   | {part}")
        rows.append({"outcome": olabel, "score": slabel, **res})
    for a, b, label in EDGES:
        e = edge(a, b, oc)
        if e:
            out(f"  {label}: {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] on the same {e[3]} players")

pd.DataFrame(rows).to_csv("outcome_test_table.csv", index=False)

out("\nPlayers with All-Pro or Pro Bowl votes (Combine Juice+ and pick):")
out(D[D.honors][["display_name", "nfl_position", "draft_year", "pick", "juice_final"]]
    .sort_values("juice_final", ascending=False).round(1).to_string(index=False))
out("\nHow to read this: an interval that includes 0 means no clear link. A gap between two scores only"
    " matters if the edge interval excludes 0. Juice+ was built and tuned on the same players as the Game"
    " rows, so any gain there is exploratory. The EPA, coverage, pressure and get-off rows are new evidence.")
Path("outcome_test_report.txt").write_text("\n".join(lines))
out("\nSaved: outcome_test_report.txt and outcome_test_table.csv")