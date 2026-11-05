"""What can Juice+ predict? Run after juice_transfer.py, on the same run ID. All four tests are EXPLORATORY.

  1. Snaps: Juice+, NGS, RAS-style and the 40 against career scrimmage snaps, games started and reaching the field.
     Raw, then holding draft pick fixed. Raw links are partly the draft: teams pick athletic players early and play
     high picks. Only the link that survives holding pick is evidence the score sees something the draft missed.
  2. Reliability ceiling of each outcome: how repeatable is it within a player? A score cannot correlate with an
     outcome more than about sqrt(reliability), so a low ceiling means the outcome is the wrong yardstick.
  3. Beyond draft slot: the same scores against every outcome, holding pick fixed.
  4. Each position's best stat, holding the 40 and pick fixed: what does Juice+ add beyond the stopwatch and the draft?

Every comparison uses within-position (and, for career totals, within-class) percentile ranks, pooled with group
dummies, and 1,000-resample bootstrap intervals over players.
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
from juice_common import OUT, get_file, load_run_id, read_out, save_with_run, section_rng  # noqa: E402

RUN_ID = load_run_id()
B = 1000
rng = np.random.default_rng(7)
LINES = []
GROUPS = ["DB", "DL", "EDGE", "OL", "TE", "WR"]


def out(s=""):
    LINES.append(str(s))
    print(s)


def hdr(s):
    """Section header. Also reseeds the shared generator from the section label, so adding or reordering
    sections never moves another section's bootstrap draws."""
    global rng
    rng = section_rng("predictive", s.split()[0])
    out("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def residualize(v, C):
    if C is None or C.shape[1] == 0:
        return v - v.mean()
    X = np.column_stack([np.ones(len(v)), C])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def pr(x, y, C=None):
    rx, ry = residualize(x, C), residualize(y, C)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan


def dummies(g):
    d = pd.get_dummies(pd.Series(g).astype(str), drop_first=True).astype(float).to_numpy()
    return d if d.shape[1] else None


def stack(*cols):
    cols = [c for c in cols if c is not None]
    return np.column_stack([c if np.ndim(c) == 2 else np.asarray(c)[:, None] for c in cols]) if cols else None


def fmt(r, lo, hi):
    return f"{r:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ---------------------------------------------------------------------
# Data: one row per Combine-scored player, with career totals
# ---------------------------------------------------------------------
P = read_out("player_outcomes.csv", RUN_ID)
PLO = read_out("play_level_outcomes.csv", RUN_ID)
CS = pd.read_csv(get_file("player_career_successes.csv"))
P = P.merge(CS, on="nfl_id", how="left")
for c in ["career_offensive_snaps", "career_defensive_snaps", "career_games_started", "career_games_active"]:
    P[c] = P[c].fillna(0)
P["career_snaps"] = P.career_offensive_snaps + P.career_defensive_snaps
P["seasons"] = (2026 - P.draft_year).clip(lower=1)
P["start_share"] = P.career_games_started / (17 * P.seasons)
P["reached_field"] = (P.game_juice.notna()).astype(float)        # 50+ usable plays in the tracking data
P["tracked_plays"] = P.npl_any
P["neg_pick"] = -P.pick                                           # earlier pick = higher
SCORES = [("Juice+", "juice_final"), ("Juice+ tracked-only", "juice_tracked_only"),
          ("NGS athleticism", "ngs_athleticism_score"), ("RAS-style", "ras_style"), ("Official 40", "neg_forty")]
CAREER = [("Career scrimmage snaps", "career_snaps"), ("Share of games started", "start_share"),
          ("Usable plays in tracking", "tracked_plays"), ("Reached 50+ tracked plays", "reached_field")]
out(f"Run {RUN_ID}. Combine-scored players: {len(P)}")


def rank_cols(d, cols, by):
    for c in cols:
        d["X_" + c] = d.groupby(by)[c].rank(pct=True)
    return d


def assoc(d, x, y, controls=(), by_y=("pgm",), reps=B):
    """Partial correlation of within-group ranks of x and y, holding control ranks and group dummies."""
    d = d.dropna(subset=[x, y] + list(controls)).copy()
    if len(d) < 30:
        return None
    d = rank_cols(d, [x] + list(controls), "pgm")
    d["Y"] = d.groupby(list(by_y))[y].rank(pct=True)
    grp = d[list(by_y)].astype(str).agg("|".join, axis=1)
    xv, yv = d["X_" + x].to_numpy(), d.Y.to_numpy()
    C = stack(*[d["X_" + c].to_numpy() for c in controls], dummies(grp))
    r = pr(xv, yv, C)
    idx = rng.integers(0, len(d), (reps, len(d)))
    bs = [pr(xv[i], yv[i], None if C is None else C[i]) for i in idx]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, lo, hi, len(d)


# =====================================================================
hdr("1. SCORES VS SNAPS AND PLAYING TIME (career totals ranked within position and draft class)")
# =====================================================================
out("Raw = no controls. 'Holding pick' = the part of the score the draft did not already price in.")
out("Same players in every row of a block (players with all five scores).")
common = P.dropna(subset=[c for _, c in SCORES]).copy()
SNAP_ROWS = []
for olabel, ocol in CAREER:
    out(f"\n{olabel}  (n={len(common)})")
    by_y = ("pgm", "draft_year") if ocol in ("career_snaps", "start_share", "tracked_plays") else ("pgm",)
    for slabel, scol in SCORES + [("Draft pick (earlier = higher)", "neg_pick")]:
        raw = assoc(common, scol, ocol, by_y=by_y)
        held = assoc(common, scol, ocol, controls=["neg_pick"], by_y=by_y) if scol != "neg_pick" else None
        SNAP_ROWS.append({"outcome": olabel, "score": slabel, "raw": raw[0], "raw_lo": raw[1], "raw_hi": raw[2],
                          "holding_pick": held[0] if held else np.nan, "hp_lo": held[1] if held else np.nan,
                          "hp_hi": held[2] if held else np.nan, "n": raw[3]})
        out(f"  {slabel:30s} raw {fmt(*raw[:3])}   holding pick "
            + (fmt(*held[:3]) if held else "   n/a"))
SNAPS = pd.DataFrame(SNAP_ROWS)
save_with_run(SNAPS, "snaps_test.csv", RUN_ID)

# Paired edges on the main snap outcome, holding pick: Juice+ minus each other score.
out("\nJuice+ minus each score, career snaps holding pick (paired bootstrap, same players):")
d = common.copy()
d = rank_cols(d, [c for _, c in SCORES] + ["neg_pick"], "pgm")
d["Y"] = d.groupby(["pgm", "draft_year"]).career_snaps.rank(pct=True)
C0 = stack(d.X_neg_pick.to_numpy(), dummies(d.pgm.astype(str) + "|" + d.draft_year.astype(str)))
yv = d.Y.to_numpy()
idx = rng.integers(0, len(d), (B, len(d)))
EDGE_ROWS = []
for slabel, scol in SCORES[2:]:
    xa, xb = d.X_juice_final.to_numpy(), d["X_" + scol].to_numpy()
    est = pr(xa, yv, C0) - pr(xb, yv, C0)
    bs = [pr(xa[i], yv[i], C0[i]) - pr(xb[i], yv[i], C0[i]) for i in idx]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    verdict = "clear edge" if lo > 0 else ("behind" if hi < 0 else "no clear difference")
    EDGE_ROWS.append({"juice_minus": slabel, "edge": est, "lo": lo, "hi": hi, "verdict": verdict})
    out(f"  Juice+ minus {slabel:16s} {fmt(est, lo, hi)}  {verdict}")
save_with_run(pd.DataFrame(EDGE_ROWS), "snaps_edges.csv", RUN_ID)

out("\nBy position, career snaps holding pick (Spearman partial, within class):")
POS_ROWS = []
for grp in GROUPS:
    g = common[common.pgm == grp]
    row = {"group": grp, "n": len(g)}
    for slabel, scol in SCORES:
        res = assoc(g, scol, "career_snaps", controls=["neg_pick"], by_y=("pgm", "draft_year"), reps=300)
        row[slabel] = res[0] if res else np.nan
    POS_ROWS.append(row)
POS = pd.DataFrame(POS_ROWS)
out(POS.round(2).to_string(index=False))
save_with_run(POS, "snaps_by_position.csv", RUN_ID)

# =====================================================================
hdr("2. RELIABILITY CEILING OF EACH OUTCOME (split-half within player, Spearman-Brown corrected)")
# =====================================================================
out("Each player's plays are split at random into two halves; the outcome mean in one half is correlated with the")
out("other across players (30 splits, players with 2x the floor). Ceiling = sqrt(reliability) = the most any score")
out("could correlate with the outcome at the typical play count.")
REL_SPECS = [("WR", "sep_c", "Separation at throw", 30), ("WR", "rec_yds_route", "Yards per route", 50),
             ("WR", "yac_oe", "YAC over expected", 10), ("TE", "sep_c", "Separation at throw", 30),
             ("TE", "rec_yds_route", "Yards per route", 50), ("DL", "get_off_c", "Get-off", 30),
             ("DL", "pressure", "Pressure rate", 30), ("EDGE", "get_off_c", "Get-off", 30),
             ("EDGE", "pressure", "Pressure rate", 30), ("OL", "ppp_held_c", "Peak pressure prob. (flipped)", 30),
             ("OL", "held", "Pressure held rate", 30), ("DB", "cov_epa", "Coverage EPA (flipped)", 50)]
for grp in GROUPS:
    REL_SPECS.append((grp, "epa_side", "EPA on field (player's side)", 200))
defense = PLO.pgm.isin(["DL", "EDGE", "DB"])
PLO["epa_side"] = PLO.expected_points_added.where(~defense, -PLO.expected_points_added)


def split_half(df, col, floor, reps=30):
    d = df[["nfl_id", col]].dropna()
    cnt = d.groupby("nfl_id").size()
    d = d[d.nfl_id.isin(cnt[cnt >= 2 * floor].index)].reset_index(drop=True)
    if d.nfl_id.nunique() < 15:
        return np.nan, d.nfl_id.nunique(), np.nan
    vals = []
    for _ in range(reps):
        d["h"] = rng.integers(0, 2, len(d))
        m = d.groupby(["nfl_id", "h"])[col].mean().unstack().dropna()
        r = stats.spearmanr(m[0], m[1])[0]
        if np.isfinite(r) and r > -0.99:
            vals.append(2 * r / (1 + r))
    return float(np.mean(vals)), d.nfl_id.nunique(), float(cnt[cnt >= 2 * floor].median())


REL_ROWS = []
for grp, col, label, floor in REL_SPECS:
    rel, n, med = split_half(PLO[PLO.pgm == grp], col, floor)
    ceil = np.sqrt(max(rel, 0)) if np.isfinite(rel) else np.nan
    REL_ROWS.append({"group": grp, "outcome": label, "reliability": rel, "ceiling": ceil, "players": n,
                     "median_plays": med})
REL = pd.DataFrame(REL_ROWS)
out(REL.round(2).to_string(index=False))
save_with_run(REL, "outcome_reliability.csv", RUN_ID)
# Game pieces for comparison (from juice.py's manifest).
man = json.loads((OUT / "run_manifest.json").read_text())
out("For comparison, Game Juice+ pieces (split-half, well-sampled players): "
    + ", ".join(f"{k} {v:.2f}" for k, v in man["reliability"]["game_split_half"].items()))

# =====================================================================
hdr("3. BEYOND DRAFT SLOT: EVERY OUTCOME, HOLDING PICK")
# =====================================================================
OUTS = {"WR": [("o_sep_c", "Separation at throw"), ("o_rec_yds_route", "Yards per route"), ("o_yac_oe", "YAC over expected")],
        "TE": [("o_sep_c", "Separation at throw"), ("o_rec_yds_route", "Yards per route")],
        "DL": [("o_get_off_c", "Get-off"), ("o_pressure", "Pressure rate")],
        "EDGE": [("o_get_off_c", "Get-off"), ("o_pressure", "Pressure rate")],
        "OL": [("o_ppp_held_c", "Peak pressure prob. (flipped)"), ("o_held", "Pressure held rate")],
        "DB": [("o_cov_epa", "Coverage EPA (flipped)")]}
BEYOND = []
for grp, outs in OUTS.items():
    g = P[P.pgm == grp]
    for ocol, olabel in outs:
        out(f"\n{grp}: {olabel}")
        for slabel, scol in SCORES:
            res = assoc(g, scol, ocol, controls=["neg_pick"], reps=500)
            if res:
                BEYOND.append({"group": grp, "outcome": olabel, "score": slabel, "partial_r": res[0],
                               "lo": res[1], "hi": res[2], "n": res[3]})
                out(f"  {slabel:22s} {fmt(*res[:3])}  n={res[3]}")
save_with_run(pd.DataFrame(BEYOND), "beyond_draft_slot.csv", RUN_ID)

# =====================================================================
hdr("4. EACH POSITION'S BEST STAT, HOLDING THE 40 AND PICK")
# =====================================================================
out("What the tracked pieces and Juice+ add beyond the stopwatch and the draft, on the stat that fits each position.")
BEST = [("WR", "o_rec_yds_route", "Yards per route"), ("WR", "o_sep_c", "Separation at throw"),
        ("TE", "o_rec_yds_route", "Yards per route"), ("EDGE", "o_pressure", "Pressure rate"),
        ("DL", "o_pressure", "Pressure rate"), ("DL", "o_get_off_c", "Get-off"),
        ("EDGE", "o_get_off_c", "Get-off"), ("OL", "o_ppp_held_c", "Peak pressure prob. (flipped)"),
        ("DB", "o_cov_epa", "Coverage EPA (flipped)")]
PIECE_SCORES = [("Juice+", "juice_final"), ("Burst (tracked)", "E_trk_plus"), ("Speed (tracked)", "T_trk_plus"),
                ("Agility (tracked)", "A_trk_plus"), ("Game Juice+ (link 2)", "game_juice")]
BEST_ROWS = []
for grp, ocol, olabel in BEST:
    g = P[P.pgm == grp]
    out(f"\n{grp}: {olabel}")
    for slabel, scol in PIECE_SCORES:
        controls = ["neg_pick"] + (["neg_forty"] if scol != "neg_forty" else [])
        res = assoc(g, scol, ocol, controls=controls, reps=500)
        if res:
            BEST_ROWS.append({"group": grp, "outcome": olabel, "score": slabel, "partial_r": res[0], "lo": res[1],
                              "hi": res[2], "n": res[3]})
            out(f"  {slabel:22s} {fmt(*res[:3])}  n={res[3]}")
save_with_run(pd.DataFrame(BEST_ROWS), "best_stat_by_position.csv", RUN_ID)

# =====================================================================
hdr("5. SNAPS: RAW EDGES, AND LATE-ROUND PLAYERS ONLY (round 4+ and undrafted)")
# =====================================================================


def paired_edges(d, ocol, controls, by_y, label):
    d = d.dropna(subset=[c for _, c in SCORES] + [ocol] + list(controls)).copy()
    d = rank_cols(d, [c for _, c in SCORES] + list(controls), "pgm")
    d["Y"] = d.groupby(list(by_y))[ocol].rank(pct=True)
    grp = d[list(by_y)].astype(str).agg("|".join, axis=1)
    C = stack(*[d["X_" + c].to_numpy() for c in controls], dummies(grp))
    yv = d.Y.to_numpy()
    idx = rng.integers(0, len(d), (B, len(d)))
    rows = []
    for slabel, scol in SCORES[2:]:
        xa, xb = d.X_juice_final.to_numpy(), d["X_" + scol].to_numpy()
        est = pr(xa, yv, C) - pr(xb, yv, C)
        bs = [pr(xa[i], yv[i], None if C is None else C[i]) - pr(xb[i], yv[i], None if C is None else C[i]) for i in idx]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        verdict = "clear edge" if lo > 0 else ("behind" if hi < 0 else "no clear difference")
        rows.append({"sample": label, "outcome": ocol, "controls": "+".join(controls) or "none",
                     "juice_minus": slabel, "edge": est, "lo": lo, "hi": hi, "verdict": verdict, "n": len(d)})
        out(f"  Juice+ minus {slabel:16s} {fmt(est, lo, hi)}  {verdict}  (n={len(d)})")
    return rows


E5 = []
out("All players, career snaps, NO pick control (paired bootstrap):")
E5 += paired_edges(common, "career_snaps", [], ("pgm", "draft_year"), "all")

late = P[(P.draft_round.fillna(99) >= 4)].copy()
late_c = late.dropna(subset=[c for _, c in SCORES]).copy()
out(f"\nLate-round players (round 4+ and undrafted): {len(late)}; with all five scores: {len(late_c)}")
out(f"  rounds: {late.draft_round.fillna(0).astype(int).replace(0, 'UDFA').value_counts().to_dict()}")
LATE_ROWS = []
for olabel, ocol in CAREER:
    by_y = ("pgm", "draft_year") if ocol in ("career_snaps", "start_share", "tracked_plays") else ("pgm",)
    out(f"\n{olabel}  (late-round, n={len(late_c)})")
    for slabel, scol in SCORES + [("Draft pick (earlier = higher)", "neg_pick")]:
        raw = assoc(late_c, scol, ocol, by_y=by_y)
        held = assoc(late_c, scol, ocol, controls=["neg_pick"], by_y=by_y) if scol != "neg_pick" else None
        if raw is None:
            continue
        LATE_ROWS.append({"outcome": olabel, "score": slabel, "raw": raw[0], "raw_lo": raw[1], "raw_hi": raw[2],
                          "holding_pick": held[0] if held else np.nan, "hp_lo": held[1] if held else np.nan,
                          "hp_hi": held[2] if held else np.nan, "n": raw[3]})
        out(f"  {slabel:30s} raw {fmt(*raw[:3])}   holding pick " + (fmt(*held[:3]) if held else "   n/a"))
save_with_run(pd.DataFrame(LATE_ROWS), "snaps_late_round.csv", RUN_ID)
out("\nLate-round, career snaps, Juice+ minus each score:")
out(" no pick control:")
E5 += paired_edges(late_c, "career_snaps", [], ("pgm", "draft_year"), "late")
out(" holding pick:")
E5 += paired_edges(late_c, "career_snaps", ["neg_pick"], ("pgm", "draft_year"), "late")
save_with_run(pd.DataFrame(E5), "snaps_edges_raw_and_late.csv", RUN_ID)

out("\nLate-round, share in the top third of career snaps (within position and class) by Juice+ tier:")
lt = late.copy()
lt["snap_pct"] = lt.groupby(["pgm", "draft_year"]).career_snaps.rank(pct=True)
lt["tier"] = pd.cut(lt.juice_final, [-np.inf, 90, 110, np.inf], labels=["<90", "90-110", "110+"], right=False)
for t_, g in lt.dropna(subset=["tier"]).groupby("tier", observed=True):
    k = int((g.snap_pct > 2 / 3).sum())
    out(f"  {t_:7s} {k}/{len(g)} = {k / len(g):.0%}")
for slabel, scol in [("NGS", "ngs_athleticism_score"), ("RAS-style", "ras_style")]:
    q = lt.dropna(subset=[scol]).copy()
    q["pct"] = q.groupby("pgm")[scol].rank(pct=True)
    top = q[q.pct >= q.pct.quantile(0.85)]
    out(f"  For comparison, top 15% by {slabel}: {int((top.snap_pct > 2 / 3).sum())}/{len(top)} "
        f"= {(top.snap_pct > 2 / 3).mean():.0%}")
top_j = lt.dropna(subset=["juice_final"])
top_j = top_j[top_j.groupby("pgm").juice_final.rank(pct=True) >= 0.85]
out(f"  Top 15% by Juice+ (same rule): {int((top_j.snap_pct > 2 / 3).sum())}/{len(top_j)} = {(top_j.snap_pct > 2 / 3).mean():.0%}")

# =====================================================================
hdr("6. FOOTBALL RESULTS, RAW: EVERY ATHLETICISM SCORE, SAME PLAYERS PER ROW")
# =====================================================================
out("Spearman within position, no controls, on the players who have every score in that block. 95% Fisher interval.")
out("Pooled rows combine all blocks on within-group ranks; the paired bootstrap tests Juice+ minus each score.")
RAW_SCORES = [("Juice+", "juice_final"), ("NGS", "ngs_athleticism_score"), ("RAS-style", "ras_style"),
              ("40", "neg_forty"), ("Pick", "neg_pick")]


def fisher(r, n):
    if not np.isfinite(r) or n <= 3:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


RAW_ROWS, POOL = [], []
for grp, outs in OUTS.items():
    for ocol, olabel in outs:
        cols = [c for _, c in RAW_SCORES if c != "ras_style" or P.loc[P.pgm == grp, "ras_style"].notna().sum() >= 25]
        g = P[(P.pgm == grp)].dropna(subset=cols + [ocol])
        if len(g) < 20:
            out(f"  {grp} {olabel}: only {len(g)} players with every score; skipped")
            continue
        line = f"  {grp:4s} {olabel[:30]:30s} n={len(g):3d} "
        for slabel, scol in RAW_SCORES:
            if scol not in cols:
                line += f"| {slabel} n/a        "
                continue
            r = stats.spearmanr(g[scol], g[ocol])[0]
            lo, hi = fisher(r, len(g))
            RAW_ROWS.append({"group": grp, "outcome": olabel, "score": slabel, "rho": r, "lo": lo, "hi": hi, "n": len(g)})
            line += f"| {slabel} {r:+.2f} "
        out(line)
        q = g[["pgm"] + cols + [ocol]].copy()
        q["block"] = grp + "|" + olabel
        for c in cols + [ocol]:
            q["X_" + c] = q[c].rank(pct=True)
        q["Y"] = q["X_" + ocol]
        POOL.append(q)
RAW = pd.DataFrame(RAW_ROWS)
save_with_run(RAW, "football_results_raw.csv", RUN_ID)

wins = RAW[RAW.score != "Pick"].loc[lambda d: d.groupby(["group", "outcome"]).rho.idxmax()].score.value_counts()
out(f"\nBlocks where each athleticism score has the highest correlation: {wins.to_dict()} (of {RAW.groupby(['group', 'outcome']).ngroups})")
clear = RAW[(RAW.lo > 0) | (RAW.hi < 0)]
out("Correlations whose interval excludes 0:")
for _, r in clear.iterrows():
    out(f"  {r.group:4s} {r.outcome:30s} {r.score:9s} {fmt(r.rho, r.lo, r.hi)}  n={r.n}")

# Pooled across all position-outcome blocks (one row per player-outcome; bootstrap over players).
PL = pd.concat(POOL, ignore_index=True)
out("\nPooled over every position-outcome block (within-block ranks):")
pooled_rows = []
blocks = PL.block.unique()
for slabel, scol in RAW_SCORES:
    d = PL.dropna(subset=["X_" + scol])
    pooled_rows.append((slabel, np.corrcoef(d["X_" + scol], d.Y)[0, 1], len(d)))
for slabel, r, n in pooled_rows:
    out(f"  {slabel:9s} {r:+.3f}  (player-outcome pairs = {n})")
d = PL.dropna(subset=["X_juice_final", "X_ngs_athleticism_score", "X_neg_forty"])
idx = rng.integers(0, len(d), (B, len(d)))
for slabel, scol in [("NGS", "ngs_athleticism_score"), ("40", "neg_forty"), ("RAS-style", "ras_style")]:
    dd = d.dropna(subset=["X_" + scol])
    ii = rng.integers(0, len(dd), (B, len(dd)))
    xa, xb, yv = dd.X_juice_final.to_numpy(), dd["X_" + scol].to_numpy(), dd.Y.to_numpy()
    est = np.corrcoef(xa, yv)[0, 1] - np.corrcoef(xb, yv)[0, 1]
    bs = [np.corrcoef(xa[i], yv[i])[0, 1] - np.corrcoef(xb[i], yv[i])[0, 1] for i in ii]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out(f"  Juice+ minus {slabel:9s} {fmt(est, lo, hi)}  ({'clear' if lo > 0 or hi < 0 else 'no clear difference'}; n={len(dd)})")
out("  Note: pairs from the same player in different blocks are not independent, so these intervals are a bit narrow.")

out("\nNegative correlations per score (raw):")
neg = RAW[(RAW.rho < 0) & (RAW.score != "Pick")]
out("  " + str(neg.groupby("score").size().to_dict()))
for _, r in neg.sort_values("rho").iterrows():
    out(f"  {r.group:4s} {r.outcome:30s} {r.score:9s} {r.rho:+.2f}")

# =====================================================================
hdr("7. FOOTBALL RESULTS, SIZE-FREE: WEIGHT AND HEIGHT REMOVED FROM EVERY SCORE")
# =====================================================================
out("Section 6 is unfair to Juice+: within position, lighter players have better results, and NGS, RAS-style and")
out("the 40 partly reward being light while Juice+ removes size by design. Here weight and height ranks are")
out("partialled out of every score AND the outcome, within position, so all four compete on equal size-free terms.")
out("Paired bootstrap over player-outcome pairs (same caveat as section 6: intervals a bit narrow).")
SF_SCORES = [s for s in RAW_SCORES if s[0] != "Pick"]
size_rows, sf_blocks = [], []
for grp, outs in OUTS.items():
    for ocol, olabel in outs:
        need = [c for _, c in SF_SCORES] + [ocol, "combine_weight", "combine_height"]
        g = P[P.pgm == grp].dropna(subset=need)
        if len(g) < 20:
            continue
        R = g[need].rank(pct=True)
        C = R[["combine_weight", "combine_height"]].to_numpy()
        y = residualize(R[ocol].to_numpy(), C)
        rw = stats.spearmanr(g.combine_weight, g[ocol])[0]
        q = pd.DataFrame({"block": grp + "|" + olabel, "ocol": ocol, "Y": y})
        line = f"  {grp:4s} {olabel[:30]:30s} n={len(g):3d} weight-rho {rw:+.2f} "
        for slabel, scol in SF_SCORES:
            x = residualize(R[scol].to_numpy(), C)
            q["Z_" + slabel] = x
            r = float(np.corrcoef(x, y)[0, 1])
            size_rows.append({"group": grp, "outcome": olabel, "score": slabel, "partial_r": r, "n": len(g),
                              "outcome_weight_rho": rw})
            line += f"| {slabel} {r:+.2f} "
        out(line)
        sf_blocks.append(q)
SF = pd.DataFrame(size_rows)
wins_sf = SF.loc[SF.groupby(["group", "outcome"]).partial_r.idxmax()].score.value_counts()
out(f"\nBlocks where each score is highest, size-free: {wins_sf.to_dict()} (of {SF.groupby(['group', 'outcome']).ngroups})")
sf_pool = []
for label, drop_yac in [("All blocks", False), ("Without YAC over expected", True)]:
    D = pd.concat([q for q in sf_blocks if not (drop_yac and q.ocol.iloc[0] == "o_yac_oe")], ignore_index=True)
    out(f"\nPooled, {label} (pairs = {len(D)}):")
    yv = D.Y.to_numpy()
    for slabel, _ in SF_SCORES:
        r = float(np.corrcoef(D["Z_" + slabel], yv)[0, 1])
        sf_pool.append({"pool": label, "score": slabel, "estimate": r, "ci_low": np.nan, "ci_high": np.nan, "n": len(D)})
        out(f"  {slabel:9s} {r:+.3f}")
    idx = rng.integers(0, len(D), (B, len(D)))
    a = D["Z_Juice+"].to_numpy()
    for slabel in ["NGS", "RAS-style", "40"]:
        b = D["Z_" + slabel].to_numpy()
        est = np.corrcoef(a, yv)[0, 1] - np.corrcoef(b, yv)[0, 1]
        bs = [np.corrcoef(a[i], yv[i])[0, 1] - np.corrcoef(b[i], yv[i])[0, 1] for i in idx]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sf_pool.append({"pool": label, "score": f"Juice+ minus {slabel}", "estimate": est, "ci_low": lo,
                        "ci_high": hi, "n": len(D)})
        out(f"  Juice+ minus {slabel:9s} {fmt(est, lo, hi)}  ({'clear' if lo > 0 or hi < 0 else 'no clear difference'})")
save_with_run(SF, "football_results_size_free.csv", RUN_ID)
save_with_run(pd.DataFrame(sf_pool), "football_results_size_free_pooled.csv", RUN_ID)

(OUT / "predictive_report.txt").write_text("\n".join(LINES))
print(f"\nSaved predictive_report.txt and tables to {OUT.resolve()} (run {RUN_ID})")
