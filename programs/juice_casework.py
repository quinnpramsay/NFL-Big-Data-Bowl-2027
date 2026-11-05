"""Case work on the two leads that survived: WR/DB agility and the OL burst chain.

Runs after juice.py, juice_physics.py and juice_transfer.py, on the same run ID. Four parts, in priority order:

  A  The agility case for WR and DB: trait specificity, robustness, year-2 catch-up, link to football results.
  B  The OL chain (Combine burst -> game burst -> peak pressure): what the plays are, play-mix controls, mediation.
  C  Late-round scouting examples: what the stopwatch missed, with a systematic pair test (no cherry-picking).
  D  Selection bias and robustness: who reaches the field, inverse-probability weights, snap floors, permutation.

Secondary checks S1-S4 were written in PRIMARY_TESTS.md before this script was run. Everything else is EXPLORATORY.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import OUT, get_file, load_run_id, read_out, save_with_run, section_rng, size_free  # noqa: E402

RUN_ID = load_run_id()
B = 2000
rng = np.random.default_rng(2027)
LINES = []
CHECKS = []


def out(text=""):
    LINES.append(str(text))
    print(text)


def hdr(text):
    """Section header. Also reseeds the shared generator from the section label, so adding or reordering
    sections never moves another section's bootstrap draws."""
    global rng
    rng = section_rng("casework", text.split()[0])
    out("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def fmt(r, lo, hi):
    return f"{r:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ---------------------------------------------------------------------
# Statistics helpers (rank partial correlations, optionally weighted)
# ---------------------------------------------------------------------

def resid(v, C, w=None):
    X = np.ones((len(v), 1)) if C is None else np.column_stack([np.ones(len(v)), C])
    if w is None:
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)[0]
    return v - X @ beta


def pcorr(x, y, C=None, w=None):
    rx, ry = resid(x, C, w), resid(y, C, w)
    if w is None:
        w = np.ones(len(x))
    w = w / w.sum()
    mx, my = (w * rx).sum(), (w * ry).sum()
    cov = (w * (rx - mx) * (ry - my)).sum()
    den = np.sqrt((w * (rx - mx) ** 2).sum() * (w * (ry - my) ** 2).sum())
    return float(cov / den) if den > 0 else np.nan


def dummies(g):
    d = pd.get_dummies(pd.Series(np.asarray(g)), drop_first=True).astype(float).to_numpy()
    return d if d.shape[1] else None


def stack(*cols):
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    return np.column_stack([c if np.ndim(c) == 2 else np.asarray(c, float)[:, None] for c in cols])


def ranked(d, cols, by="pgm"):
    """Within-group percentile ranks, so pooled groups compete on the same scale. Same values as pandas
    groupby().rank(pct=True) (average ties, NaN left out), done in numpy because the bootstraps call it thousands
    of times."""
    g = d[by].to_numpy()
    groups = [np.flatnonzero(g == u) for u in pd.unique(g)]
    res = {}
    for c in cols:
        v = d[c].to_numpy(float)
        r = np.full(len(v), np.nan)
        for ix in groups:
            vv = v[ix]
            ok = np.isfinite(vv)
            if ok.any():
                r[ix[ok]] = rankdata(vv[ok]) / ok.sum()
        res[c] = r
    return res


def rank_partial(d, x, y, controls=(), weights=None, reps=B):
    """Partial Spearman of x with y within group, holding controls, with group dummies when pooled.
    Returns estimate, 95% percentile bootstrap interval, n."""
    cols = [x, y] + list(controls)
    d = d.dropna(subset=cols + ([weights] if weights else [])).reset_index(drop=True)
    n = len(d)
    if n < 20:
        return np.nan, np.nan, np.nan, n
    R = ranked(d, cols)
    C = stack(*[R[c] for c in controls], dummies(d.pgm))
    w = d[weights].to_numpy() if weights else None
    est = pcorr(R[x], R[y], C, w)
    vals = []
    for idx in rng.integers(0, n, (reps, n)):
        dd = d.iloc[idx].reset_index(drop=True)
        Rb = ranked(dd, cols)
        Cb = stack(*[Rb[c] for c in controls], dummies(dd.pgm))
        vals.append(pcorr(Rb[x], Rb[y], Cb, None if w is None else w[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi, n


def line(label, res, width=58):
    est, lo, hi, n = res
    if not np.isfinite(est):
        return f"  {label:{width}s}  n={n} (too few)"
    return f"  {label:{width}s} {fmt(est, lo, hi)}  n={n}"


def check(cid, claim, res, passed, note=""):
    est, lo, hi, n = res if len(res) == 4 else (*res, np.nan)
    CHECKS.append({"id": cid, "claim": claim, "estimate": est, "ci_low": lo, "ci_high": hi, "n": n,
                   "verdict": "passes" if passed else "does not pass", "note": note})


# size_free (residual on weight and height within position) lives in juice_common: groups under 10 players are
# left blank, because a 3-parameter fit on 2-3 players is exact and leaves only rounding noise.


def logistic(X, y, iters=50):
    """Plain Newton-Raphson logistic regression with an intercept and a tiny ridge for stability."""
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b, 1 / (1 + np.exp(-X @ b))


# ---------------------------------------------------------------------
# Load this run's tables
# ---------------------------------------------------------------------
P = read_out("player_outcomes.csv", RUN_ID)
PLM = read_out("play_level_movement.csv", RUN_ID)
PLO = read_out("play_level_outcomes.csv", RUN_ID)
CS = pd.read_csv(get_file("player_career_successes.csv"))
P = P.merge(CS[["nfl_id", "career_offensive_snaps", "career_defensive_snaps", "career_games_started",
                "career_games_active"]], on="nfl_id", how="left")
for c in ["career_offensive_snaps", "career_defensive_snaps", "career_games_started", "career_games_active"]:
    P[c] = P[c].fillna(0)
P["career_snaps"] = P.career_offensive_snaps + P.career_defensive_snaps
P["snap_pct"] = P.groupby(["pgm", "draft_year"]).career_snaps.rank(pct=True)
P["reached"] = P.game_juice.notna().astype(float)
P["udfa"] = P.draft_overall_pick.isna() | (P.draft_overall_pick <= 0)
WRDB = ["WR", "DB"]
out(f"Run {RUN_ID}. Combine-scored players: {len(P)}; WR {int((P.pgm == 'WR').sum())}, DB {int((P.pgm == 'DB').sum())}, "
    f"OL {int((P.pgm == 'OL').sum())}")

# Raw game agility per player from plays (no floor), for floor sensitivity and the late-round table.
plm_ag = PLM[(PLM.n_after >= 20) & PLM.alat_c.notna()]

# =====================================================================
hdr("A. THE AGILITY CASE FOR WR AND DB")
# =====================================================================
out("Within-group percentile ranks; pooled rows add a group dummy. Combine pieces are the tracked-only, size-adjusted")
out("scores; game pieces are the size-free Game Juice+ pieces (50+ plays). 2,000 bootstrap resamples of players.")

out("\nA1. Trait specificity: which Combine number predicts which game trait?")
out("Each tracked Combine piece holds the 40 fixed; the 40 row holds the Combine agility piece fixed.")
SPEC_ROWS = []
for grp_label, grps in (("WR", ["WR"]), ("DB", ["DB"]), ("WR + DB", WRDB)):
    d = P[P.pgm.isin(grps)]
    out(f"\n  {grp_label}")
    for gcol, glabel in (("g_agil_plus", "game agility"), ("g_top_plus", "game top speed"),
                         ("g_expl_plus", "game burst")):
        for xcol, xlabel, ctrl in (("A_trk_plus", "Combine agility", ["neg_forty"]),
                                   ("T_trk_plus", "Combine speed", ["neg_forty"]),
                                   ("E_trk_plus", "Combine burst", ["neg_forty"]),
                                   ("neg_forty", "Official 40", ["A_trk_plus"])):
            res = rank_partial(d, xcol, gcol, ctrl, reps=1000)
            SPEC_ROWS.append({"group": grp_label, "game_trait": glabel, "combine": xlabel,
                              "holding": "+".join(ctrl), "r": res[0], "lo": res[1], "hi": res[2], "n": res[3]})
            out(line(f"{xlabel:16s} -> {glabel:15s} (holding {'the 40' if ctrl == ['neg_forty'] else 'agility'})",
                     res, 52))
SPEC = pd.DataFrame(SPEC_ROWS)
save_with_run(SPEC, "agility_specificity.csv", RUN_ID)

# S1: A vs game agility holding the 40, and A minus the 40 (each holding the other), paired bootstrap.
d1 = P[P.pgm.isin(WRDB)].dropna(subset=["A_trk_plus", "neg_forty", "g_agil_plus"]).reset_index(drop=True)
s1 = rank_partial(d1, "A_trk_plus", "g_agil_plus", ["neg_forty"])


def a_minus_40(dd):
    R = ranked(dd, ["A_trk_plus", "neg_forty", "g_agil_plus"])
    G = dummies(dd.pgm)
    return (pcorr(R["A_trk_plus"], R["g_agil_plus"], stack(R["neg_forty"], G))
            - pcorr(R["neg_forty"], R["g_agil_plus"], stack(R["A_trk_plus"], G)))


diff = a_minus_40(d1)
dvals = [a_minus_40(d1.iloc[i].reset_index(drop=True)) for i in rng.integers(0, len(d1), (B, len(d1)))]
dlo, dhi = np.percentile(dvals, [2.5, 97.5])
out(f"\n  S1  Combine agility -> game agility, holding the 40 (WR + DB): {fmt(*s1[:3])}  n={s1[3]}")
out(f"      Agility minus the 40 (each holding the other):             {fmt(diff, dlo, dhi)}")
check("S1", "WR+DB: Combine agility -> game agility beyond the 40", s1, s1[1] > 0 and dlo > 0,
      f"A minus 40 = {diff:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")

out("\nA2. Robustness of S1 (WR + DB, Combine agility -> game agility, holding the 40)")
for ctrl, lab in ((["neg_forty", "ras_style"], "holding 40 + RAS-style"),
                  (["neg_forty", "pick"], "holding 40 + draft pick"),
                  (["neg_forty", "log_snaps"], "holding 40 + log plays"),
                  (["neg_forty", "ras_style", "pick", "log_snaps"], "holding 40 + RAS + pick + plays"),
                  (["neg_forty", "combine_weight", "combine_height"], "holding 40 + weight + height")):
    out(line(lab, rank_partial(d1, "A_trk_plus", "g_agil_plus", ctrl, reps=1000)))
for yr in sorted(d1.draft_year.unique()):
    out(line(f"draft class {int(yr)} only", rank_partial(d1[d1.draft_year == yr], "A_trk_plus", "g_agil_plus",
                                                          ["neg_forty"], reps=1000)))
    out(line(f"drop class {int(yr)}", rank_partial(d1[d1.draft_year != yr], "A_trk_plus", "g_agil_plus",
                                                    ["neg_forty"], reps=1000)))
# Influence: drop the players whose removal lowers the estimate most (jackknife), 1 to 10 at a time.
R1 = ranked(d1, ["A_trk_plus", "neg_forty", "g_agil_plus"])
base = s1[0]
jk = np.array([pcorr(np.delete(R1["A_trk_plus"], i), np.delete(R1["g_agil_plus"], i),
                     stack(np.delete(R1["neg_forty"], i), dummies(np.delete(d1.pgm.to_numpy(), i))))
               for i in range(len(d1))])
infl = np.argsort(base - jk)[::-1]     # largest drop first
for k in (1, 3, 5, 10):
    keep = d1.drop(index=infl[:k]).reset_index(drop=True)
    R_ = ranked(keep, ["A_trk_plus", "neg_forty", "g_agil_plus"])
    r_ = pcorr(R_["A_trk_plus"], R_["g_agil_plus"], stack(R_["neg_forty"], dummies(keep.pgm)))
    out(f"  drop the {k:2d} most influential players{'':22s} {r_:+.3f}  n={len(keep)}")
out("  most influential: " + ", ".join(f"{d1.display_name[i]} ({d1.nfl_position[i]})" for i in infl[:5]))
# Permutation: shuffle Combine agility within group.
# Shuffling A within group just permutes its within-group ranks, so the ranks are computed once.
perm = []
Rp = ranked(d1, ["A_trk_plus", "neg_forty", "g_agil_plus"])
Cp = stack(Rp["neg_forty"], dummies(d1.pgm))
grp_ix = [np.flatnonzero(d1.pgm.to_numpy() == u) for u in pd.unique(d1.pgm)]
for _ in range(B):
    ra = Rp["A_trk_plus"].copy()
    for ix in grp_ix:
        ra[ix] = Rp["A_trk_plus"][rng.permutation(ix)]
    perm.append(pcorr(ra, Rp["g_agil_plus"], Cp))
p_perm = (np.sum(np.abs(perm) >= abs(base)) + 1) / (B + 1)
out(f"  permutation p (agility shuffled within group, {B} draws): {p_perm:.4f}")

out("\nA3. Year-2 catch-up, WR + DB only (game agility season means, 50+ plays per season, size-free)")
s = plm_ag.groupby(["nfl_id", "season"]).alat_c.agg(["mean", "count"]).reset_index()
s = s[s["count"] >= 50].merge(P[["nfl_id", "pgm", "draft_year", "A_trk_plus", "neg_forty", "combine_weight",
                                 "combine_height"]], on="nfl_id")
s = s[s.pgm.isin(WRDB)]
s["sf"] = pd.concat([size_free(g, "mean") for _, g in s.groupby("season")])
s["z"] = s.groupby(["pgm", "season"]).sf.transform(lambda v: (v - v.mean()) / v.std())
y1 = s[s.season == s.draft_year].set_index("nfl_id")
y2 = s[s.season == s.draft_year + 1].set_index("nfl_id")
yd = y1[["pgm", "z", "A_trk_plus", "neg_forty"]].rename(columns={"z": "y1"}).join(
    y2[["z"]].rename(columns={"z": "y2"})).dropna()
for c in ("A_trk_plus", "neg_forty"):
    yd[c] = yd.groupby("pgm")[c].transform(lambda v: (v - v.mean()) / v.std())
X = np.column_stack([np.ones(len(yd)), yd.y1, yd.A_trk_plus, yd.neg_forty])
bt = np.linalg.lstsq(X, yd.y2, rcond=None)[0]
bs = np.array([np.linalg.lstsq(X[i], yd.y2.to_numpy()[i], rcond=None)[0] for i in rng.integers(0, len(yd), (B, len(yd)))])
out(f"  year-2 agility ~ year-1 + Combine agility + 40, n={len(yd)}")
out(f"    Combine agility {fmt(bt[2], *np.percentile(bs[:, 2], [2.5, 97.5]))}   "
    f"the 40 {fmt(bt[3], *np.percentile(bs[:, 3], [2.5, 97.5]))}   year 1 {bt[1]:+.2f}")
yd["gap"] = yd.A_trk_plus - yd.y1
fl = yd[yd.gap >= 1.0]
out(f"    flagged 'unused agility' (Combine agility 1+ SD above year-1 game agility): {len(fl)}; "
    f"mean change year 1 -> 2: {(fl.y2 - fl.y1).mean():+.2f} SD vs {(yd.y2 - yd.y1)[yd.gap < 1].mean():+.2f} for the rest")
UNUSED = fl.join(P.set_index("nfl_id")[["display_name", "nfl_position", "draft_year", "draft_overall_pick"]])

out("\nA4. Link 2: does game agility show up in football results? (exploratory)")
pp = pd.read_csv(get_file("player_play.csv"), usecols=["game_id", "play_id", "nfl_id", "team_coverage_man_zone",
                                                         "coverage_assignment", "pass_length", "pass_result",
                                                         "dropback_duration"])
plo = PLO.merge(pp, on=["game_id", "play_id", "nfl_id"], how="left")
man = plo[(plo.pgm == "DB") & plo.cov_epa.notna() & (plo.coverage_assignment == "MAN")]
zone = plo[(plo.pgm == "DB") & plo.cov_epa.notna() & (plo.coverage_assignment != "MAN")]
for frame, col in ((man, "o_cov_man"), (zone, "o_cov_zone")):
    agg = frame.groupby("nfl_id").cov_epa.agg(["mean", "count"])
    P[col] = P.nfl_id.map(agg["mean"].where(agg["count"] >= 30))
for grp, ocol, olab in (("WR", "o_sep_c", "separation at throw"), ("WR", "o_rec_yds_route", "yards per route"),
                        ("DB", "o_cov_epa", "coverage EPA (flipped), all"),
                        ("DB", "o_cov_man", "coverage EPA (flipped), man only"),
                        ("DB", "o_cov_zone", "coverage EPA (flipped), zone only")):
    d = P[P.pgm == grp]
    out(line(f"{grp} game agility -> {olab}", rank_partial(d, "g_agil_plus", ocol, reps=1000)))
    out(line(f"{grp} game agility -> {olab}, holding game top speed",
             rank_partial(d, "g_agil_plus", ocol, ["g_top_plus"], reps=1000)))

# =====================================================================
hdr("B. THE OL CHAIN: COMBINE BURST -> GAME BURST -> PEAK PRESSURE ALLOWED")
# =====================================================================
ol_b = PLM[(PLM.pgm == "OL") & PLM.expl_c.notna()].merge(pp, on=["game_id", "play_id", "nfl_id"], how="left")
per = ol_b.groupby("nfl_id").size()
out("B0. What an OL 'burst' play is: a pass-pro snap where the lineman got 10+ yards from his snap spot from a")
out("    standing start (screens, and chasing after the catch).")
out(f"    plays {len(ol_b):,}, players {per.size}; per player median {per.median():.0f}, quartiles "
    f"{per.quantile(.25):.0f}-{per.quantile(.75):.0f}; throws at or behind the line {(ol_b.pass_length <= 0).mean():.0%}")
ol_o = plo[(plo.pgm == "OL") & plo.ppp_held_c.notna()].copy()
key = ol_b[["game_id", "play_id", "nfl_id"]].assign(is_burst=1)
ol_o = ol_o.merge(key, on=["game_id", "play_id", "nfl_id"], how="left")
ol_o["is_burst"] = ol_o.is_burst.fillna(0)
ol_o["screen"] = (ol_o.pass_length <= 0).astype(float)
out(f"    share of OL peak-pressure plays that are also burst plays: {ol_o.is_burst.mean():.1%}")
agg_all = ol_o.groupby("nfl_id").ppp_held_c.agg(["mean", "count"])
agg_nb = ol_o[ol_o.is_burst == 0].groupby("nfl_id").ppp_held_c.agg(["mean", "count"])
agg_ns = ol_o[(ol_o.is_burst == 0) & (ol_o.screen == 0)].groupby("nfl_id").ppp_held_c.agg(["mean", "count"])
P["o_ppp_noburst"] = P.nfl_id.map(agg_nb["mean"].where(agg_nb["count"] >= 30))
P["o_ppp_noscreen"] = P.nfl_id.map(agg_ns["mean"].where(agg_ns["count"] >= 30))
P["screen_share"] = P.nfl_id.map(ol_o.groupby("nfl_id").screen.mean())
P["dropback_len"] = P.nfl_id.map(ol_o.groupby("nfl_id").dropback_duration.mean())
P["n_burst"] = P.nfl_id.map(per).fillna(0)
OL = P[P.pgm == "OL"].copy()
out(f"    OL with game burst score: {OL.g_expl_plus.notna().sum()}; game burst vs number of burst plays "
    f"(Spearman): {OL[['g_expl_plus', 'n_burst']].corr('spearman').iloc[0, 1]:+.2f}; vs screen share: "
    f"{OL[['g_expl_plus', 'screen_share']].corr('spearman').iloc[0, 1]:+.2f}")

out("\nB1. Each link with intervals")
out(line("L1  Combine burst -> game burst", rank_partial(OL, "E_trk_plus", "g_expl_plus")))
out(line("L1  Combine burst -> game burst, holding the 40", rank_partial(OL, "E_trk_plus", "g_expl_plus", ["neg_forty"])))
out(line("L1  ... holding the 40 + number of burst plays",
         rank_partial(OL, "E_trk_plus", "g_expl_plus", ["neg_forty", "n_burst"])))
out(line("L2  game burst -> peak pressure (flipped), all dropbacks", rank_partial(OL, "g_expl_plus", "o_ppp_held_c")))
out(line("L2  ... dropbacks that are not burst plays", rank_partial(OL, "g_expl_plus", "o_ppp_noburst")))
out(line("L2  ... not burst plays and not screens", rank_partial(OL, "g_expl_plus", "o_ppp_noscreen")))
s3 = rank_partial(OL, "g_expl_plus", "o_ppp_noburst", ["screen_share", "n_burst"])
out(line("L2  ... not burst plays, holding screen share + n burst plays", s3))
out(line("L2  ... plus pick and log plays", rank_partial(OL, "g_expl_plus", "o_ppp_noburst",
                                                         ["screen_share", "n_burst", "pick", "log_snaps"])))
out(line("L2  ... plus mean dropback length", rank_partial(OL, "g_expl_plus", "o_ppp_noburst",
                                                           ["screen_share", "n_burst", "dropback_len"])))
out(line("L2  game top speed -> same outcome (placebo trait)", rank_partial(OL, "g_top_plus", "o_ppp_noburst",
                                                                           ["screen_share", "n_burst"])))
check("S3", "OL: game burst -> peak pressure on non-burst dropbacks, holding play mix", s3, s3[1] > 0)
out(line("direct  Combine burst -> peak pressure, all dropbacks", rank_partial(OL, "E_trk_plus", "o_ppp_held_c")))
out(line("direct  ... non-burst dropbacks, holding the 40", rank_partial(OL, "E_trk_plus", "o_ppp_noburst", ["neg_forty"])))
for yr in sorted(OL.draft_year.dropna().unique()):
    out(line(f"L2 non-burst, class {int(yr)} only", rank_partial(OL[OL.draft_year == yr], "g_expl_plus",
                                                               "o_ppp_noburst", reps=1000)))

out("\nB2. Mediation: Combine burst -> game burst -> peak pressure on non-burst dropbacks")
dm = OL.dropna(subset=["E_trk_plus", "g_expl_plus", "o_ppp_noburst"]).reset_index(drop=True)


def ab(dd):
    R = ranked(dd, ["E_trk_plus", "g_expl_plus", "o_ppp_noburst"])
    z = {k: (v - v.mean()) / v.std() for k, v in R.items()}
    a = np.polyfit(z["E_trk_plus"], z["g_expl_plus"], 1)[0]
    Xb = np.column_stack([np.ones(len(dd)), z["g_expl_plus"], z["E_trk_plus"]])
    bb = np.linalg.lstsq(Xb, z["o_ppp_noburst"], rcond=None)[0]
    return a, bb[1], a * bb[1], bb[2]


a_, b_, ab_, c_ = ab(dm)
abs_ = np.array([ab(dm.iloc[i].reset_index(drop=True)) for i in rng.integers(0, len(dm), (B, len(dm)))])
lo_ab, hi_ab = np.percentile(abs_[:, 2], [2.5, 97.5])
out(f"  a (Combine -> game) {a_:+.3f}   b (game -> outcome, holding Combine) {b_:+.3f}   direct c' {c_:+.3f}")
out(f"  S4  indirect a x b {fmt(ab_, lo_ab, hi_ab)}  n={len(dm)}  (standardized ranks; total = c' + ab = {c_ + ab_:+.3f})")
check("S4", "OL: indirect effect Combine burst -> game burst -> peak pressure", (ab_, lo_ab, hi_ab, len(dm)), lo_ab > 0)

out("\nB3. Exploratory rescue (rule in PRIMARY_TESTS.md, written after S3/S4 failed): other game burst measures")
out("    q95 = the Game Juice+ piece; mean = mean per play; mixed = random-intercept player effect (every player")
out("    with 1+ burst play, shrunk by play count). All size-free within OL.")
gb = PLM[(PLM.pgm == "OL") & PLM.expl_c.notna()].groupby("nfl_id").expl_c
st = gb.agg(["mean", "var", "size"])
sig2 = float((st["var"].fillna(0) * (st["size"] - 1)).sum() / max((st["size"] - 1).sum(), 1))
mu = st["mean"].mean()
tau2 = max(float(st["mean"].var() - (sig2 / st["size"]).mean()), 1e-9)
blup = mu + tau2 / (tau2 + sig2 / st["size"]) * (st["mean"] - mu)
out(f"    burst plays: within-player variance {sig2:.3f}, between-player variance {tau2:.3f} "
    f"(share {tau2 / (tau2 + sig2):.3f}); a lineman with 24 plays keeps {tau2 / (tau2 + sig2 / 24):.0%} of his own mean")
P["gb_mixed"] = P.nfl_id.map(blup)
for fl in (10, 20):
    P[f"gb_mean{fl}"] = P.nfl_id.map(st["mean"].where(st["size"] >= fl))
for c in ("gb_mixed", "gb_mean10", "gb_mean20"):
    P[c] = size_free(P, c)
OL = P[P.pgm == "OL"].copy()
RESCUE = []
for col, lab in (("g_expl_plus", "q95 (current, 50+ plays)"), ("gb_mean10", "mean, 10+ burst plays"),
                 ("gb_mean20", "mean, 20+ burst plays"), ("gb_mixed", "mixed-model effect, 1+ play")):
    rho_n = OL[[col, "n_burst"]].corr("spearman").iloc[0, 1]
    l1 = rank_partial(OL, "E_trk_plus", col, ["neg_forty"])
    l2 = rank_partial(OL, col, "o_ppp_noburst", ["screen_share", "n_burst"])
    ok = l1[1] > 0 and l2[1] > 0
    RESCUE.append({"measure": lab, "rho_with_n_burst": rho_n, "l1": l1[0], "l1_lo": l1[1], "l1_hi": l1[2],
                   "l1_n": l1[3], "l2": l2[0], "l2_lo": l2[1], "l2_hi": l2[2], "l2_n": l2[3], "improves": ok})
    out(f"  {lab}  (vs number of burst plays {rho_n:+.2f})")
    out(line("    L1 Combine burst -> game burst, holding the 40", l1))
    out(line("    L2 game burst -> non-burst peak pressure, play-mix controls", l2))
    out(f"    {'meets the rule' if ok else 'does not meet the rule'}")
save_with_run(pd.DataFrame(RESCUE), "ol_burst_rescue.csv", RUN_ID)


# =====================================================================
hdr("C. LATE-ROUND SCOUTING EXAMPLES (pick 140+ or undrafted)")
# =====================================================================
for c in ("A_trk_plus", "neg_forty", "E_trk_plus", "g_agil_plus", "g_expl_plus", "g_top_plus", "o_ppp_noburst"):
    P["pct_" + c] = P.groupby("pgm")[c].rank(pct=True)
# Raw (no floor) game agility, size-free, so players under 50 plays still show something (labelled with play count).
ga = plm_ag.groupby("nfl_id").alat_c.agg(q95=lambda v: v.quantile(0.95), n="size")
P["g_agil_raw"] = P.nfl_id.map(ga.q95)
P["g_agil_n"] = P.nfl_id.map(ga.n).fillna(0).astype(int)
P["g_agil_raw_sf"] = size_free(P, "g_agil_raw")
P.loc[P.g_agil_n < 20, "g_agil_raw_sf"] = np.nan
P["pct_g_agil_raw"] = P.groupby("pgm").g_agil_raw_sf.rank(pct=True)
late = P[(P.pick >= 140) & P.pgm.isin(WRDB) & P.A_trk_plus.notna() & P.neg_forty.notna()].copy()
late["gap"] = late.pct_A_trk_plus - late.pct_neg_forty
out(f"Late-round WR + DB with Combine agility and a 40: {len(late)}")
missed = late[(late.pct_A_trk_plus >= 2 / 3) & (late.gap >= 0.25)].sort_values("pct_A_trk_plus", ascending=False)
out("\nC1. 'The stopwatch missed it': top-third Combine agility, agility percentile 25+ points above the 40 percentile")
cols = ["display_name", "nfl_position", "draft_year", "pick", "pct_neg_forty", "pct_A_trk_plus", "g_agil_n",
        "pct_g_agil_raw", "career_snaps", "snap_pct", "career_games_started"]


def show(df):
    t = df[cols].copy()
    t["pick"] = np.where(t.pick >= 260, "UDFA", t.pick.astype(int).astype(str))
    for c in ("pct_neg_forty", "pct_A_trk_plus", "pct_g_agil_raw", "snap_pct"):
        t[c] = (100 * t[c]).round(0)
    t.columns = ["player", "pos", "class", "pick", "40 pct", "agility pct", "plays", "game agil pct",
                 "career snaps", "snap pct (class)", "starts"]
    return t.to_string(index=False)


out(show(missed) if len(missed) else "  none")
hi_a = late[late.pct_A_trk_plus >= 2 / 3]
rest = late[late.pct_A_trk_plus < 2 / 3]
for lab, frame in (("top-third Combine agility", hi_a), ("everyone else", rest)):
    g = frame.dropna(subset=["pct_g_agil_raw"])
    out(f"  {lab:26s}: game agility top half {int((g.pct_g_agil_raw >= .5).sum())}/{len(g)}"
        f" = {(g.pct_g_agil_raw >= .5).mean():.0%};  reached 20+ open plays {int((frame.g_agil_n >= 20).sum())}/{len(frame)}"
        f";  median snap pct in class {frame.snap_pct.median():.0%}")
save_with_run(late.sort_values("gap", ascending=False)[cols + ["gap"]], "late_round_agility.csv", RUN_ID)

out("\nC2. Pair test (all pairs, no cherry-picking): same position group and class, 40 within 10 percentile points,")
out("    Combine agility 30+ points apart, both with 20+ open plays. Does the more agile player move better in games?")
PR = []
pool = P[P.pgm.isin(WRDB)].dropna(subset=["pct_A_trk_plus", "pct_neg_forty", "g_agil_raw_sf"])
for (_, _), g in pool.groupby(["pgm", "draft_year"]):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.iloc[i], g.iloc[j]
            if abs(a.pct_neg_forty - b.pct_neg_forty) <= .10 and abs(a.pct_A_trk_plus - b.pct_A_trk_plus) >= .30:
                hi_, lo_ = (a, b) if a.pct_A_trk_plus > b.pct_A_trk_plus else (b, a)
                PR.append({"agile": hi_.display_name, "other": lo_.display_name, "pgm": hi_.pgm,
                           "class": int(hi_.draft_year), "agile_wins_game": float(hi_.g_agil_raw_sf > lo_.g_agil_raw_sf),
                           "agile_more_snaps": float(hi_.career_snaps > lo_.career_snaps),
                           "late_involved": bool(hi_.pick >= 140 or lo_.pick >= 140)})
PR = pd.DataFrame(PR)
if len(PR):
    # Players appear in several pairs, so resample players' pairs by bootstrap over pairs (approximate).
    w = PR.agile_wins_game.mean()
    wv = PR.agile_wins_game.to_numpy(float)
    bw = wv[rng.integers(0, len(wv), (B, len(wv)))].mean(axis=1)
    out(f"  pairs {len(PR)} (players appear in several pairs, so the interval is approximate)")
    out(f"  more agile player has higher game agility: {w:.0%} [{np.percentile(bw, 2.5):.0%}, {np.percentile(bw, 97.5):.0%}]"
        f" (50% = coin flip)")
    out(f"  more agile player has more career snaps: {PR.agile_more_snaps.mean():.0%}")
    lp = PR[PR.late_involved]
    out(f"  pairs with a late-round player: {len(lp)}; agile wins in games {lp.agile_wins_game.mean():.0%}")
    save_with_run(PR, "agility_pairs.csv", RUN_ID)

out("\nC3. Unused agility (year-2 catch-up candidates, WR + DB)")
if len(UNUSED):
    u = UNUSED.assign(change=UNUSED.y2 - UNUSED.y1).sort_values("change", ascending=False)
    for nid, r in u.iterrows():
        pk = "UDFA" if not (r.draft_overall_pick > 0) else int(r.draft_overall_pick)
        out(f"  {r.display_name:24s} {r.nfl_position:3s} {int(r.draft_year)} pick {pk!s:>4s}  year-1 {r.y1:+.2f} -> year-2 "
            f"{r.y2:+.2f} SD (Combine {r.A_trk_plus:+.2f})")

out("\nC4. Late-round OL with top-third Combine burst")
lo_ol = P[(P.pgm == "OL") & (P.pick >= 140) & (P.pct_E_trk_plus >= 2 / 3)].sort_values("pct_E_trk_plus", ascending=False)
for _, r in lo_ol.iterrows():
    pk = "UDFA" if r.pick >= 260 else int(r.pick)
    out(f"  {r.display_name:24s} {r.nfl_position:2s} {int(r.draft_year)} pick {pk!s:>4s}  Combine burst pct "
        f"{100 * r.pct_E_trk_plus:3.0f}  game burst pct {'' if np.isnan(r.pct_g_expl_plus) else f'{100 * r.pct_g_expl_plus:3.0f}':>3s}"
        f"  pressure-held pct {'' if np.isnan(r.pct_o_ppp_noburst) else f'{100 * r.pct_o_ppp_noburst:3.0f}':>3s}"
        f"  snaps {int(r.career_snaps)}")

# =====================================================================
hdr("D. SELECTION BIAS AND ROBUSTNESS")
# =====================================================================
out("D1. Who reaches a game score (50+ plays)? Logistic regression, WR + DB + OL, within-group ranks.")
sel = P[P.pgm.isin(WRDB + ["OL"])].copy()
for c in ("pick", "A_trk_plus", "E_trk_plus", "neg_forty"):
    sel["r_" + c] = sel.groupby("pgm")[c].rank(pct=True)
for lab, grps, piece in (("WR + DB", WRDB, "A_trk_plus"), ("OL", ["OL"], "E_trk_plus")):
    d = sel[sel.pgm.isin(grps)].dropna(subset=["r_pick", "r_" + piece, "r_neg_forty"]).copy()
    Xs = stack(d.r_pick, d["r_" + piece], d.r_neg_forty, dummies(d.pgm), dummies(d.draft_year))
    bsel, p_hat = logistic(Xs, d.reached.to_numpy())
    out(f"  {lab}: n={len(d)}, reached {int(d.reached.sum())}; coefficients (per full rank range): pick (later) "
        f"{bsel[1]:+.2f}, {piece.split('_')[0]} {bsel[2]:+.2f}, 40 {bsel[3]:+.2f}")
    P.loc[d.index, "p_reach"] = p_hat
P["ipw"] = (1 / P.p_reach.clip(0.05, 1)).where(P.reached == 1)

out("\nD2. Key results re-weighted by 1 / P(reach 50 plays) (stands in for the players who never got there)")
s2a = rank_partial(P[P.pgm.isin(WRDB)], "A_trk_plus", "g_agil_plus", ["neg_forty"], weights="ipw")
out(line("S2a WR + DB agility -> game agility, holding the 40, weighted", s2a))
out(line("    OL game burst -> non-burst peak pressure, play-mix controls, weighted",
         rank_partial(P[P.pgm == "OL"], "g_expl_plus", "o_ppp_noburst", ["screen_share", "n_burst"], weights="ipw")))

out("\nD3. Snap floor: S1 on game agility rebuilt from plays at each floor (mean is unbiased at low counts; the")
out("    95th percentile, which Game Juice+ uses, sinks with few plays). Size-free within group.")
FL = []
agg = plm_ag.groupby("nfl_id").alat_c.agg(mean="mean", q95=lambda v: v.quantile(0.95), n="size")
for floor in (10, 25, 50, 100, 200):
    for stat in ("mean", "q95"):
        col = f"ga_{stat}_{floor}"
        P[col] = P.nfl_id.map(agg[stat].where(agg.n >= floor))
        P[col] = size_free(P, col)
        res = rank_partial(P[P.pgm.isin(WRDB)], "A_trk_plus", col, ["neg_forty"], reps=500)
        FL.append({"floor": floor, "stat": stat, "r": res[0], "lo": res[1], "hi": res[2], "n": res[3]})
FL = pd.DataFrame(FL)
for _, r in FL.iterrows():
    out(f"  floor {r.floor:3d}  {r.stat:4s}  {fmt(r.r, r.lo, r.hi)}  n={r.n}")
save_with_run(FL, "agility_floor_sensitivity.csv", RUN_ID)
s2b_ok = bool((FL[FL.stat == "mean"].r > 0).all())
check("S2", "S1 survives selection weights and every snap floor (mean)", s2a,
      s2a[1] > 0 and s2b_ok, f"floor estimates (mean): {', '.join(f'{v:+.2f}' for v in FL[FL.stat == 'mean'].r)}")

out("\nD4. Does reaching the field depend on the trait? (selection on the predictor, WR + DB)")
for lab, frame in (("reached 50+ plays", P[P.pgm.isin(WRDB) & (P.reached == 1)]),
                   ("did not", P[P.pgm.isin(WRDB) & (P.reached == 0)])):
    out(f"  {lab:18s} n={len(frame):3d}  mean Combine agility pct {frame.pct_A_trk_plus.mean():.2f}"
        f"  mean 40 pct {P.loc[frame.index, 'pct_neg_forty'].mean():.2f}  mean pick {frame.pick.mean():.0f}")

# =====================================================================
hdr("E. SECONDARY CHECKS (from PRIMARY_TESTS.md)")
# =====================================================================
CK = pd.DataFrame(CHECKS)
out(CK[["id", "claim", "estimate", "ci_low", "ci_high", "n", "verdict"]].round(3).to_string(index=False))
for _, r in CK.iterrows():
    if r.note:
        out(f"  {r.id}: {r.note}")
save_with_run(CK, "secondary_checks.csv", RUN_ID)
(OUT / "casework_report.txt").write_text("\n".join(LINES) + "\n")
print(f"\nWrote {OUT / 'casework_report.txt'}")
