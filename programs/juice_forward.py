"""2026 forward test (F1 in PRIMARY_TESTS.md): predict 2026 WR separation from 2023-25 data only.

  python programs/juice_forward.py --freeze      once, after juice_routes.py: writes
                                                 forward_test/forward_predictions_2026.csv and refuses to overwrite it
  python programs/juice_forward.py --check-ngs   validates the outcome measure on 2023-25 (uses no 2026 data):
                                                 does Next Gen Stats separation over expected agree with the in-data
                                                 separation for the same player-seasons?
  python programs/juice_forward.py --score       scores the frozen predictions on the latest nflverse 2026 release.
                                                 Refuses to run before SCORE_AFTER unless --force (no early peeking).

Outcome (pre-registered): 2026 regular-season NGS separation over expected for cohort WRs with 30+ 2026 targets,
where expected = linear fit of avg_separation on avg_intended_air_yards and avg_cushion over every 2026 NGS receiver
(season-to-date row, week 0). nflverse data is public and free (competition rules, section 2.6).
"""

import argparse
import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import CACHE, get_file, load_run_id, read_out, section_rng  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FWD = ROOT / "forward_test"
PRED = FWD / "forward_predictions_2026.csv"
NGS_URL = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_receiving.csv.gz"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
SCORE_AFTER = "2026-12-24"
MIN_ROUTES, MIN_TARGETS, B = 50, 30, 2000
PREDICTORS = {"pred_sep_2325": "(a) route-adjusted separation 2023-25 (baseline)",
              "pred_prethrow": "(b) pre-throw movement score (cut speed + top speed before throw)",
              "pred_game_juice": "(c) Game Juice+",
              "pred_combine_juice": "(d) Combine Juice+"}
rng = section_rng("forward", "score")


def fetch(url, name, refresh):
    dest = CACHE / name
    if refresh or not dest.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "juice-plus-bdb"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def ngs_soe(season, refresh):
    """Season-to-date NGS rows (week 0, regular season) with separation over expected, mapped to nfl_id."""
    ngs = pd.read_csv(fetch(NGS_URL, "ngs_receiving.csv.gz", refresh), low_memory=False)
    ngs = ngs[(ngs.season == season) & (ngs.season_type == "REG") & (ngs.week == 0)].copy()
    ngs = ngs.dropna(subset=["avg_separation", "avg_intended_air_yards", "avg_cushion"])
    X = np.column_stack([np.ones(len(ngs)), ngs.avg_intended_air_yards, ngs.avg_cushion])
    ngs["soe_ngs"] = ngs.avg_separation - X @ np.linalg.lstsq(X, ngs.avg_separation, rcond=None)[0]
    ids = pd.read_csv(fetch(PLAYERS_URL, "nflverse_players.csv", refresh), usecols=["gsis_id", "nfl_id"],
                      low_memory=False).dropna()
    ids["nfl_id"] = pd.to_numeric(ids.nfl_id, errors="coerce")
    ngs = ngs.merge(ids.dropna(), left_on="player_gsis_id", right_on="gsis_id", how="left")
    return ngs, len(X)


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    vals = [spearman(x[i], y[i]) for i in rng.integers(0, n, (B, n))]
    return spearman(x, y), *np.nanpercentile(vals, [2.5, 97.5]), n


def partial_rank(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = [v - np.polyval(np.polyfit(rz, v, 1), rz) for v in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


def freeze():
    if PRED.exists():
        raise SystemExit(f"{PRED} already exists. Frozen predictions are never overwritten.")
    run_id = load_run_id()
    RP = read_out("route_players.csv", run_id)
    w = RP[(RP.pgm == "WR") & (RP.routes >= MIN_ROUTES)].copy()
    # Game Juice+ and draft year from their own tables: route_players.csv only carries them for players who also
    # have a Combine Juice+ score.
    GJ = read_out("game_juice_scores.csv", run_id).set_index("nfl_id")
    w["game_juice"] = w.nfl_id.map(GJ.game_juice)
    w["draft_year"] = w.nfl_id.map(pd.read_csv(get_file("players.csv")).set_index("nfl_id").draft_year)
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    w["pred_prethrow"] = (z(w.v_cut) + z(w.peak_pre)) / 2
    w = w.rename(columns={"sep": "pred_sep_2325", "game_juice": "pred_game_juice",
                          "juice_final": "pred_combine_juice"})
    cols = ["nfl_id", "display_name", "draft_year", "routes"] + list(PREDICTORS)
    w = w[cols].sort_values("nfl_id")
    w.insert(0, "frozen_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    w.insert(0, "run_id", run_id)
    FWD.mkdir(exist_ok=True)
    w.to_csv(PRED, index=False, float_format="%.10g")
    sha = hashlib.sha256(PRED.read_bytes()).hexdigest()
    (FWD / "forward_predictions_2026.sha256").write_text(f"{sha}  {PRED.name}\n")
    print(f"Froze {len(w)} WR predictions from run {run_id} to {PRED} (sha256 {sha[:16]}...). Commit it now.")


def check_ngs(refresh):
    """Outcome validity on 2023-25: NGS SOE vs the in-data route-adjusted separation, same player-seasons."""
    run_id = load_run_id()
    SP = read_out("soe_players.csv", run_id)
    rows = []
    for season in (2023, 2024, 2025):
        ngs, n_all = ngs_soe(season, refresh)
        m = ngs[ngs.targets >= MIN_TARGETS].merge(SP[(SP.season == season) & (SP.pgm == "WR")], on="nfl_id")
        if len(m) < 5:
            print(f"  {season}: {len(m)} cohort WRs with {MIN_TARGETS}+ NGS targets; skipped")
            continue
        rows.append(m.assign(season=season))
        print(f"  {season}: NGS fit on {n_all} receivers; {len(m)} cohort WR seasons with {MIN_TARGETS}+ targets")
        refresh = False
    m = pd.concat(rows)
    r_soe, r_sep = boot(m.soe_ngs, m.soe), boot(m.soe_ngs, m.sep)
    print(f"NGS separation over expected vs in-data SOE: {r_soe[0]:+.2f} [{r_soe[1]:+.2f}, {r_soe[2]:+.2f}] n={r_soe[3]}")
    print(f"NGS separation over expected vs in-data separation (raw): {r_sep[0]:+.2f} [{r_sep[1]:+.2f}, {r_sep[2]:+.2f}]")
    FWD.mkdir(exist_ok=True)
    pd.DataFrame([{"comparison": "NGS SOE vs in-data SOE", "rho": r_soe[0], "ci_low": r_soe[1], "ci_high": r_soe[2],
                   "n": r_soe[3]},
                  {"comparison": "NGS SOE vs in-data separation", "rho": r_sep[0], "ci_low": r_sep[1],
                   "ci_high": r_sep[2], "n": r_sep[3]}]).assign(run_id=run_id).to_csv(
        FWD / "ngs_outcome_check_2023_25.csv", index=False, float_format="%.10g")


def score(force, season=2026):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today < SCORE_AFTER and not force:
        raise SystemExit(f"Scoring is pre-registered for the latest release before 2027-01-01; it is {today}. "
                         f"Pass --force only if you mean to look early (and say so in the writeup).")
    if not PRED.exists():
        raise SystemExit("No frozen predictions. Run --freeze first (before 2026 outcomes are seen).")
    sha = hashlib.sha256(PRED.read_bytes()).hexdigest()
    rec = (FWD / "forward_predictions_2026.sha256").read_text().split()[0]
    if sha != rec:
        raise SystemExit("forward_predictions_2026.csv does not match its recorded sha256. It was edited.")
    pred = pd.read_csv(PRED)
    ngs, n_all = ngs_soe(season, refresh=True)
    last_week = int(pd.read_csv(CACHE / "ngs_receiving.csv.gz", low_memory=False)
                    .query("season == @season and week > 0").week.max())
    m = pred.merge(ngs[ngs.targets >= MIN_TARGETS][["nfl_id", "targets", "soe_ngs"]], on="nfl_id")
    lines = [f"{season} forward test, predictions frozen {pred.frozen_utc.iloc[0]} from run {pred.run_id.iloc[0]}",
             f"NGS {season} through week {last_week}: expected separation fit on {n_all} receivers; "
             f"{len(m)} cohort WRs with {MIN_TARGETS}+ targets"]
    rows = []
    for c, lab in PREDICTORS.items():
        r = boot(m[c], m.soe_ngs)
        rows.append({"predictor": lab, "rho": r[0], "ci_low": r[1], "ci_high": r[2], "n": r[3]})
        lines.append(f"  {lab:62s} {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] n={r[3]}")
    t = m.dropna(subset=["pred_prethrow", "pred_sep_2325", "soe_ngs"])
    pr = partial_rank(t.pred_prethrow, t.soe_ngs, t.pred_sep_2325)
    bs = [partial_rank(*(t.iloc[i][c].to_numpy() for c in ("pred_prethrow", "soe_ngs", "pred_sep_2325")))
          for i in rng.integers(0, len(t), (B, len(t)))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    rows.append({"predictor": "(b) holding (a)", "rho": pr, "ci_low": lo, "ci_high": hi, "n": len(t)})
    lines.append(f"  {'(b) holding (a), partial Spearman':62s} {pr:+.2f} [{lo:+.2f}, {hi:+.2f}] n={len(t)}")
    passed = rows[1]["ci_low"] > 0
    lines.append(f"F1 {'PASS' if passed else 'not shown'}: (b) interval above 0")
    pd.DataFrame(rows).assign(through_week=last_week, passed_f1=passed).to_csv(
        FWD / f"forward_results_{season}.csv", index=False, float_format="%.10g")
    (FWD / f"forward_report_{season}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--check-ngs", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true", help="score before SCORE_AFTER")
    ap.add_argument("--refresh", action="store_true", help="re-download nflverse files for --check-ngs")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.check_ngs:
        check_ngs(a.refresh)
    else:
        score(a.force)
