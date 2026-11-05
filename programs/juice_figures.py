"""Figures for the writeup (audit item 8). Run after juice_transfer.py, on the same run ID.

Six figures, every number from one run (the run ID is printed in each caption line of figure_captions.txt):
  fig1_chain.png            The chain: Combine -> game movement (link 1) and game movement -> result (link 2), by position
  fig2_asp_overlay.png      Same physics on both sides: one player's Combine A-S line over his game A-S line
  fig3_tracking_adds.png    What each tracked piece adds beyond the 40, by position, with 95% intervals
  fig4_snap_takeover.png    A Combine is worth N snaps: how fast rookie game data overtakes the Combine
  fig5_tiers.png            Tiers, not ranks: share elite in games by Combine Juice+ tier, with Wilson intervals
  fig6_prospect_card.png    One prospect card: Combine pieces next to game pieces

Palette: Okabe-Ito (color-blind safe), as in the original notebook. Blue = Combine, vermillion = games,
gray = reference (the stopwatch). Green is never placed next to gray (they fail the protan/deutan check).
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from juice_common import FIG, OUT, load_run_id, read_out  # noqa: E402

RUN_ID = load_run_id()
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#999999"
INK, INK2, GRID = "#1a1a1a", "#555555", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 200,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})
CAPTIONS = []
GROUP_NAME = {"WR": "Wide receivers", "TE": "Tight ends", "DB": "Defensive backs", "OL": "Offensive line",
              "DL": "Interior DL", "EDGE": "Edge rushers"}


def finish(fig, name, caption):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CAPTIONS.append(f"{name}: {caption} (run {RUN_ID})")
    print("saved", FIG / name)


# ---------------------------------------------------------------------
# Figure 1: the chain
# ---------------------------------------------------------------------
CHAIN = read_out("chain_links.csv", RUN_ID)
LEAD_OUTCOME = {"WR": "Separation at throw", "TE": "Separation at throw", "DB": "EPA allowed",
                "DL": "Get-off", "EDGE": "Get-off", "OL": "Peak pressure"}
TRAIT_LABEL = {"T": "Speed", "E": "Burst", "A": "Agility"}
order = ["WR", "DB", "TE", "EDGE", "DL", "OL"]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, constrained_layout=True)
for ax, grp in zip(axes.flat, order):
    d = CHAIN[(CHAIN.group == grp) & CHAIN.trait.isin(TRAIT_LABEL) &
              CHAIN.outcome.str.startswith(LEAD_OUTCOME[grp])].copy()
    d["label"] = d.trait.map(TRAIT_LABEL)
    d = d.set_index("label").reindex(["Speed", "Burst", "Agility"])
    yy = np.arange(len(d))
    for off, col, lo, hi, color, name in ((0.14, "link1_combine_to_game", "l1_lo", "l1_hi", BLUE, "Combine → game"),
                                          (-0.14, "link2_game_to_result", "l2_lo", "l2_hi", VERM, "Game → result")):
        ok = d[col].notna()
        ax.hlines(yy[ok] + off, d.loc[ok, lo], d.loc[ok, hi], color=color, lw=2, alpha=0.9)
        ax.plot(d.loc[ok, col], yy[ok] + off, "o", color=color, ms=7, mec="white", mew=1.5, label=name)
    for i, (lab, r) in enumerate(d.iterrows()):
        if pd.isna(r.link1_combine_to_game):
            ax.text(0, i, "no agility score\n(drills fail the 0.5 gate)", ha="center", va="center", fontsize=8,
                    color=INK2)
    ax.axvline(0, color="#888888", lw=1)
    ax.set_yticks(yy, d.index)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.75, 0.95)
    n1 = int(d.l1_n.max()) if d.l1_n.notna().any() else 0
    ax.set_title(f"{GROUP_NAME[grp]}", loc="left", pad=18)
    ax.text(0.0, 1.02, f"result: {LEAD_OUTCOME[grp].lower()} · n≈{n1}", transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="bottom", ha="left")
    ax.grid(axis="y", visible=False)
for ax in axes[-1]:
    ax.set_xlabel("Spearman correlation (95% interval)")
h, l_ = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=10)
fig.suptitle("Where does the juice leak?", x=0.01, ha="left", y=1.12, fontsize=14, fontweight="bold")
finish(fig, "fig1_chain.png",
       "Link 1 is the Combine piece vs the same piece in games; link 2 is the game piece vs the football result. "
       "A link whose interval crosses 0 is where the chain breaks.")

# ---------------------------------------------------------------------
# Figure 2: A-S overlay for one player
# ---------------------------------------------------------------------
try:
    ASP = read_out("asp_profiles.csv", RUN_ID)
    PTS = read_out("asp_points.csv", RUN_ID)
    J = read_out("juice_player_table.csv", RUN_ID)
    cand = (ASP[ASP.scope == "game_reg"].dropna(subset=["a0"]).merge(
        ASP[ASP.scope == "combine_all"][["nfl_id", "a0", "v0", "r2"]].dropna(), on="nfl_id", suffixes=("_g", "_c"))
        .merge(J[["nfl_id", "npl"]], on="nfl_id"))
    cand = cand[(cand.pgm == "WR") & (cand.r2_g > 0.9) & (cand.r2_c > 0.7)].sort_values("npl", ascending=False)
    pid = int(cand.iloc[0].nfl_id)
    r = cand.iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for scope, color, name, a0, v0 in (("combine_all", BLUE, "Combine drills", r.a0_c, r.v0_c),
                                       ("game_reg", VERM, "NFL games", r.a0_g, r.v0_g)):
        p = PTS[(PTS.nfl_id == pid) & (PTS.scope == scope)]
        ax.scatter(p.speed, p.acc, s=14, color=color, alpha=0.35, linewidths=0)
        xs = np.linspace(0, v0, 50)
        ax.plot(xs, a0 - a0 / v0 * xs, color=color, lw=2.2,
                label=f"{name}: first step a0 = {a0:.1f} yd/s², top gear v0 = {v0:.1f} yd/s")
    ax.axvspan(0, 3.28, color="#f2f2f2", zorder=0)
    ax.text(1.64, ax.get_ylim()[1] * 0.95, "below 3 m/s:\nnot used in the fit", ha="center", va="top",
            fontsize=8, color=INK2)
    ax.set_xlim(0, max(r.v0_c, r.v0_g) * 1.05)
    ax.set_ylim(0, None)
    ax.set_xlabel("Speed (yd/s)")
    ax.set_ylabel("Acceleration (yd/s²)")
    ax.set_title(f"Same physics on both sides: {r.display_name}", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    finish(fig, "fig2_asp_overlay.png",
           f"Top two accelerations per 0.2 m/s speed bin across every frame, with the fitted acceleration-speed "
           f"line. {r.display_name}, {int(r.npl)} usable NFL plays.")
except Exception as e:  # noqa: BLE001
    print("fig2 skipped:", e)

# ---------------------------------------------------------------------
# Figure 3: what tracking adds beyond the 40
# ---------------------------------------------------------------------
ADDS = read_out("tracking_adds.csv", RUN_ID)
A = ADDS[(ADDS.holding == "the 40") & (ADDS.target == "Game Juice+")]
pieces = [("Speed (T, fastest 10 yd)", "Speed"), ("Burst (E, yard 2-10 split)", "Burst"),
          ("Agility (A, position drills)", "Agility")]
rows = ["ALL", "WR + DB", "WR", "DB", "TE", "EDGE", "DL", "OL"]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, constrained_layout=True)
for ax, (pname, short) in zip(axes, pieces):
    d = A[A.piece == pname].set_index("group").reindex(rows)
    yy = np.arange(len(rows))
    clear = d.ci_low > 0
    for i, (g, r) in enumerate(d.iterrows()):
        if pd.isna(r.partial_r):
            ax.text(0, i, "no score", ha="center", va="center", fontsize=8, color=INK2)
            continue
        color = BLUE if clear[g] else GRAY
        ax.hlines(i, r.ci_low, r.ci_high, color=color, lw=2)
        ax.plot(r.partial_r, i, "o", color=color, ms=8 if g in ("ALL", "WR + DB") else 6.5, mec="white", mew=1.5)
        if clear[g]:
            ax.text(r.ci_high + 0.03, i, f"{r.partial_r:+.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color="#888888", lw=1)
    ax.axhline(1.5, color=GRID, lw=1)
    ax.set_title(short, loc="left")
    ax.set_xlim(-0.65, 0.8)
    ax.set_yticks(yy, [f"{g} (n={int(n)})" if np.isfinite(n) else g for g, n in zip(rows, d.n)])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Partial correlation with Game Juice+,\nholding the official 40")
fig.suptitle("Speed you can get from a stopwatch. Agility is where tracking earns its keep.", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
finish(fig, "fig3_tracking_adds.png",
       "Partial correlation on within-group percentile ranks; 95% bootstrap intervals. Blue = interval above 0. "
       "EDGE and TE have no agility score because their drills disagree with each other (alpha under 0.5).")

# ---------------------------------------------------------------------
# Figure 4: snap takeover
# ---------------------------------------------------------------------
TAKE = read_out("snap_takeover.csv", RUN_ID)
mets = [m for m in ["game top speed", "game agility", "game burst"] if m in set(TAKE.metric)]
fig, axes = plt.subplots(1, len(mets), figsize=(4 * len(mets), 3.9), sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes)
for ax, m in zip(axes, mets):
    d = TAKE[TAKE.metric == m]
    c0 = d.r2_combine_only.iloc[0]
    ax.axhline(c0, color=GRAY, lw=1.5, ls=(0, (4, 3)))
    ax.text(250, c0 - 0.008, "Combine only", color=INK2, fontsize=8.5, ha="right", va="top")
    ax.plot(d.k, d.r2_game_only, color=VERM, lw=2, marker="o", ms=5, mec="white", label="First k plays only")
    ax.plot(d.k, d.r2_both, color=BLUE, lw=2, marker="o", ms=5, mec="white", label="First k plays + Combine")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_xlim(0.8, 260)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_title(m[0].upper() + m[1:] + f" (n={int(d.players.iloc[0])})", loc="left")
    ax.set_xlabel("Rookie snaps seen (k)")
axes[0].set_ylabel("Out-of-class R², rest of rookie season")
h, l_ = axes[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.08))
fig.suptitle("The Combine keeps adding to top speed; for agility and burst, game film wins at once", x=0.01,
             ha="left", y=1.14, fontsize=13, fontweight="bold")
finish(fig, "fig4_snap_takeover.png",
       "Target: the 95th percentile of each rookie's plays after his 100th. Leave-one-class-out R². "
       "The gap between the blue and vermillion lines is what the Combine adds once k plays are on film.")

# ---------------------------------------------------------------------
# Figure 5: tiers
# ---------------------------------------------------------------------
T = read_out("tiers.csv", RUN_ID)
fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
x = np.arange(len(T))
ax.bar(x, 100 * T.elite_share, width=0.55, color=BLUE, edgecolor="white", linewidth=2)
ax.errorbar(x, 100 * T.elite_share, yerr=[100 * (T.elite_share - T.elite_lo), 100 * (T.elite_hi - T.elite_share)],
            fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
for i, r in T.iterrows():
    ax.text(i, 100 * r.elite_hi + 1.5, f"{int(r.elite_in_games)} of {int(r.n)}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [f"Combine Juice+ {t}" for t in T.tier])
ax.set_ylabel("Share in the top 16% in games (%)")
ax.set_ylim(0, max(100 * T.elite_hi) + 10)
ax.grid(axis="x", visible=False)
ax.set_title("Tiers, not ranks: low Combine scores rarely turn elite", loc="left")
finish(fig, "fig5_tiers.png", "Players with a Game Juice+ score. Bars: share in the top 16% of Game Juice+ within "
       "position (about +1 SD); whiskers: 95% Wilson interval.")

# ---------------------------------------------------------------------
# Figure 6: prospect card (late-round 110+ player with the highest game score)
# ---------------------------------------------------------------------
CJ = read_out("combine_juice_scores.csv", RUN_ID)
GJ = read_out("game_juice_scores.csv", RUN_ID)
card = CJ.merge(GJ[["nfl_id", "game_juice", "g_top_plus", "g_expl_plus", "g_agil_plus", "npl"]], on="nfl_id")
card["pick"] = card.draft_overall_pick.fillna(260)
pool = card[(card.pick >= 140) & (card.juice_final >= 110)].sort_values("game_juice", ascending=False)
if len(pool):
    r = pool.iloc[0]
    labels = ["Speed", "Burst", "Agility", "Juice+"]
    comb = [r.T_trk_plus, r.E_trk_plus, r.A_trk_plus, r.juice_final]
    game = [r.g_top_plus, r.g_expl_plus, r.g_agil_plus, r.game_juice]
    fig, ax = plt.subplots(figsize=(7, 3.9), constrained_layout=True)
    y = np.arange(len(labels))
    for vals, off, color, name in ((comb, -0.18, BLUE, "Combine"), (game, 0.18, VERM, "NFL games")):
        v = np.array(vals, float)
        ok = np.isfinite(v)
        ax.barh(y[ok] + off, v[ok] - 100, left=100, height=0.34, color=color, edgecolor="white", linewidth=2,
                label=name)
        for yi, vi in zip(y[ok] + off, v[ok]):
            ax.text(vi + (1 if vi >= 100 else -1), yi, f"{vi:.0f}", va="center", ha="left" if vi >= 100 else "right",
                    fontsize=9, color=INK)
    ax.axvline(100, color="#888888", lw=1)
    lo_v = np.nanmin(np.r_[comb, game].astype(float))
    ax.set_xlim(min(lo_v - 6, 94), None)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score (100 = position average)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=1)
    pick_txt = "undrafted" if r.pick >= 260 else f"pick {int(r.pick)}"
    ax.set_title(f"{r.display_name} · {r.nfl_position} · {int(r.draft_year)} · {pick_txt}", loc="left")
    finish(fig, "fig6_prospect_card.png",
           f"Late-round (pick 140+ or undrafted) player with Combine Juice+ 110+ and the highest Game Juice+. "
           f"{int(r.npl)} usable plays. Pieces with no bar have no score for this position.")

(OUT / "figure_captions.txt").write_text("\n".join(CAPTIONS) + "\n")
print(f"\nSaved {len(CAPTIONS)} figures to {FIG.resolve()} (run {RUN_ID})")
