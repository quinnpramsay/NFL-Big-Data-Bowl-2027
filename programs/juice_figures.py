"""Figures for the writeup (audit item 8). Run last (after juice_routes.py), on the same run ID.

Nine figures, every number from one run (the run ID is printed in each caption line of figure_captions.txt):
  fig1_chain.png            The chain: Combine -> game movement (link 1) and game movement -> result (link 2), by position
  fig2_asp_overlay.png      Same physics on both sides: one player's Combine A-S line over his game A-S line
  fig3_tracking_adds.png    What each tracked piece adds beyond the 40, by position, with 95% intervals
  fig4_snap_takeover.png    A Combine is worth N snaps: how fast rookie game data overtakes the Combine
  fig5_tiers.png            Tiers, not ranks: share elite in games by Combine Juice+ tier, with Wilson intervals
  fig6_prospect_card.png    One prospect card: Combine pieces next to game pieces
  fig7_separation_chain.png The WR chain in yards and points: Combine -> game movement -> separation -> EPA
  fig8_route_example.png    One route, two receivers: high vs low cut speed, colored by speed, to the throw
  fig9_year2_slope.png      Year-2 game agility: players flagged below their Combine read vs everyone else

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
EXPLORATORY_PANELS = {"OL"}   # the OL burst chain is a lead, not a confirmed chain: drawn in gray
for ax, grp in zip(axes.flat, order):
    d = CHAIN[(CHAIN.group == grp) & CHAIN.trait.isin(TRAIT_LABEL) &
              CHAIN.outcome.str.startswith(LEAD_OUTCOME[grp])].copy()
    d["label"] = d.trait.map(TRAIT_LABEL)
    d = d.set_index("label").reindex(["Speed", "Burst", "Agility"])
    yy = np.arange(len(d))
    expl = grp in EXPLORATORY_PANELS
    for off, col, lo, hi, color, name in ((0.14, "link1_combine_to_game", "l1_lo", "l1_hi", BLUE, "Combine → game"),
                                          (-0.14, "link2_game_to_result", "l2_lo", "l2_hi", VERM, "Game → result")):
        ok = d[col].notna()
        color = GRAY if expl else color
        ax.hlines(yy[ok] + off, d.loc[ok, lo], d.loc[ok, hi], color=color, lw=2, alpha=0.9)
        ax.plot(d.loc[ok, col], yy[ok] + off, "o", color=color, ms=7, mec="white", mew=1.5,
                label=None if expl else name)
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
    ax.text(0.0, 1.02, f"result: {LEAD_OUTCOME[grp].lower()} · n≈{n1}" + (" · exploratory" if expl else ""),
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom", ha="left")
    ax.grid(axis="y", visible=False)
for ax in axes[-1]:
    ax.set_xlabel("Spearman correlation (95% interval)")
h, l_ = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=10)
fig.suptitle("Where does the juice leak?", x=0.01, ha="left", y=1.12, fontsize=14, fontweight="bold")
finish(fig, "fig1_chain.png",
       "Link 1 is the Combine piece vs the same piece in games; link 2 is the game piece vs the football result. "
       "A link whose interval crosses 0 is where the chain breaks. Offensive line in gray: exploratory, not a "
       "confirmed chain.")

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
    ax.text(1.64, ax.get_ylim()[1] * 0.95, "below 3.3 yd/s (3 m/s):\nnot used in the fit", ha="center", va="top",
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
pieces = [("Speed (T, fastest 10 yd)", "Speed"), ("Burst (E, yard 2-10 split + accel)", "Burst"),
          ("Agility (A, position drills)", "Agility")]
rows = ["ALL", "WR + DB", "WR", "DB", "TE", "EDGE", "DL", "OL"]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, constrained_layout=True)
NO_AGILITY = {"EDGE", "TE"}   # drills fail the 0.5 agreement gate, so there is no agility score at all
for ax, (pname, short) in zip(axes, pieces):
    d = A[A.piece == pname].set_index("group").reindex(rows)
    yy = np.arange(len(rows))
    clear = d.ci_low > 0
    for i, (g, r) in enumerate(d.iterrows()):
        if pd.isna(r.partial_r):
            why = "no agility score (drills disagree)" if short == "Agility" and g in NO_AGILITY else \
                "too few players (n < 25)"
            ax.text(0.075, i, why, ha="center", va="center", fontsize=7.5, color=INK2)
            continue
        # n inside each panel: the panels share a y axis, so tick labels can only carry one panel's n.
        ax.text(0.94, i, f"n={int(r.n)}", ha="right", va="center", fontsize=7.5, color=INK2)
        color = BLUE if clear[g] else GRAY
        ax.hlines(i, r.ci_low, r.ci_high, color=color, lw=2)
        ax.plot(r.partial_r, i, "o", color=color, ms=8 if g in ("ALL", "WR + DB") else 6.5, mec="white", mew=1.5)
        if clear[g]:
            ax.text(r.ci_high + 0.03, i, f"{r.partial_r:+.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color="#888888", lw=1)
    ax.axhline(1.5, color=GRID, lw=1)
    ax.set_title(short, loc="left")
    ax.set_xlim(-0.65, 0.95)
    ax.set_yticks(yy, rows)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Partial correlation with Game Juice+,\nholding the official 40")
fig.suptitle("Speed you can get from a stopwatch. Agility is where tracking earns its keep.", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
finish(fig, "fig3_tracking_adds.png",
       "Partial correlation on within-group percentile ranks; 95% bootstrap intervals. Blue = interval above 0. "
       "EDGE and TE have no agility score because their drills disagree with each other (alpha under 0.5); "
       "pieces with fewer than 25 scored players are not estimated. n is printed in each panel.")

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
    if c0 < 0:
        ax.text(0.03, 0.04, f"Combine only: R² = {c0:.2f} (below 0 = worse than\nguessing the class average)",
                transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
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
# Figure 6: prospect card for the lead story (WR/DB agility). Rule: late round (pick 140+ or undrafted), WR or DB,
# 100+ usable plays, highest tracked Combine agility. Chosen as a case study, not a test.
# ---------------------------------------------------------------------
CJ = read_out("combine_juice_scores.csv", RUN_ID)
GJ = read_out("game_juice_scores.csv", RUN_ID)
card = CJ.merge(GJ[["nfl_id", "game_juice", "g_top_plus", "g_expl_plus", "g_agil_plus", "npl"]], on="nfl_id")
card["pick"] = card.draft_overall_pick.fillna(260)
pool = card[(card.pick >= 140) & card.pgm.isin(["WR", "DB"]) & (card.npl >= 100) & card.A_trk_plus.notna()]
pool = pool.sort_values("A_trk_plus", ascending=False)
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
    hi_v = np.nanmax(np.r_[comb, game].astype(float))
    ax.set_xlim(min(lo_v - 6, 94), max(hi_v + 5, 106))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score (100 = position average)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, frameon=False)
    pick_txt = "undrafted" if r.pick >= 260 else f"pick {int(r.pick)}"
    ax.set_title(f"{r.display_name} · {r.nfl_position} · {int(r.draft_year)} · {pick_txt}", loc="left")
    finish(fig, "fig6_prospect_card.png",
           f"Late-round (pick 140+ or undrafted) WR or DB with 100+ usable plays and the highest tracked Combine "
           f"agility. Official 40: {r.forty:.2f} s. {int(r.npl)} usable plays. Case study, not a test.")

# ---------------------------------------------------------------------
# Figure 7: the WR separation chain in yards and points (from juice_routes.py)
# ---------------------------------------------------------------------
try:
    RC = read_out("route_chain.csv", RUN_ID)
    RV = read_out("separation_value.csv", RUN_ID).set_index("outcome")
    RP = read_out("route_players.csv", RUN_ID)
    rows_ = RC[RC.preregistered.astype(bool)].reset_index(drop=True)
    epa_yd = RV.loc["EPA per target", "per_yard"]
    sd_sep = RP[(RP.pgm == "WR") & (RP.routes >= 50)].sep.std()
    fig, ax = plt.subplots(figsize=(11, 0.5 + 1.45 * len(rows_)), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(rows_) - 0.45)
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.45", fc="white", ec="#bbbbbb", lw=1.2)
    xs_ = [0.9, 3.6, 6.3, 8.9]
    for i, r in rows_.iterrows():
        y = len(rows_) - 1 - i
        labels = [f"Combine\n{r.combine_piece.replace('Combine ', '')}", f"Game\n{r.game_piece}", "Separation\nat the throw",
                  "Points\n(EPA per target)"]
        for xx, lab, col in zip(xs_, labels, (BLUE, VERM, INK, INK)):
            ax.text(xx, y, lab, ha="center", va="center", fontsize=9.5, color=col, bbox=box)
        arrows = [(f"{r.a_sd_per_sd:+.2f} SD per SD\n[{r.a_lo:+.2f}, {r.a_hi:+.2f}]", 0),
                  (f"{r.b_yd_per_sd:+.3f} yd per SD\n[{r.b_lo:+.3f}, {r.b_hi:+.3f}]", 1),
                  (f"{epa_yd:+.3f} EPA per yd", 2)]
        for lab, k in arrows:
            ax.annotate("", xy=(xs_[k + 1] - 0.95, y), xytext=(xs_[k] + 0.95, y),
                        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.4))
            ax.text((xs_[k] + xs_[k + 1]) / 2, y + 0.2, lab, ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.text(xs_[0] - 0.85, y - 0.42, f"End to end, per Combine SD: {r.ab_yd:+.3f} yd [{r.ab_lo:+.3f}, "
                f"{r.ab_hi:+.3f}] = {r.epa_per_100:+.2f} EPA per 100 targets [{r.epa_lo:+.2f}, {r.epa_hi:+.2f}]"
                f" · n = {int(r.n)} WRs", ha="left", va="center", fontsize=9, color=INK)
    holds = bool((rows_.a_lo > 0).all() and (rows_.b_lo > 0).all() and (rows_.ab_lo > 0).all())
    ax.set_title("Each link holds. End to end, the Combine is worth a small, measurable amount of separation"
                 if holds else "The WR chain in yards and points: where it holds and where it breaks", loc="left")
    finish(fig, "fig7_separation_chain.png",
           f"WRs with 50+ routes. Movement and separation measured from the snap to the throw, adjusted for route, "
           f"man/zone and time to throw. Joint bootstrap over players. For scale, 1 SD of WR separation is "
           f"{sd_sep:.2f} yd, worth {100 * sd_sep * epa_yd:.1f} EPA per 100 targets.")
except Exception as e:  # noqa: BLE001
    print("fig7 skipped:", e)

# ---------------------------------------------------------------------
# Figure 8: one route, two receivers (from juice_routes.py; an illustration, not a test)
# ---------------------------------------------------------------------
try:
    from matplotlib.collections import LineCollection
    EXR = read_out("route_example.csv", RUN_ID)
    labs = ["high cut speed", "low cut speed"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharex=True, sharey=True, constrained_layout=True)
    norm = plt.Normalize(0, max(9.0, EXR.s.max()))
    for ax, lab in zip(axes, labs):
        e = EXR[EXR.label == lab].sort_values("frame")
        lat, down = e.y.to_numpy(), e.x.to_numpy()
        if lat[-1] - lat[max(0, len(lat) - 6)] < 0:
            lat = -lat              # mirror so both routes finish moving to the right (break to the same side)
        pts = np.column_stack([lat, down]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="cividis", norm=norm, lw=5)
        lc.set_array(e.s.to_numpy()[:-1])
        ax.add_collection(lc)
        ax.plot(lat[0], down[0], "o", color=INK, ms=6)
        ax.plot(lat[-1], down[-1], "s", color=INK, ms=6)
        r = e.iloc[0]
        ax.text(lat[-1], down[-1] + 0.8, f"throw: {r.sep:.1f} yd of separation", ha="center", fontsize=8.5,
                color=INK)
        ax.set_title(f"{r.display_name}\n{lab}: {r.v_cut:.1f} yd/s through the cut", loc="left", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Across the field (yd)")
    axes[0].set_ylabel("Downfield from the snap spot (yd)")
    for ax in axes:
        ax.autoscale()
        ax.margins(0.25)
    cb = fig.colorbar(lc, ax=axes, shrink=0.8, label="Speed (yd/s)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Same route ({EXR.route.iloc[0].lower()}), snap to throw", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    finish(fig, "fig8_route_example.png",
           "Rule: the WRs (50+ routes) at the 90th and 10th percentile of route-adjusted cut speed, on the break "
           "route both ran most; for each, the play with a 45+ degree cut closest to his own median cut speed and "
           "separation on that route. Circle = snap, square = throw. An illustration, not a test.")
except Exception as e:  # noqa: BLE001
    print("fig8 skipped:", e)

# ---------------------------------------------------------------------
# Figure 9: year-2 game agility, flagged players vs everyone else (from juice_transfer.py section F)
# ---------------------------------------------------------------------
try:
    UJP = read_out("unused_juice_players.csv", RUN_ID)
    u = UJP[UJP.metric == "game agility"]
    UJS = read_out("unused_juice.csv", RUN_ID).set_index("metric").loc["game agility"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for flag, color, lw, alpha, name in ((False, GRAY, 0.8, 0.35, "everyone else"),
                                         (True, BLUE, 1.2, 0.8, "flagged: Combine agility 1+ SD above year 1")):
        g = u[u.flagged.astype(bool) == flag]
        for _, r in g.iterrows():
            ax.plot([0, 1], [r.y1, r.y2], color=color, lw=lw, alpha=alpha)
        ax.plot([0, 1], [g.y1.mean(), g.y2.mean()], color=VERM if flag else INK, lw=3.2, marker="o", ms=7,
                mec="white", label=f"{name} (n={len(g)}): mean change {g.y2.mean() - g.y1.mean():+.2f} SD")
    ax.set_xticks([0, 1], ["Rookie season", "Year 2"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel("Game agility (size-free z within position)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    finish(fig, "fig9_year2_slope.png",
           f"Players with 50+ plays in both seasons. Holding year 1, the Combine agility coefficient for year 2 is "
           f"{UJS.beta_combine:+.2f} [{UJS.ci_low:+.2f}, {UJS.ci_high:+.2f}], n={int(UJS.n)}. Bold lines are group "
           f"means; the flagged group's rise is partly regression to the mean, which the coefficient accounts for.")
except Exception as e:  # noqa: BLE001
    print("fig9 skipped:", e)

(OUT / "figure_captions.txt").write_text("\n".join(CAPTIONS) + "\n")
print(f"\nSaved {len(CAPTIONS)} figures to {FIG.resolve()} (run {RUN_ID})")
