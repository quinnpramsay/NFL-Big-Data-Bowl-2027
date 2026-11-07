"""Builds notebooks/Juice+_remake.ipynb: the original Juice+ notebook layout, rebuilt on the current pipeline run."""
import sys
import nbformat as nbf

OUT_PATH = sys.argv[1]
HIDE = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c.metadata.update(HIDE)
    cells.append(c)


# ---------------------------------------------------------------------------------------------------------------------
md(r'''
# Juice+: Finding the Extra Gear in the "Underwear Olympics"

Scouts say a player has **juice** when he has an extra gear the players around him don't. It's the receiver who runs past a corner who had him covered, or the safety who closes on a ball that looked out of reach. It's the movement that **pops off the screen**.

The Combine is often mocked as the **"Underwear Olympics"**. Prospects run and jump in shorts, with no pads and no opponent, so many evaluators doubt the drills say much about football. We agree they don't tell you who can *play*. We ask a narrower question a front office can act on:

> **If a team wants to add juice to its roster, does the way a prospect moves in Combine drills show up in how he moves in NFL games? And what can tracking see that a stopwatch can't?**

**Juice+** answers with one number, built from the Combine's 10 Hz tracking and rebuilt the same way from NFL game tracking.

**Key findings (270 rookies with both scores, 2023 to 2025 draft classes):**
- **Combine juice shows up on Sundays.** Players at 110+ became elite movers in games **33%** of the time. Under 90, only **1 of 39** did.
- **For speed, the stopwatch already knows.** Juice+ ties a size-adjusted 40 on game movement and clearly beats NGS and RAS.
- **Agility is what the 40 misses.** For receivers and defensive backs, tracked drill agility predicts how hard a player cuts in games even after his 40 is known (**+0.33**).
- **Agility predicts growth.** Receivers and DBs whose drill agility ran ahead of their rookie play improved in year 2. The 40 predicted none of it.
''')

md(r'''
## 1. What Juice+ measures

Juice+ borrows from baseball's **OPS+**. Compare a player with his peers, remove context that isn't ability, and report one number where **100 is average**. Here the peers are players at the same position, and the context removed is **body size and draft class**. **Every 10 points is one standard deviation**, so 110 means one standard deviation better than the average player at his position.

It has three pieces, each measured with the same tracking chips at the Combine and in games:

| Piece | Combine Juice+ (tracking) | Game Juice+ | Stopwatch tests averaged in |
|---|---|---|---|
| **Top speed** (the extra gear) | Fastest 10-yard stretch of the 40 | Peak speed on his best plays | 40 time |
| **Burst** (first step) | Time and acceleration from yard 2 to yard 10 of the 40 | The same split, timed from where he stood at the snap | 10-yard split, vertical, broad jump |
| **Agility** (change of direction) | Pace and braking through his position drills | Hardest sideways acceleration on his best plays | 3-cone, shuttle |

Speed and burst are timed the same way in both places. Agility is not. At the Combine it is drill pace, and in games it is how hard a player actually cuts. So when one predicts the other, two different measurements agree, not one formula counted twice.
''')

code(r"""
# Setup: data folder, output folders and libraries. Change DATA if your competition files live somewhere else.
import os, sys, re, json, glob, subprocess, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import spearmanr
from IPython.display import display, Image
warnings.filterwarnings("ignore")

DATA, CACHE = "/Users/ramsay_5/Desktop/NFL Big Data Bowl 2027/nfl-big-data-bowl-2027/", "juice_cache"
WORK = Path.cwd()
if os.path.isdir("/kaggle/input"):
    hits = glob.glob("/kaggle/input/**/players.csv", recursive=True)
    DATA = os.path.dirname(hits[0]) + "/"
    WORK, CACHE = Path("/kaggle/working"), "/kaggle/working/juice_cache"
elif not os.path.isdir(DATA):
    # Fall back to a Kaggle download cache or a data/ folder next to this notebook
    spots = [os.getenv("JUICE_DATA_DIR", ""), "~/.cache/kagglehub/competitions/nfl-big-data-bowl-2027", "data", "../data"]
    hits = [h for s in spots if s for h in sorted(glob.glob(os.path.expanduser(s) + "/**/players.csv", recursive=True))]
    assert hits, "Competition files not found. Set DATA to the folder that holds players.csv."
    DATA = os.path.dirname(hits[0]) + "/"
OUT, CACHE, CODE = WORK / "juice_nb_output", WORK / CACHE, WORK / "juice_code"
CODE.mkdir(exist_ok=True)
os.environ.update(JUICE_DATA_DIR=DATA, JUICE_OUT_DIR=str(OUT), JUICE_CACHE_DIR=str(CACHE))
print("Data:", DATA)
""")

# The pipeline: each script is written to juice_code/ by a hidden cell, then run in order below.
PIPE_DIR = "/Users/quinnramsay/Desktop/NFL-Big-Data-Bowl-2027/programs/"
PIPE = ["juice.py", "juice_common.py", "juice_physics.py", "juice_transfer.py", "juice_predictive.py",
        "juice_casework.py", "juice_routes.py"]
HIDE_ALL = {"_kg_hide-input": True, "_kg_hide-output": True, "jupyter": {"source_hidden": True, "outputs_hidden": True}}
import base64, lzma, json as _json
_blob = base64.b64encode(lzma.compress(_json.dumps({f: open(PIPE_DIR + f).read() for f in PIPE}).encode(), preset=9)).decode()
assert _json.loads(lzma.decompress(base64.b64decode(_blob))) == {f: open(PIPE_DIR + f).read() for f in PIPE}
code("# The Juice+ pipeline (7 scripts), packed into one line so it stays out of the way.\n"
     "# The readable copy of every script is in Appendix B at the end of the notebook.\n"
     "import base64, lzma\n"
     f'PIPELINE = "{_blob}"\n'
     "for _name, _src in json.loads(lzma.decompress(base64.b64decode(PIPELINE))).items():\n"
     "    if not (CODE / _name).exists():\n"
     "        (CODE / _name).write_text(_src)")

code(r"""
# Run the Juice+ pipeline once (about 25 minutes the first time). Later runs reuse the saved outputs.
FORCE_RERUN = False
STEPS = ["juice.py", "juice_physics.py", "juice_transfer.py", "juice_predictive.py", "juice_casework.py",
         "juice_routes.py"]
if FORCE_RERUN or not (OUT / "routes_report.txt").exists():
    log = WORK / "juice_pipeline_log.txt"
    with open(log, "w") as fh:
        for step in STEPS:
            t0 = time.time()
            r = subprocess.run([sys.executable, str(CODE / step)], cwd=str(WORK), stdout=fh, stderr=subprocess.STDOUT)
            if r.returncode:
                print(log.read_text()[-3000:])
                raise RuntimeError(f"{step} failed. Full log: {log}")
            print(f"{step:22s} done in {(time.time() - t0) / 60:.1f} min")
sys.path.insert(0, str(CODE))
from juice_common import get_file  # noqa: E402

RUN = json.loads((OUT / "run_manifest.json").read_text())


def T(name):
    # One table from this run
    return pd.read_csv(OUT / name).drop(columns=["run_id"], errors="ignore")


def report(name):
    return (OUT / name).read_text()


def wilson(k, n, z=1.96):
    p_ = k / n
    den = 1 + z ** 2 / n
    ctr = (p_ + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


# Accessible, colour-blind-safe palette (Okabe-Ito) and readable defaults
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
GROUP_COL = dict(zip(["DB", "DL", "EDGE", "OL", "TE", "WR"], OI))
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
YD_TO_MPH = 3600 / 1760
C_ = RUN["counts"]
print(f"Run {RUN['run_id']}: {C_['combine_tracked']} tracked players, {C_['combine_scored']} Combine scores, "
      f"{C_['game_scored']} game scores, {C_['both']} with both")
""")

md(r'''
## 2. Building Juice+

**Data.** Every 2023 to 2025 rookie with Combine tracking in WR, TE, DB, OL, interior DL and EDGE: 505 players, 414 with every piece.

1. **Top speed and burst** come from the tracked 40, the one drill with the same layout for everyone. Top speed repeats almost perfectly between a player's two attempts (0.91 on a 0 to 1 scale) and matches the official 40 at 0.89. Burst is noisier (0.67).
2. **Agility** comes from each position's drills (bag, route and back-pedal drills) plus the 3-cone and shuttle. It only counts where a group's drills agree with each other. DL, OL, DB and WR pass. EDGE and TE don't, so their Juice+ is speed and burst only.
3. **Size and class are removed.** Within each group, a leave-one-out regression predicts each piece from weight, weight², height and draft class. Juice+ keeps what's left over. A 330-pound guard who moves like a 300-pound guard grades out *positive*.
4. **Noisy pieces are pulled toward average**, then **each tracked piece is averaged 50/50 with the stopwatch tests of the same trait**, since two imperfect measures cancel some noise. The pieces are averaged and put on the 100 scale.

Without step 3, tracked speed is mostly a scale reading. Among defensive linemen, weight and raw top speed have a rank correlation of -0.70 (0 means no link). After the adjustment it is +0.15 (Figure 1). Overall, Juice+ has almost no link to weight (-0.03), while the 40 (-0.34) and a RAS-style score (-0.29) partly reward being light.
''')

code(r'''
# Figure 1: why size normalization matters (defensive line)
CJ = T("combine_juice_scores.csv")
dd = CJ[(CJ.pgm == "DL") & CJ.fly10.notna() & CJ.T_trk_plus.notna()]
r_raw = spearmanr(dd.combine_weight, dd.fly10)[0]
r_adj = spearmanr(dd.combine_weight, dd.T_trk_plus)[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(dd.combine_weight, dd.fly10 * YD_TO_MPH, s=18, color=OI[0])
ax[0].set_title(f"Raw: heavier linemen are slower (rho = {r_raw:.2f})")
ax[0].set_xlabel("Weight (lb)"); ax[0].set_ylabel("Fastest 10 yards (mph)")
ax[1].scatter(dd.combine_weight, dd.T_trk_plus, s=18, color=OI[1])
ax[1].axhline(100, color="grey", lw=0.8, ls="--")
ax[1].set_title(f"Size-normalized: weight no longer matters (rho = {r_adj:+.2f})")
ax[1].set_xlabel("Weight (lb)"); ax[1].set_ylabel("Speed vs. size expectation\n(100 = average)")
fig.suptitle(f"Figure 1. Defensive line top speed before and after removing size (n = {len(dd)})", x=0.01, ha="left")
plt.tight_layout(); plt.show()
''')

md(r'''
## 3. Seeing juice translate

**Game Juice+** rebuilds the same pieces from every 2023 to 2025 regular-season snap:
- **Like is compared with like.** Each play is compared only with plays at the same position, alignment, route and length. Very short snaps and snaps in decided games are dropped.
- **Best efforts count.** Each piece uses a player's 95th-percentile play, so loafing snaps don't drag him down.
- **Burst starts from his spot, not the snap.** Timing yards 2 to 10 from where he stood keeps reaction time out, just like the 40.
- **A player needs 50 usable plays.** The pieces are weighted by how well they repeat. 320 players qualify.

Game movement is a stable trait: split a player's plays in half at random and the halves agree at 0.94 (top speed), 0.89 (agility) and 0.81 (burst).

Figure 2 shows what translation looks like. **Malik Heath** (WR, undrafted, 2023) ran a 4.64 40, in the bottom 5% of receivers. His overall Combine Juice+ was 91, so a 40-first board passes on him. But his drill agility was 119, among the best of any receiver, and in 229 NFL plays his game agility came in above average at 106. Left: his sharpest route drill. Right: his sharpest cut on an NFL route. Bottom: both speed traces on one clock.
''')

code(r'''
# Figure 2: one player in a position drill at the Combine and on a matching NFL snap.
# Rule (no hand-picking): the late-round (pick 140+ or undrafted) WR or DB with 100+ usable plays and the highest
# tracked Combine agility.
GJ = T("game_juice_scores.csv")
card = CJ.merge(GJ[["nfl_id", "game_juice", "g_agil_plus", "npl"]], on="nfl_id")
card["pick"] = card.draft_overall_pick.fillna(260)
row = (card[(card.pick >= 140) & card.pgm.isin(["WR", "DB"]) & (card.npl >= 100) & card.A_trk_plus.notna()]
       .sort_values("A_trk_plus", ascending=False).iloc[0])
pid, pname, ppos, pg = int(row.nfl_id), row.display_name, row.nfl_position, row.pgm
pick_txt = "undrafted" if row.pick >= 260 else f"pick {int(row.pick)}"
CACHE.mkdir(parents=True, exist_ok=True)
gif_path = CACHE / f"fig2_drill_vs_game_{pid}.gif"

if not gif_path.exists():
    # Combine: his position drill with the hardest braking and cutting (95th-percentile acceleration)
    cols = ["nfl_id", "entity_type", "event_id", "drill_name", "time", "x", "y", "s", "a"]
    ct = pd.concat(ch[ch.nfl_id == pid] for ch in pd.read_csv(get_file("combine_tracking.csv"), usecols=cols,
                                                              chunksize=1_000_000))
    ct = ct[ct.entity_type == "PLAYER"].copy()
    ct["time"] = pd.to_datetime(ct.time)
    ct = ct.sort_values(["event_id", "time"])
    ct["t"] = (ct.time - ct.groupby("event_id").time.transform("min")).dt.total_seconds()
    word = {"WR": "ROUTE", "DB": "BACK_PEDAL"}[pg]
    drills = ct[ct.drill_name.astype(str).str.contains(word)]
    if drills.empty:
        drills = ct[ct.drill_name != "FORTY_YARD_DASH"]

    def acc95(e):
        mv = e[e.s >= 1.0]
        return np.nanpercentile(mv.a, 95) if len(mv) >= 10 else np.nan

    ev = drills.groupby("event_id").apply(acc95).idxmax()
    cf = ct[ct.event_id == ev].sort_values("t")
    dname = str(cf.drill_name.iloc[0])
    on = np.where(cf.s.values >= 1.0)[0]
    cf = cf.iloc[max(on[0] - 3, 0):] if len(on) else cf
    cf = cf[cf.t - cf.t.iloc[0] <= 6.0]                  # the drill itself, not the jog out afterwards
    cx, cy = cf.x.values, cf.y.values
    cs, ctt = cf.s.values * YD_TO_MPH, cf.t.values - cf.t.values[0]

    # Game: his sharpest context-adjusted cuts in his role (the same route as the drill when possible)
    plm = T("play_level_movement.csv")
    mine = plm[(plm.nfl_id == pid) & plm.role.astype(bool) & plm.alat_c.notna()]
    if pg == "WR":
        same = mine[mine.route.astype(str).apply(lambda r: r not in ("NO_ROUTE", "nan") and r in dname.split("_"))]
        mine = same if len(same) >= 3 else mine[mine.route.astype(str) != "NO_ROUTE"]
    cands = mine.nlargest(5, "alat_c")[["game_id", "play_id", "season"]].astype(int)
    hits = []
    for seas_ in cands.season.unique():
        for chunk in pd.read_csv(get_file(f"game_tracking_{seas_}.csv"), chunksize=2_000_000,
                                 usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]):
            h = chunk[chunk.game_id.isin(cands.game_id)]
            if len(h):
                hits.append(h.merge(cands[["game_id", "play_id"]], on=["game_id", "play_id"]))
    gp_all = pd.concat(hits, ignore_index=True)
    gp_all["time"] = pd.to_datetime(gp_all.time)
    me = pd.DataFrame()
    for _, cand in cands.iterrows():
        gp = gp_all[(gp_all.game_id == cand.game_id) & (gp_all.play_id == cand.play_id)].copy()
        snap_t = gp.loc[gp.event == "ball_snap", "time"].min()
        if pd.isna(snap_t):
            continue
        gp["t"] = (gp.time - snap_t).dt.total_seconds().round(1)
        gp = gp[(gp.t >= 0) & (gp.t <= 6)].sort_values("t")
        me = gp[gp.nfl_id == pid].drop_duplicates("t")
        if len(me) >= 10:
            seas = int(cand.season)
            break
    others = {t: d for t, d in gp[gp.nfl_id != pid].groupby("t")}
    gx, gy, gs, gtt = me.x.values, me.y.values, me.s.values * YD_TO_MPH, me.t.values

    def field(ax, xs, ys, face, line):
        pad = 3
        xc, yc = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
        ax.set_xlim(xc - half, xc + half); ax.set_ylim(yc - half, yc + half); ax.set_aspect("equal")
        ax.set_facecolor(face); ax.set_xticks([]); ax.set_yticks([])
        for v in range(int(np.floor((xc - half) / 5) * 5), int(xc + half) + 1, 5):
            ax.axvline(v, color=line, lw=1.0, zorder=0)
        for v in range(int(np.floor((yc - half) / 5) * 5), int(yc + half) + 1, 5):
            ax.axhline(v, color=line, lw=1.0, zorder=0)

    fig = plt.figure(figsize=(10, 8))
    gsp = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.3], hspace=0.35, wspace=0.08)
    axc, axg, axs = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1]), fig.add_subplot(gsp[1, :])
    fig.suptitle(f"Figure 2. {pname} ({ppos}, {pick_txt}): 40 time {row.forty:.2f} s  |  "
                 f"Combine agility {row.A_trk_plus:.0f}  |  Game agility {row.g_agil_plus:.0f}",
                 x=0.02, ha="left", fontsize=13)
    field(axc, cx, cy, "#eeeeee", "#b5b5b5")
    axc.set_title("Combine drill:\n" + dname.replace("_", " ").title(), loc="left", fontsize=10)
    field(axg, gx, gy, "#dcefdc", "#a9c9a9")
    axg.set_title(f"NFL game ({seas}):\nsharpest cut on a {'route' if pg == 'WR' else 'coverage'} snap",
                  loc="left", fontsize=10)
    c_trail, = axc.plot([], [], color=OI[0], lw=2.2)
    c_dot, = axc.plot([], [], "o", color=OI[0], ms=11)
    o_dots = axg.scatter([], [], s=40, color="grey", alpha=0.6, label="other tracked players")
    g_trail, = axg.plot([], [], color=OI[1], lw=2.2)
    g_dot, = axg.plot([], [], "o", color=OI[1], ms=11, label=pname)
    axg.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    axc.set_xlabel("grid lines every 5 yards in both panels", fontsize=8, color="grey")
    tmax = max(ctt.max(), gtt.max())
    axs.set_xlim(0, tmax); axs.set_ylim(0, max(cs.max(), gs.max()) * 1.15)
    axs.set_xlabel("seconds since he starts moving (Combine) / since the snap (game)")
    axs.set_ylabel("speed (mph)")
    c_line, = axs.plot([], [], color=OI[0], lw=2.5, label="Combine drill")
    g_line, = axs.plot([], [], color=OI[1], lw=2.5, ls="--", label="NFL snap")
    axs.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)

    n_frames = int(round(tmax * 10)) + 15          # hold the last frame for 1.5 s

    def frame(i):
        ic, ig = min(i, len(cx) - 1), min(i, len(gx) - 1)
        c_trail.set_data(cx[:ic + 1], cy[:ic + 1]); c_dot.set_data([cx[ic]], [cy[ic]])
        g_trail.set_data(gx[:ig + 1], gy[:ig + 1]); g_dot.set_data([gx[ig]], [gy[ig]])
        od = others.get(gtt[ig])
        o_dots.set_offsets(od[["x", "y"]].values if od is not None else np.empty((0, 2)))
        c_line.set_data(ctt[:ic + 1], cs[:ic + 1]); g_line.set_data(gtt[:ig + 1], gs[:ig + 1])
        return c_trail, c_dot, g_trail, g_dot, o_dots, c_line, g_line

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=100, blit=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig)

display(Image(filename=str(gif_path)))
''')

md(r'''
## 4. Does Combine juice show up in games?

**Yes.** Across the 270 players with both scores, the rank correlation between Combine and Game Juice+ is **0.28** (95% interval 0.17 to 0.39). A formula built without each player's own draft class gives the same answer (0.27). It holds for part-timers (50 to 199 plays: 0.26) and starters (600+: 0.21). It is strongest for defensive backs (0.41), edge rushers (0.35), tight ends (0.53, only 20 players) and receivers (0.28), and weakest for the lines (OL 0.12, interior DL 0.10), where games rarely ask big men to run in space.
''')

code(r'''
# Figure 3: Combine Juice+ vs Game Juice+
J = T("juice_player_table.csv")
BP = T("by_position_group.csv").set_index("group")
fig, ax = plt.subplots(figsize=(7, 5.2))
for g, d in J.groupby("pgm"):
    ax.scatter(d.juice_final, d.game_juice, s=24, alpha=0.8, color=GROUP_COL[g], label=g)
ax.axhline(100, color="grey", lw=0.8); ax.axvline(100, color="grey", lw=0.8)
lo, hi = min(J.juice_final.min(), J.game_juice.min()) - 3, max(J.juice_final.max(), J.game_juice.max()) + 3
ax.plot([lo, hi], [lo, hi], color="grey", lw=0.7, ls=":")
a = BP.loc["ALL"]
ax.text(0.02, 0.97, f"rank correlation {a['corr']:.2f} (95% CI {a['ci_low']:.2f} to {a['ci_high']:.2f}), n = {int(a['n'])}",
        transform=ax.transAxes, va="top", fontsize=10)
for txt, (xx, yy) in {"juice translates": (0.97, 0.97), "hidden juice": (0.03, 0.88),
                      "didn't translate": (0.97, 0.03), "limitations persist": (0.03, 0.03)}.items():
    ax.text(xx, yy, txt, transform=ax.transAxes, ha="right" if xx > 0.5 else "left",
            va="top" if yy > 0.5 else "bottom", color="grey", fontsize=9, style="italic")
ax.set_xlabel("Combine Juice+ (tracking + stopwatch)"); ax.set_ylabel("Game Juice+ (NFL snaps)")
ax.set_title("Figure 3. Combine juice vs. game juice (100 = position average)", loc="left")
ax.legend(title="Group", fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.08))
plt.tight_layout(); plt.show()
''')

md(r'''
The more useful view for a draft room is by tier (Figure 4):
- **Red flag:** only 1 of 39 players under 90 became an elite mover in games (top 16% at his position).
- **Green flag:** 20 of 61 players at 110+ did (33%), more than double the 15% in the middle tier.
- **Reaching the field:** 88% of players at 110+ logged 50 usable NFL plays, against 57% of players under 90.

Inside the 90 to 110 tier, the score barely orders players (0.07). So read Juice+ as three tiers, not a ranking.
''')

code(r'''
# Figure 4: what happens in games to each Combine Juice+ tier
TI, RF = T("tiers.csv"), T("reached_field.csv")
LAB = ["Low\n(<90)", "Middle\n(90-110)", "High\n(110+)"]
cols = [OI[3], OI[5], OI[2]]
panels = [("Above average in games", TI.above_avg, TI.n),
          ("Elite in games (top 16%)", TI.elite_in_games, TI.n),
          ("Reached 50 usable NFL plays", RF.scored_50plus, RF.n)]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True)
for a_, (lab, k, n) in zip(ax, panels):
    k, n = k.to_numpy(), n.to_numpy()
    p_ = 100 * k / n
    ci = np.array([wilson(ki, ni) for ki, ni in zip(k, n)]) * 100
    a_.bar(range(3), p_, color=cols)
    a_.errorbar(range(3), p_, yerr=[p_ - ci[:, 0], ci[:, 1] - p_], fmt="none", ecolor="black", elinewidth=1, capsize=4)
    base = 100 * k.sum() / n.sum()
    a_.axhline(base, color="black", ls="--", lw=1)
    for i in range(3):
        a_.text(i, ci[i, 1] + 2, f"{k[i]} of {n[i]}", ha="center", fontsize=9.5)
    a_.set_xticks(range(3)); a_.set_xticklabels(LAB, fontsize=9)
    a_.set_title(f"{lab}\nall players: {base:.0f}% (dashed line)")
ax[0].set_ylabel("% of players in the Combine tier")
ax[0].set_ylim(0, 105)
fig.text(0.01, -0.03, "Whiskers: 95% intervals. First two panels: the 270 players with both scores. "
         "Last panel: all 414 players with a Combine score.", fontsize=9)
fig.suptitle("Figure 4. What happens in games to each Combine Juice+ tier", x=0.01, ha="left")
plt.tight_layout(); plt.show()
''')

md(r'''
**Which piece carries beyond the 40?** Figure 5 asks each Combine number to predict each game trait for receivers and defensive backs, with the 40 held fixed. A number that only repeats the stopwatch goes blank. Drill agility lights up its own trait: **+0.33** (0.15 to 0.49). Tracked speed adds a little to game top speed (+0.20). Burst adds nothing yet. The 40 itself predicts game top speed (+0.32) but not game agility. The stopwatch can't see change of direction.
''')

code(r'''
# Figure 5: Combine number vs game trait, WR + DB, holding the 40 fixed (the 40 row holds Combine agility instead)
SP = T("agility_specificity.csv")
SP = SP[SP.group == "WR + DB"]
rows_lab = ["Combine speed", "Combine burst", "Combine agility", "Official 40"]
cols_lab = ["game top speed", "game burst", "game agility"]
cm, ok = np.full((4, 3), np.nan), np.zeros((4, 3), bool)
for i, rl in enumerate(rows_lab):
    for j, cl in enumerate(cols_lab):
        r_ = SP[(SP["combine"] == rl) & (SP.game_trait == cl)].iloc[0]
        cm[i, j], ok[i, j] = r_.r, r_.lo > 0
fig, ax = plt.subplots(figsize=(6.6, 4.6))
im = ax.imshow(cm, vmin=-0.4, vmax=0.4, cmap="RdBu_r")
ax.set_xticks(range(3)); ax.set_xticklabels(["Game\ntop speed", "Game\nburst", "Game\nagility"])
ax.set_yticks(range(4)); ax.set_yticklabels(["Combine speed (T)", "Combine burst (E)", "Combine agility (A)",
                                             "Official 40\n(holding agility)"])
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]:+.2f}" + ("*" if ok[i, j] else ""), ha="center", va="center", fontsize=12,
                fontweight="bold" if ok[i, j] else "normal", color="white" if abs(cm[i, j]) > 0.25 else "black")
ax.axhline(2.5, color="white", lw=3)
plt.colorbar(im, ax=ax, fraction=0.046, label="correlation, 40 held fixed")
ax.set_title(f"Figure 5. What each Combine number adds beyond the 40\n(WR and DB, n = {int(SP.n.iloc[0])}; "
             "bold * = 95% interval above 0)", loc="left")
plt.tight_layout(); plt.show()
''')

md(r'''
## 5. Juice+ vs. the stopwatch

On the same 249 players, Combine Juice+ ranks game movement at **0.27**. The NGS Athleticism Score reaches 0.16 and a RAS-style score 0.14 (Figure 6). Both gaps are clear (+0.11 and +0.13, with intervals above 0).

The 40 is the real benchmark. The plain 40 reaches 0.24. Take body size out of it and it reaches 0.29, level with Juice+. For straight-line speed, nothing beats a stopwatch. Juice+'s value is the part a stopwatch can't time:

- **Agility beyond the 40.** For WR and DB, drill agility predicts game agility with the 40 held fixed at **+0.33** (n = 125). It stays at +0.33 to +0.35 when RAS-style, draft pick, play count or body size are also held fixed. Shuffling agility scores within each position never comes close (p < 0.001). Across all groups with agility, it adds +0.17 (0.05 to 0.30) to overall Game Juice+.
- **It isn't who got on the field.** WR and DB who never reached 50 plays had the same Combine agility as those who did (49th vs. 51st percentile).
- **It predicts growth.** Holding a player's rookie game agility fixed, his Combine agility still predicts his year-2 game agility: **+0.34** (0.12 to 0.54, n = 63). The 40 in the same model: **+0.01**. The 16 WR and DB whose drill agility ran a full standard deviation (SD) ahead of their rookie play gained 0.71 SD the next year. Everyone else slipped 0.48 SD.
''')

code(r'''
# Figure 6: how well each Combine score ranks game movement, same players
CO = T("carry_over.csv").set_index("score")
SCORES = [("Juice+", "juice_final", "Juice+", OI[0]),
          ("Official 40, size-adjusted", "forty_adj", "40, size-adjusted", OI[1]),
          ("RAS-style", "ras_style", "RAS-style", OI[4]),
          ("NGS athleticism", "ngs_athleticism_score", "NGS", "#999999")]
V = J.dropna(subset=["ngs_athleticism_score", "ras_style", "forty_adj", "forty_neg", "ngs_adj", "ras_adj"])
order = ["ALL", "DB", "EDGE", "TE", "WR", "DL", "OL"]
fig, ax = plt.subplots(figsize=(10, 4.1))
xs, wbar = np.arange(len(order)), 0.2
for k, (co_name, col, lab, colr) in enumerate(SCORES):
    vals = [CO.loc[co_name, "corr_with_game_juice"]] + [spearmanr(d[col], d.game_juice)[0]
                                                         for g in order[1:] for d in [V[V.pgm == g]]]
    off = (k - 1.5) * wbar
    ax.bar(xs + off, vals, wbar, color=colr, label=lab)
    ax.errorbar(off, CO.loc[co_name, "corr_with_game_juice"],
                yerr=[[CO.loc[co_name, "corr_with_game_juice"] - CO.loc[co_name, "ci_low"]],
                      [CO.loc[co_name, "ci_high"] - CO.loc[co_name, "corr_with_game_juice"]]],
                fmt="none", ecolor="black", elinewidth=1, capsize=3)
ax.axhline(0, color="black", lw=0.6)
n_g = V.pgm.value_counts()
ax.set_xticks(xs); ax.set_xticklabels([f"All players\n(n = {len(V)})"] + [f"{g}\n(n = {n_g[g]})" for g in order[1:]])
ax.set_ylabel("rank correlation with\nGame Juice+")
ax.set_ylim(-0.02, 0.7)
ax.legend(frameon=False, ncol=4, loc="upper right")
ax.set_title(f"Figure 6. How well each Combine score ranks game movement (same {len(V)} players; "
             "whiskers = 95% interval)", loc="left")
plt.tight_layout(); plt.show()
''')

md(r'''
## 6. Using Juice+ to add juice to a roster

Teams already pay for juice somewhat (first-round picks average 107, undrafted players 94), but the draft prices the whole player and leaves athletes on the board. A team can use Juice+ three ways:

1. **Draft-board filter.** Flag prospects at 110+ and red-flag those under 90. Juice+ says who has the gear. Scouts decide whether he can play. Late-round players at 110+ finished in the top half of game movement 4 of 7 times (other late picks: 37%), including Eric Watts (DE), Will Mallory (TE) and Marques Sigle (SS).
2. **The agility flag for WR and DB.** A receiver or corner whose drill agility is far ahead of his 40 is the prospect a stopwatch board misses, like Heath. Use it to sort a tier, not to call one player: in matched pairs with the same 40, the more agile player moved better in games only 53% of the time.
3. **Week to week, and year to year.** For top speed, a rookie's first 10 plays already beat his Combine, but the Combine keeps adding (+0.07 R² even after 100 plays). For agility, game film wins right away. So in season, trust the film for agility. Across seasons, trust the Combine: a rookie whose drill agility outran his game agility is a candidate to grow in year 2 (Table 1).
''')

code(r'''
# Table 1: WR and DB whose Combine agility was 1+ SD ahead of their rookie game agility, and what year 2 showed
cw = report("casework_report.txt")
sec = cw.split("C3. Unused agility")[1].split("C4.")[0]
pat = re.compile(r"^\s+(.+?)\s{2,}(\w+)\s+(\d{4}) pick\s+(\S+)\s+year-1 ([+-][\d.]+) -> year-2 ([+-][\d.]+) SD "
                 r"\(Combine ([+-][\d.]+)\)", re.M)
U = pd.DataFrame(pat.findall(sec), columns=["Player", "Pos", "Class", "Pick", "y1", "y2", "comb"])
for c in ["y1", "y2", "comb"]:
    U[c] = U[c].astype(float)
U["Pick"] = U.Pick.replace({"UDFA": "undrafted"})
tbl = pd.DataFrame({"Player": U.Player, "Pos": U.Pos, "Class": U.Class, "Pick": U.Pick,
                    "Combine agility": U.comb, "Rookie game agility": U.y1, "Year-2 game agility": U.y2,
                    "Change": U.y2 - U.y1})
print("Table 1. Unused agility: WR and DB whose Combine agility ran 1+ SD ahead of their rookie game agility "
      "(all values in SD within position)")
fmt = {c: "{:+.2f}" for c in ["Combine agility", "Rookie game agility", "Year-2 game agility", "Change"]}
try:
    display(tbl.style.hide(axis="index").format(fmt)
            .bar(subset=["Change"], align="zero", color=[OI[3], OI[0]], vmin=-2, vmax=2))
except Exception:
    display(tbl.round(2).reset_index(drop=True))
''')

md(r'''
**Does juice show up in results?** A little, and the stars had it. Of the 9 All-Pro or Pro Bowl players with a Combine score, 8 were above 100 (all scored players: 50%). Moving well also buys a receiver separation, and each yard of separation at the throw is worth +0.083 EPA per target. Figure 7 follows the chain for WRs: the Combine predicts game movement, and game movement predicts separation. End to end, one SD at the Combine is worth about 0.03 yards of separation, or 0.2 to 0.3 EPA per 100 targets. Real, but small.
''')

code(r'''
# Figure 7: the WR chain in yards and points (Combine -> game movement -> separation -> EPA)
RC = T("route_chain.csv")
RV = T("separation_value.csv").set_index("outcome")
rows_ = RC[RC.preregistered.astype(bool)].reset_index(drop=True)
epa_yd = RV.loc["EPA per target", "per_yard"]
NAMES = {"40-yard dash": "40-yard dash", "Combine drill cut speed": "cut speed in\nroute drills"}
fig, ax = plt.subplots(figsize=(11, 0.6 + 1.5 * len(rows_)))
ax.set_xlim(0, 10); ax.set_ylim(-0.7, len(rows_) - 0.4); ax.axis("off")
box = dict(boxstyle="round,pad=0.45", fc="white", ec="#bbbbbb", lw=1.2)
xs_ = [0.9, 3.6, 6.3, 8.9]
for i, r in rows_.iterrows():
    y = len(rows_) - 1 - i
    labels = [f"Combine\n{NAMES.get(r.combine_piece, r.combine_piece)}", f"Game\n{r.game_piece}",
              "Separation\nat the throw", "Points\n(EPA per target)"]
    for xx, lab, col in zip(xs_, labels, (OI[0], OI[1], "black", "black")):
        ax.text(xx, y, lab, ha="center", va="center", fontsize=9.5, color=col, bbox=box)
    arrows = [f"{r.a_sd_per_sd:+.2f} SD per SD", f"{r.b_yd_per_sd:+.3f} yd per SD", f"{epa_yd:+.3f} EPA per yd"]
    for k, lab in enumerate(arrows):
        ax.annotate("", xy=(xs_[k + 1] - 0.95, y), xytext=(xs_[k] + 0.95, y),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.4))
        ax.text((xs_[k] + xs_[k + 1]) / 2, y + 0.18, lab, ha="center", va="bottom", fontsize=9, color="#555555")
    ax.text(xs_[0] - 0.85, y - 0.45, f"End to end, per Combine SD: {r.ab_yd:+.3f} yd [{r.ab_lo:+.3f}, {r.ab_hi:+.3f}]"
            f" = {r.epa_per_100:+.2f} EPA per 100 targets  ·  n = {int(r.n)} WRs", ha="left", va="center",
            fontsize=9.5)
ax.set_title("Figure 7. From the Combine to separation to points (WR; movement and separation adjusted for route, "
             "coverage and time to throw)", loc="left", fontsize=11)
plt.tight_layout(); plt.show()
''')

md(r'''
## 7. Limits and next steps

- **Movement, not production.** Juice+ says how a player moves next to same-position, same-size peers. On results like separation, pressure and coverage EPA, no athleticism score (ours, NGS, RAS-style or the 40) predicts well once size is removed (all near +0.03 to +0.08).
- **Agility is strong but not settled.** For WR and DB it holds in the 2023 (+0.40) and 2024 (+0.35) classes, but the 2025 class alone is weaker (+0.15). Head to head with the 40 for game agility, agility leads, but not by a clear margin.
- **Small samples.** 20 to 64 players per position, so position-level intervals are wide.
- **Linemen.** OL and interior DL movement barely carries into games.
- **Scope.** Regular season only. Game tracking covers only this rookie group.

**Next:** score our 2026 receiver predictions, made before the season, once it ends.

*All numbers come from one run of the code on the competition data. Code cells are hidden. Expand any cell to see how a figure was made. The appendix recomputes every quoted number.*
''')

code(r'''
# Appendix A: every number quoted in the text, recomputed from this run's tables. A failed line means the text is stale.
cw, tr, ro = report("casework_report.txt"), report("transfer_report.txt"), report("routes_report.txt")
num = lambda pat, txt, g=1: float(re.search(pat, txt).group(g))
bv = T("burst_validity.csv").set_index("feature"); sd = T("size_dependence.csv").set_index("score")
CO = T("carry_over.csv").set_index("score"); ED = T("edges_vs_other_scores.csv").set_index("Juice+ minus")
ADD = T("tracking_adds.csv"); SPx = T("agility_specificity.csv")
spec = lambda comb, trait: SPx[(SPx.group == "WR + DB") & (SPx["combine"] == comb) & (SPx.game_trait == trait)].iloc[0]
agi = ADD[(ADD.piece.str.startswith("Agility")) & (ADD.group == "ALL") & (ADD.holding == "the 40")
          & (ADD.target == "Game Juice+")].iloc[0]
summ = json.loads((OUT / "transfer_summary.json").read_text())
TK = T("snap_takeover.csv"); TI_ = T("tiers.csv").set_index("tier"); RF_ = T("reached_field.csv").set_index("tier")
LR = T("late_round_agility.csv").set_index("display_name"); PR = T("agility_pairs.csv")
SFP = T("football_results_size_free_pooled.csv"); SFP = SFP[SFP["pool"] == "All blocks"]
heath = card.set_index("display_name").loc["Malik Heath"]
CJ["rd"] = CJ.draft_round.fillna(8)
cs_ = pd.read_csv(get_file("player_career_successes.csv"))
hon = cs_[(cs_.ap_all_pro_1st_team + cs_.ap_all_pro_2nd_team + cs_.pro_bowl_original_ballot) > 0]
hj = CJ[CJ.nfl_id.isin(hon.nfl_id)]
A2 = cw.split("A2.")[1].split("A3.")[0]
robust = [num(rf"{k}\s+([+-][\d.]+)", A2) for k in ["holding 40 \+ RAS-style ", "holding 40 \+ draft pick",
                                                       "holding 40 \+ log plays", "holding 40 \+ RAS \+ pick \+ plays",
                                                       "holding 40 \+ weight \+ height"]]
chain = T("route_chain.csv"); chain = chain[chain.preregistered.astype(bool)]
checks = {
    "tracked players 505": (C_["combine_tracked"], 505), "Combine scored 414": (C_["combine_scored"], 414),
    "game scored 320": (C_["game_scored"], 320), "both 270": (C_["both"], 270), "same players 249": (C_["head_to_head"], 249),
    "top speed retest 0.91": (bv.loc["fly10", "retest"], 0.91), "burst retest 0.67": (RUN["reliability"]["combine_retest"]["E_comp"], 0.67),
    "top speed vs 40 0.89": (bv.loc["fly10", "vs_forty"], 0.89),
    "DL weight vs raw speed -0.70": (r_raw, -0.70), "DL weight vs adjusted +0.15": (r_adj, 0.15),
    "Juice+ vs weight -0.03": (sd.loc["Juice+", "corr_with_weight"], -0.03),
    "40 vs weight -0.34": (sd.loc["Official 40", "corr_with_weight"], -0.34),
    "RAS vs weight -0.29": (sd.loc["RAS-style", "corr_with_weight"], -0.29),
    "game split-half top 0.94": (RUN["reliability"]["game_split_half"]["top"], 0.94),
    "game split-half agility 0.89": (RUN["reliability"]["game_split_half"]["agil"], 0.89),
    "game split-half burst 0.81": (RUN["reliability"]["game_split_half"]["expl"], 0.81),
    "Heath 40 4.64": (heath.forty, 4.64), "Heath 40 bottom 5%": (LR.loc["Malik Heath", "pct_neg_forty"] < 0.05, True),
    "Heath Juice+ 91": (heath.juice_final, 91), "Heath agility 119": (heath.A_trk_plus, 119),
    "Heath game agility 106": (heath.g_agil_plus, 106), "Heath plays 229": (heath.npl, 229),
    "carry 0.28": (BP.loc["ALL", "corr"], 0.28), "carry low 0.17": (BP.loc["ALL", "ci_low"], 0.17),
    "carry high 0.39": (BP.loc["ALL", "ci_high"], 0.39),
    "held-out formula 0.27": (CO.loc["Juice+ (held-out formula)", "corr_with_game_juice"], 0.27),
    "50-199 plays 0.26": (num(r"50-199\s+n=\s*\d+\s+([+-][\d.]+)", tr), 0.26),
    "600+ plays 0.21": (num(r"600\+\s+n=\s*\d+\s+([+-][\d.]+)", tr), 0.21),
    "DB 0.41": (BP.loc["DB", "corr"], 0.41), "EDGE 0.35": (BP.loc["EDGE", "corr"], 0.35),
    "TE 0.53": (BP.loc["TE", "corr"], 0.53), "TE n 20": (BP.loc["TE", "n"], 20), "WR 0.28": (BP.loc["WR", "corr"], 0.28),
    "OL 0.12": (BP.loc["OL", "corr"], 0.12), "DL 0.10": (BP.loc["DL", "corr"], 0.10),
    "under 90 elite 1 of 39": ((TI_.loc["<90", "elite_in_games"], TI_.loc["<90", "n"]) == (1, 39), True),
    "110+ elite 20 of 61": ((TI_.loc["110+", "elite_in_games"], TI_.loc["110+", "n"]) == (20, 61), True),
    "middle elite 15%": (TI_.loc["90-110", "elite_share"], 0.15),
    "reached 110+ 88%": (RF_.loc["110+", "share_scored"], 0.88), "reached <90 57%": (RF_.loc["<90", "share_scored"], 0.57),
    "inside 90-110 0.07": (summ["smaller_checks"]["tiers_inside_90_110"], 0.07),
    "agility beyond 40 +0.33": (spec("Combine agility", "game agility").r, 0.33),
    "agility low 0.15": (spec("Combine agility", "game agility").lo, 0.15),
    "agility high 0.49": (spec("Combine agility", "game agility").hi, 0.49),
    "agility n 125": (spec("Combine agility", "game agility").n, 125),
    "speed -> top +0.20": (spec("Combine speed", "game top speed").r, 0.20),
    "burst adds nothing (interval crosses 0)": (spec("Combine burst", "game burst").lo < 0, True),
    "40 -> top +0.32": (spec("Official 40", "game top speed").r, 0.32),
    "40 -> agility not clear": (spec("Official 40", "game agility").lo < 0, True),
    "Juice+ 0.27": (CO.loc["Juice+", "corr_with_game_juice"], 0.27),
    "NGS 0.16": (CO.loc["NGS athleticism", "corr_with_game_juice"], 0.16),
    "RAS 0.14": (CO.loc["RAS-style", "corr_with_game_juice"], 0.14),
    "plain 40 0.24": (CO.loc["Official 40", "corr_with_game_juice"], 0.24),
    "size-adjusted 40 0.29": (CO.loc["Official 40, size-adjusted", "corr_with_game_juice"], 0.29),
    "edge NGS +0.11": (ED.loc["NGS athleticism", "edge"], 0.11), "edge NGS low 0.02": (ED.loc["NGS athleticism", "ci_low"], 0.02),
    "edge NGS high 0.20": (ED.loc["NGS athleticism", "ci_high"], 0.20),
    "edge RAS +0.13": (ED.loc["RAS-style", "edge"], 0.13), "edge RAS low 0.05": (ED.loc["RAS-style", "ci_low"], 0.05),
    "edge RAS high 0.23": (ED.loc["RAS-style", "ci_high"], 0.23),
    "robust min 0.33": (min(robust), 0.33), "robust max 0.35": (max(robust), 0.35),
    "permutation p < 0.001": (num(r"permutation p .*?: ([\d.]+)", cw) < 0.001, True),
    "pooled agility +0.17": (agi.partial_r, 0.17), "pooled low 0.05": (agi.ci_low, 0.05), "pooled high 0.30": (agi.ci_high, 0.30),
    "reached field agility 51st pct": (num(r"reached 50\+ plays\s+n=\s*\d+\s+mean Combine agility pct ([\d.]+)", cw), 0.51),
    "did not reach 49th pct": (num(r"did not\s+n=\s*\d+\s+mean Combine agility pct ([\d.]+)", cw), 0.49),
    "year 2 agility +0.34": (num(r"Combine agility ([+-][\d.]+) \[", cw), 0.34),
    "year 2 low 0.12": (num(r"Combine agility [+-][\d.]+ \[([+-][\d.]+),", cw), 0.12),
    "year 2 high 0.54": (num(r"Combine agility [+-][\d.]+ \[[+-][\d.]+, ([+-][\d.]+)\]", cw), 0.54),
    "year 2 the 40 +0.01": (num(r"the 40 ([+-][\d.]+) \[", cw), 0.01),
    "year 2 n 63": (num(r"year-1 \+ Combine agility \+ 40, n=(\d+)", cw), 63),
    "flagged 16": (num(r"year-1 game agility\): (\d+);", cw), 16),
    "flagged gain +0.71": (num(r"mean change year 1 -> 2: ([+-][\d.]+) SD", cw), 0.71),
    "others -0.48": (num(r"SD vs ([+-][\d.]+) for the rest", cw), -0.48),
    "round 1 mean 107": (CJ[CJ.rd == 1].juice_final.mean(), 107), "no pick mean 94": (CJ[CJ.rd == 8].juice_final.mean(), 94),
    "late 110+ 4 of 7": ((summ["smaller_checks"]["late_round"]["hi_above"], summ["smaller_checks"]["late_round"]["hi_n"]) == (4, 7), True),
    "late others 37%": (summ["smaller_checks"]["late_round"]["lo_above"] / summ["smaller_checks"]["late_round"]["lo_n"], 0.37),
    "late names": (all(n_ in tr for n_ in ["Eric Watts", "Will Mallory", "Marques Sigle"]), True),
    "pairs 53%": (PR.agile_wins_game.mean(), 0.53),
    "top speed takeover 10 plays": (summ["takeover_k"]["game top speed"], 10),
    "Combine adds at 100 plays +0.07": (TK[(TK.metric == "game top speed") & (TK.k == 100)].combine_adds.iloc[0], 0.07),
    "agility film wins at once": (summ["takeover_k"]["game agility"], 1),
    "honors with a score 9": (len(hj), 9), "honors above 100: 8": ((hj.juice_final > 100).sum(), 8),
    "all scored above 100: 50%": ((CJ.juice_final > 100).mean(), 0.50),
    "EPA per yard 0.083": (RV.loc["EPA per target", "per_yard"], 0.083),
    "chain yards ~0.03": (chain.ab_yd.round(2).tolist(), [0.03, 0.03]),
    "chain EPA 0.2 to 0.3": (bool(chain.epa_per_100.between(0.2, 0.3).all()), True),
    "size-free results 0.03 to 0.08": (bool(SFP[SFP.score.isin(["Juice+", "NGS", "RAS-style", "40"])].estimate
                                            .between(0.025, 0.085).all()), True),
    "class 2023 +0.40": (num(r"draft class 2023 only\s+([+-][\d.]+)", cw), 0.40),
    "class 2024 +0.35": (num(r"draft class 2024 only\s+([+-][\d.]+)", cw), 0.35),
    "class 2025 +0.15": (num(r"draft class 2025 only\s+([+-][\d.]+)", cw), 0.15),
    "agility vs 40 head to head not clear": (num(r"Agility minus the 40 .*?\[([+-][\d.]+),", cw) < 0, True),
    "per-position n 20 to 64": ((BP.drop("ALL").n.min(), BP.drop("ALL").n.max()) == (20, 64), True),
}


def close(v, want):
    if isinstance(want, (bool, list, tuple)):
        return v == want
    return abs(float(v) - want) <= (0.5 if abs(want) >= 2 else 0.006)


bad = {k: v for k, (v, want) in checks.items() if not close(v, want)}
print(f"{len(checks) - len(bad)} of {len(checks)} quoted numbers match run {RUN['run_id']}.")
if bad:
    print("Update the text for:", bad)
''')

md(r"""
## Appendix B: Juice+ pipeline code

These are the scripts the notebook runs, in order. Running these cells writes the same files to `juice_code/`.
""")
for f in PIPE:
    c = nbf.v4.new_code_cell(f"%%writefile juice_code/{f}\n" + open(PIPE_DIR + f).read())
    c.metadata.update(HIDE_ALL)
    cells.append(c)

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, OUT_PATH)
print("wrote", OUT_PATH, len(cells), "cells")
