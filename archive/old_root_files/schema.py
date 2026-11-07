import os
from pathlib import Path

import pandas as pd

# Find the competition folder: JUICE_DATA_DIR if set, else the kagglehub cache.
COMP = "nfl-big-data-bowl-2027"
root = os.getenv("JUICE_DATA_DIR")
if not root:
    import kagglehub
    root = kagglehub.competition_download(COMP)
root = Path(root)
if (root / COMP).is_dir():
    root = root / COMP

files = sorted(root.glob("*.csv"))
lines = [f"Folder: {root}", f"Files: {len(files)}", ""]

# For every file: size, column names with types, and the first 3 rows (headers only are read).
for f in files:
    head = pd.read_csv(f, nrows=3)
    lines.append(f"=== {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    lines.append(", ".join(f"{c} [{t}]" for c, t in head.dtypes.astype(str).items()))
    lines.append(head.to_string(index=False))
    lines.append("")

# Small files: full row counts. Big files: skip (they are slow to count).
for f in files:
    if f.stat().st_size < 20e6:
        lines.append(f"Rows in {f.name}: {len(pd.read_csv(f))}")
lines.append("")

# Role and scheme fields in player_play: what values exist (these decide what can be matched).
pp = root / "player_play.csv"
if pp.exists():
    df = pd.read_csv(pp)
    lines.append(f"Rows in player_play.csv: {len(df)}")
    for col in df.columns:
        n = df[col].nunique(dropna=True)
        if 1 < n <= 40 and df[col].dtype == object:
            lines.append(f"\n{col} ({n} values, {df[col].notna().mean():.0%} filled)")
            lines.append(df[col].value_counts().head(40).to_string())

# Career outcomes file: this is the only production-style outcome in the data.
cs = root / "player_career_successes.csv"
if cs.exists():
    d = pd.read_csv(cs)
    lines.append("\n=== player_career_successes.csv")
    lines.append(d.describe(include="all").T.to_string())

Path("schema_report.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("\nSaved: schema_report.txt")