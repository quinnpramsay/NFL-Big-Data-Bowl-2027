# Juice+: Finding the Extra Gear in the "Underwear Olympics"

NFL Big Data Bowl 2027 entry. Juice+ scores how a prospect moves in Combine drills on a scale where 100 is average for his position, then rebuilds the same score from NFL game tracking to see if the juice carries over.

## Start here

- **Write-up:** [submission/README.md](submission/README.md)
- **All the work:** [submission/Juice+_work.ipynb](submission/Juice+_work.ipynb). It runs the full pipeline, draws every figure and table in the write-up, and rechecks every number it quotes.

## Run it

1. Download the competition data from Kaggle.
2. Open `submission/Juice+_work.ipynb`. If your data is not found on its own, set `DATA` in the first cell to the folder that holds `players.csv`.
3. Run all cells. The first run takes about 20 minutes. Later runs reuse the saved outputs.

Needs Python 3 with numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib and Pillow.

## What is in this repo

| Folder | What it holds |
|---|---|
| `submission/` | The write-up, the work notebook and the figures |
| `programs/` | The Juice+ pipeline as plain scripts (the same code the work notebook carries in Appendix B) |
| `forward_test/` | Our 2026 receiver predictions, saved before the 2026 season with a checksum. Score them after the season with `python3 programs/juice_forward.py --score` |

Earlier versions, test runs and research notes are on the `Testing` branch.
