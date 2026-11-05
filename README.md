# Previous Winners/Qualifiers

Looking back on previous Big Data bowl completitions the one large trend that stands out is the people create their own new metrics within the sport instead of just building a machine learning pipeline. They also used context specific measurements, ex. 100 out of 200 tackles versus 40 out of 50 tackles. They were also incredibly digestible but also differentiated from the crowd. 

# Initial Theories

- Can we identify which Combine movement characteristics translate into effective movement in the trenches, where athletic ability must be applied under physical constraints? - how do you define effective movement within blocking. 
- Does Combine-measured lateral agility and acceleration predict a defensive lineman's ability to recover and generate pressure after his initial pass-rush movement is disrupted? - identifying player success versus opponent and scheme.
- Coverage Recovery Index: How well a DB recovers from a break in a wide receivers route and how well that can be derived from combine data in reaction time - the biggest hurdle will be identifying route breaks in NFL games. 
- Can Combine movement data identify tight ends who are effective in both receiving and blocking situations? - smaller sample size and reliably measuring blocking effectiveness may be difficult.

# Data
Uses kaggle api call to import all the data from kaggle. must pip install --upgrade kagglehub in terminal before running data.py script. 

# Notebooks
juice+ = thomas current version
juice_plus = quinn current version

# Pipeline (current)

The one pipeline is in `programs/`. Run from the repo root, in this order:

```
python3 programs/juice.py            # Combine Juice+ and Game Juice+ (about 1 min with the cached play table)
python3 programs/juice_physics.py    # acceleration-speed profiles, Combine and games; preseason counts (about 15 min first time, cached after)
python3 programs/juice_transfer.py   # the chain, what tracking adds, mixed models, snap takeover, tiers, primary tests
python3 programs/juice_figures.py    # six figures for the writeup
```

Every table in `juice_nb_output/` carries the run ID that `juice.py` wrote to `run_manifest.json`. The later scripts
refuse to mix runs, so every number in the writeup traces to one run. Main read: `juice_nb_output/transfer_report.txt`.
Primary tests are listed in `programs/PRIMARY_TESTS.md`; everything else is exploratory.

Superseded (kept for history, do not quote numbers from them): `juice.py` and `outcome.py` at the repo root,
`programs/outcome.py` (replaced by `juice_transfer.py`), `Juice+.ipynb` (older logic, see audit item 7), and the
`juice_full_output/` and `juice_experiment_output/` folders.
