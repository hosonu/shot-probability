# Will this shot be a goal?

A logistic-regression baseline that predicts `P(goal | shot)` from StatsBomb open data (La Liga 2015/16, `competition_id=11`, `season_id=27`). Models are compared on probability quality, not accuracy.

## What is predicted

For every shot, the probability that its outcome is **Goal**. Saved, blocked, off-target, and woodwork shots are labelled 0.

Features (phase 1 only):

| feature | meaning |
|---|---|
| `distance_to_goal` | Euclidean distance from the shot to the goal centre `(120, 40)` |
| `angle_to_goal` | angle in radians subtended by the two posts `(120, 36)` and `(120, 44)` |
| `body_part` | `foot`, `head`, or `other` |
| `is_set_piece` | corner, free kick, penalty, or kick-off shot |

StatsBomb's own xG is never used as an input.

## Data

`make data` fetches the events of all 380 matches, keeps only shots, and writes `data/shots.parquet`: **9,168 shots, 1,014 goals (goal rate 11.06%), 0 dropped for missing location**. The table holds `match_id` (the split key), the four features, and the `goal` label. No raw JSON is stored. Training reads only this parquet file and never calls StatsBomb.

## Loss, baseline, and measurement

- **Loss**: log loss (primary). Brier score and a calibration plot are secondary. Accuracy is descriptive only, because goals are rare: always predicting "no goal" already scores ~89%.
- **Split**: by `match_id`, 70 / 15 / 15 (seed 42), so no match appears in more than one partition. Train 6,359 shots, validation 1,388, test 1,421.
- **Baseline**: logistic regression (standardised numeric features, one-hot body part, set-piece flag). The `prior` row predicts the training goal rate for every shot and shows the floor to beat.

| model | split | log loss | Brier | accuracy |
|---|---|---|---|---|
| prior | validation | 0.3276 | 0.0908 | 0.8991 |
| prior | test | 0.3578 | 0.1021 | 0.8846 |
| logistic | validation | 0.2603 | 0.0742 | 0.9085 |
| logistic | test | 0.3023 | 0.0859 | 0.8923 |

![Calibration of the logistic baseline on validation and test](docs/calibration.png)

The plot is a snapshot of `outputs/calibration.png` from `make train`.

## Where it fails

- Only four features. It cannot see defenders, goalkeeper position, or assist type, so it cannot separate a clear one-on-one from a crowded box.
- Penalties share the `is_set_piece` flag with free kicks. In a sample of 80 matches, penalties scored ~74% and free kicks ~10%, so one flag cannot represent both.
- Each partition holds only ~160 goals, so the validation to test gap in log loss (0.260 vs 0.302) is partly sampling noise. The highest-probability calibration bin holds few shots, so it is noisy as well.
- One competition and one season. Nothing here says how it generalises to other leagues.

## Run it

```bash
make install   # local uv, Python 3.12, and .venv
make data      # build data/shots.parquet
make train     # write outputs/metrics.csv and outputs/calibration.png
make lint
make check
```

Dependencies are stable releases pinned in `uv.lock`. See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and style, and `shot-probability.spec.md` for the project invariants.

## Data attribution

Data from [StatsBomb Open Data](https://github.com/statsbomb/open-data). Review and follow StatsBomb's licence and attribution terms before publishing.
