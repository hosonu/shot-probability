# Will this shot be a goal?

Predicts `P(goal | shot)` from StatsBomb open data (La Liga 2015/16, `competition_id=11`, `season_id=27`). A logistic regression is the baseline; gradient boosting and a from-scratch Transformer are compared against it on probability quality, not accuracy.

## What is predicted

For every shot, the probability that its outcome is **Goal**. Saved, blocked, off-target, and woodwork shots are labelled 0.

Shot features (phase 1, used by every model):

| feature | meaning |
|---|---|
| `distance_to_goal` | Euclidean distance from the shot to the goal centre `(120, 40)` |
| `angle_to_goal` | angle in radians subtended by the two posts `(120, 36)` and `(120, 44)` |
| `body_part` | `foot`, `head`, or `other` |
| `is_set_piece` | corner, free kick, penalty, or kick-off shot |

Pre-shot features (phase 2, used only by the `*_context` models and, as a sequence, by the Transformer) describe earlier events **of the same possession**:

| feature | meaning |
|---|---|
| `n_prior_events`, `possession_seconds_before_shot` | how long the possession had run before the shot (uncapped) |
| `n_prior_passes`, `n_prior_carries` | passes and carries by the shooting team earlier in the possession |
| `n_prior_defending_events` | earlier events by the other team (pressures, duels, recoveries) |
| `seconds_since_previous_event`, `last_event_type` | the event right before the shot |
| `progression_in_window` | distance to goal gained between the oldest of the last 16 located events and the shot |

The Transformer instead reads the last K = 16 earlier events, each with type, location, seconds since its own predecessor, and whether the shooting team made it. K = 16 was chosen once: the median shot follows 13 events in its possession, and 16 covers 58% of shots completely.

Freeze-frame features (phase 2, used only by the `*_freeze_frame` models) describe the defensive state at shot time, read from StatsBomb's `shot.freeze_frame`:

| feature | meaning |
|---|---|
| `goalkeeper_distance_to_goal`, `goalkeeper_y_offset`, `shot_to_goalkeeper_distance` | opponent goalkeeper position relative to goal and shot |
| `nearest_opponent_distance` | distance from the shot to the closest outfield opponent |
| `opponents_in_shot_triangle`, `teammates_in_shot_triangle` | players inside the triangle (shot, low post, high post) |
| `opponents_between_shot_and_goal` | outfield opponents whose x is beyond the shot x |
| `has_freeze_frame`, `has_goalkeeper` | flags; missing-frame shots carry sentinel zeros |

StatsBomb's own xG is never used as an input.

### Leakage guard

Only events with a lower `index` than the shot, inside the same match and possession, can reach a feature. This is structural in `pre_shot.py` (events are sorted by `index`, context rows are addressed as `shot_row - lag` with `lag >= 1`, counts are exclusive), not a filter applied afterwards. `tests/test_pre_shot.py` proves it by deleting or corrupting every event after the shot and requiring identical output. StatsBomb returns rows ordered by timestamp rather than by `index`, so the tests also shuffle the input rows.

## Data

`make data` fetches the events of all 380 matches, keeps only shots, and writes `data/shots.parquet`: **9,168 shots, 1,014 goals (goal rate 11.06%), 0 dropped for missing location**. The table holds `match_id` (the split key), `event_id` (join key), the four shot features, and the `goal` label. `data/shot_context.parquet` holds the pre-shot features and the K = 16 event sequence for the same shots. No raw JSON is stored. Benchmarking reads only these parquet files and never calls StatsBomb.

## Loss, baseline, and measurement

- **Loss**: log loss (primary). Brier score and a calibration plot are secondary. Accuracy is descriptive only, because goals are rare: always predicting "no goal" already scores ~89%.
- **Split**: by `match_id`, 70 / 15 / 15 (seed 42), so no match appears in more than one partition. Train 6,359 shots, validation 1,388, test 1,421. Every model sees exactly the same rows.
- **Baseline**: logistic regression on the shot features. The `prior` row predicts the training goal rate for every shot and shows the floor to beat.
- **Fair comparison**: all models implement one interface (`fit(train, validation)`, `predict_goal_probability`). The validation split may be used for early stopping or choosing boosting rounds; the test split is never seen during fitting. Gradient boosting and the Transformer therefore have *optimistic validation scores*; the test column is the honest one.
- **Uncertainty**: shots from one match are related, so the last column resamples whole matches (2,000 paired bootstrap draws) and reports the difference in test log loss versus the logistic baseline with a 95% interval. Negative is better.

## Results

| model | features | val log loss | test log loss | test Brier | test log loss vs logistic (95% CI) |
|---|---|---|---|---|---|
| prior | none | 0.3276 | 0.3578 | 0.1021 | |
| logistic | shot | 0.2603 | 0.3023 | 0.0859 | baseline |
| logistic_context | shot + pre-shot | 0.2596 | 0.3012 | 0.0855 | -0.0011 (-0.0078, +0.0053) |
| gradient_boosting | shot | 0.2612 | 0.3039 | 0.0876 | +0.0017 (-0.0052, +0.0083) |
| gradient_boosting_context | shot + pre-shot | 0.2593 | 0.3024 | 0.0870 | +0.0001 (-0.0069, +0.0072) |
| logistic_freeze_frame | shot + freeze-frame | 0.2457 | 0.2891 | 0.0831 | **-0.0131 (-0.0242, -0.0018)** |
| gradient_boosting_freeze_frame | shot + freeze-frame | 0.2447 | 0.2894 | 0.0836 | **-0.0129 (-0.0239, -0.0021)** |
| transformer | shot + 16-event sequence | 0.2598 | 0.3048 | 0.0869 | +0.0025 (-0.0055, +0.0107) |

The first two rows are identical to phase 1.

**Freeze-frame features beat the logistic baseline; pre-shot context and the Transformer do not.** The two `*_freeze_frame` rows improve test log loss by ~0.013 nats and their 95% intervals exclude zero, so the gain is distinguishable from match-level noise. Pre-shot context improves log loss in-sample (logistic, train: 0.2819 to 0.2757) but that gain does not carry over to unseen matches; the Transformer, which reads a sequence of pre-shot events, lands in the same place. Where the shot was taken and what the defence looked like at shot time are the inputs that moved the needle; what happened earlier in the possession did not.

![Calibration of every model on validation and test](docs/calibration.png)

The plot is a snapshot of `outputs/calibration.png` from `make benchmark`. All models are calibrated about equally well; none can be separated visually.

## Where it fails

- Freeze-frame features are static snapshots: a defender one metre closer at the moment of the shot moves the probability, but player velocity, body orientation, and the goalkeeper's set position are not represented.
- About 4% of shots have no `shot.freeze_frame` (mostly penalties and some set pieces). They get sentinel zeros and `has_freeze_frame=False`, so the freeze-frame models fall back to a flagged baseline on those rows rather than modelling them well.
- Pre-shot events carry only type, location, time, and side. Pass height, cross, through ball, and cut-back flags are not used.
- Penalties share the `is_set_piece` flag with free kicks. In a sample of 80 matches, penalties scored ~74% and free kicks ~10%. The highest-probability calibration bin averages only ~0.43, so penalties are likely still underestimated.
- Each partition holds only ~160 goals. Five Transformer seeds (ad-hoc check, not part of the pipeline) gave test log loss from 0.2989 to 0.3059, a range that contains the logistic baseline (0.3023). Seed noise is as large as every difference in the table.
- Hyperparameters of gradient boosting and the Transformer are small fixed defaults; only the number of boosting rounds and the Transformer's stopping epoch are chosen on validation. No search was run, so these results do not rule out that a tuned model does better.
- One competition and one season. Nothing here says how it generalises to other leagues.

## Run it

```bash
make install   # local uv, Python 3.12, and .venv
make data      # build data/shots.parquet and data/shot_context.parquet
make benchmark # write outputs/metrics.csv, outputs/log_loss_vs_logistic.csv, outputs/calibration.png
make lint
make check
```

Dependencies are stable releases pinned in `uv.lock`. See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and style, and `shot-probability.spec.md` for the project invariants.

## Data attribution

Data from [StatsBomb Open Data](https://github.com/statsbomb/open-data). Review and follow StatsBomb's licence and attribution terms before publishing.
