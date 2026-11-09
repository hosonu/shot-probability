import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shot_probability import columns
from shot_probability.geometry import distance_to_goal

SHOT_EVENT_TYPE = "Shot"
PASS_EVENT_TYPE = "Pass"
CARRY_EVENT_TYPE = "Carry"
_OWN_TEAM_EVENTS = "own_team_events"

FloatArray = NDArray[np.float64]


def build_pre_shot_context(
    events: pd.DataFrame, match_id: int, sequence_length: int
) -> pd.DataFrame:
    """Describe what happened earlier in the same possession for every shot of one match.

    Leakage safety is structural, not filtered after the fact: events are ordered by `index`
    inside each possession, context events are addressed as `shot_row - lag` with `lag >= 1`,
    and cumulative counts are exclusive. Nothing at or after the shot row is ever read, which
    `tests/test_pre_shot.py` verifies by corrupting and deleting every later event.

    "Attacking" always means the team that took the shot, since 1.2% of shots come from a team
    that is not StatsBomb's `possession_team`.

    Lag 1 is the event right before the shot. A shot with fewer than `sequence_length` earlier
    events has missing (None / NaN) values in the remaining lags.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    # StatsBomb returns rows ordered by timestamp, not by `index`, so ordering is explicit.
    ordered = events.sort_values(["possession", "index"], kind="stable").reset_index(drop=True)
    possession_key = ordered["possession"]
    shot_rows = np.flatnonzero((ordered["type"] == SHOT_EVENT_TYPE).to_numpy())

    locations = ordered["location"].astype(object)
    x_values = locations.str[0].to_numpy(dtype=np.float64)
    y_values = locations.str[1].to_numpy(dtype=np.float64)
    seconds = pd.to_timedelta(ordered["timestamp"]).dt.total_seconds()
    seconds_values = seconds.to_numpy()
    seconds_since_previous = (
        (seconds - seconds.groupby(possession_key).shift(1)).fillna(0.0).to_numpy()
    )
    possession_start_seconds = seconds.groupby(possession_key).transform("first").to_numpy()
    teams = ordered["team"].to_numpy(dtype=object)
    event_types = ordered["type"].to_numpy(dtype=object)

    prior_event_count = possession_key.groupby(possession_key).cumcount().to_numpy()
    team_events = pd.DataFrame(
        {
            columns.N_PRIOR_PASSES: ordered["type"] == PASS_EVENT_TYPE,
            columns.N_PRIOR_CARRIES: ordered["type"] == CARRY_EVENT_TYPE,
            _OWN_TEAM_EVENTS: True,
        }
    ).astype(int)
    prior_team_counts = (
        team_events.groupby([possession_key, ordered["team"]]).cumsum() - team_events
    )

    lags = np.arange(1, sequence_length + 1)
    context_rows = shot_rows[:, None] - lags[None, :]
    is_available = lags[None, :] <= prior_event_count[shot_rows][:, None]
    safe_rows = np.where(is_available, context_rows, 0)

    no_event_types = np.full(is_available.shape, None, dtype=object)
    context_types = np.where(is_available, event_types[safe_rows], no_event_types)
    context_x = np.where(is_available, x_values[safe_rows], np.nan)
    context_y = np.where(is_available, y_values[safe_rows], np.nan)
    context_seconds = np.where(is_available, seconds_since_previous[safe_rows], np.nan)
    context_is_attacking = is_available & (teams[safe_rows] == teams[shot_rows][:, None])

    has_previous = is_available[:, 0]
    shot_seconds = seconds_values[shot_rows]
    seconds_since_previous_event = np.where(
        has_previous, shot_seconds - seconds_values[safe_rows[:, 0]], 0.0
    )

    scalars = pd.DataFrame(
        {
            columns.MATCH_ID: match_id,
            columns.EVENT_ID: ordered["id"].to_numpy()[shot_rows],
            columns.N_PRIOR_EVENTS: prior_event_count[shot_rows],
            columns.N_PRIOR_PASSES: prior_team_counts[columns.N_PRIOR_PASSES].to_numpy()[shot_rows],
            columns.N_PRIOR_CARRIES: prior_team_counts[columns.N_PRIOR_CARRIES].to_numpy()[
                shot_rows
            ],
            columns.N_PRIOR_DEFENDING_EVENTS: prior_event_count[shot_rows]
            - prior_team_counts[_OWN_TEAM_EVENTS].to_numpy()[shot_rows],
            columns.POSSESSION_SECONDS_BEFORE_SHOT: shot_seconds
            - possession_start_seconds[shot_rows],
            columns.SECONDS_SINCE_PREVIOUS_EVENT: seconds_since_previous_event,
            columns.PROGRESSION_IN_WINDOW: _progression_in_window(
                context_x, context_y, x_values[shot_rows], y_values[shot_rows]
            ),
            columns.LAST_EVENT_TYPE: np.where(
                has_previous, context_types[:, 0], columns.NO_PRIOR_EVENT
            ),
        }
    )
    sequences = {
        columns.SEQUENCE_EVENT_TYPE: context_types,
        columns.SEQUENCE_X: context_x,
        columns.SEQUENCE_Y: context_y,
        columns.SEQUENCE_SECONDS: context_seconds,
        columns.SEQUENCE_IS_ATTACKING: context_is_attacking,
    }
    sequence_frames = [
        pd.DataFrame(values, columns=columns.sequence_columns(field, sequence_length))
        for field, values in sequences.items()
    ]
    return pd.concat([scalars, *sequence_frames], axis=1)


def _progression_in_window(
    context_x: FloatArray, context_y: FloatArray, shot_x: FloatArray, shot_y: FloatArray
) -> FloatArray:
    """Reduction in distance to goal between the oldest located context event and the shot."""
    is_located = ~np.isnan(context_x) & ~np.isnan(context_y)
    oldest_lag_index = is_located.shape[1] - 1 - np.argmax(is_located[:, ::-1], axis=1)
    rows = np.arange(len(context_x))
    window_start_distance = distance_to_goal(
        context_x[rows, oldest_lag_index], context_y[rows, oldest_lag_index]
    )
    shot_distance = distance_to_goal(shot_x, shot_y)
    can_measure = is_located.any(axis=1) & ~np.isnan(shot_distance)
    return np.where(can_measure, window_start_distance - shot_distance, 0.0)
