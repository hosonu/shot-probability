import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shot_probability import columns
from shot_probability.geometry import (
    GOAL_CENTER_Y,
    GOAL_X,
    POST_HIGH_Y,
    POST_LOW_Y,
    distance_to_goal,
)

logger = logging.getLogger(__name__)

SHOT_EVENT_TYPE = "Shot"
GOALKEEPER_POSITION_NAME = "Goalkeeper"
RAW_FREEZE_FRAME_COLUMN = "shot_freeze_frame"

FloatArray = NDArray[np.float64]


def build_freeze_frame_features(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Per-shot features derived from StatsBomb's ``shot.freeze_frame``.

    The frozen player locations describe the defensive state the shot has to beat,
    the complement of the pre-shot sequence which captures build-up. Shots whose
    frame is missing (penalties and some set pieces) get sentinel zeros and
    ``has_freeze_frame=False`` so a linear or tree model can still consume the row.

    "Opponents" means the team that did not take the shot (``teammate=False``);
    the goalkeeper is the opponent whose ``position.name`` is ``Goalkeeper``.
    """
    shot_mask = events["type"] == SHOT_EVENT_TYPE
    shots = events.loc[shot_mask, ["id", "location", RAW_FREEZE_FRAME_COLUMN]].reset_index(
        drop=True
    )
    shot_count = len(shots)
    shot_locations = _coordinates(shots["location"])
    long_players = _explode_freeze_frames(shots[RAW_FREEZE_FRAME_COLUMN])

    scalars = _aggregate_player_features(long_players, shot_locations, shot_count)
    return pd.DataFrame(
        {
            columns.MATCH_ID: match_id,
            columns.EVENT_ID: shots["id"].to_numpy(),
            **scalars,
        }
    )


def _coordinates(locations: pd.Series) -> FloatArray:
    """Stack a Series of ``[x, y]`` lists into an ``(n, 2)`` float array (NaN for missing)."""
    pairs = locations.apply(lambda value: value if isinstance(value, list) else [np.nan, np.nan])
    return np.asarray(pairs.tolist(), dtype=np.float64).reshape(-1, 2)


def _explode_freeze_frames(frames: pd.Series) -> pd.DataFrame:
    """One row per frozen player with the shot index that owns them.

    A missing frame contributes zero rows; its shot still appears in the aggregates
    through ``reindex`` later, with its ``has_freeze_frame`` flag off.
    """
    frame_lists = frames.apply(lambda frame: frame if isinstance(frame, list) else [])
    shot_index = np.repeat(np.arange(len(frames)), frame_lists.str.len().to_numpy())
    exploded_frames = [player for frame in frame_lists for player in frame]
    if not exploded_frames:
        return pd.DataFrame(
            {
                "shot_index": np.array([], dtype=np.int64),
                "player_x": np.array([], dtype=np.float64),
                "player_y": np.array([], dtype=np.float64),
                "is_opponent": np.array([], dtype=bool),
                "is_goalkeeper": np.array([], dtype=bool),
            }
        )
    locations = np.asarray(
        [player.get("location", [np.nan, np.nan]) for player in exploded_frames], dtype=np.float64
    )
    teammate = np.asarray(
        [bool(player.get("teammate", True)) for player in exploded_frames], dtype=bool
    )
    position_names = np.asarray(
        [(player.get("position") or {}).get("name", "") for player in exploded_frames], dtype=object
    )
    return pd.DataFrame(
        {
            "shot_index": shot_index,
            "player_x": locations[:, 0],
            "player_y": locations[:, 1],
            "is_opponent": ~teammate,
            "is_goalkeeper": position_names == GOALKEEPER_POSITION_NAME,
        }
    )


def _aggregate_player_features(
    long_players: pd.DataFrame, shot_locations: FloatArray, shot_count: int
) -> dict[str, NDArray[np.generic]]:
    """Fold the exploded player rows into one scalar per shot per feature."""
    shot_index = long_players["shot_index"].to_numpy()
    player_x = long_players["player_x"].to_numpy()
    player_y = long_players["player_y"].to_numpy()
    is_opponent = long_players["is_opponent"].to_numpy()
    is_goalkeeper = long_players["is_goalkeeper"].to_numpy()

    shot_x = shot_locations[shot_index, 0]
    shot_y = shot_locations[shot_index, 1]
    distance_to_shot = np.hypot(player_x - shot_x, player_y - shot_y)
    in_triangle = _in_shot_triangle(shot_x, shot_y, player_x, player_y)
    between_shot_and_goal = (GOAL_X - player_x) < (GOAL_X - shot_x)

    opp_mask = is_opponent & ~is_goalkeeper
    nearest_opponent = _per_shot_reduce(
        shot_index[opp_mask], distance_to_shot[opp_mask], shot_count, np.minimum, np.inf
    )
    nearest_opponent = np.where(np.isfinite(nearest_opponent), nearest_opponent, 0.0)

    opponents_in_triangle = _per_shot_count(
        shot_index[is_opponent & in_triangle & ~is_goalkeeper], shot_count
    )
    teammates_in_triangle = _per_shot_count(shot_index[(~is_opponent) & in_triangle], shot_count)
    opponents_between = _per_shot_count(
        shot_index[is_opponent & between_shot_and_goal & ~is_goalkeeper], shot_count
    )

    goalkeeper_rows = long_players.loc[is_opponent & is_goalkeeper].drop_duplicates(
        subset="shot_index", keep="first"
    )
    has_goalkeeper = np.zeros(shot_count, dtype=bool)
    goalkeeper_x = np.zeros(shot_count, dtype=np.float64)
    goalkeeper_y = np.full(shot_count, GOAL_CENTER_Y, dtype=np.float64)
    if len(goalkeeper_rows):
        indices = goalkeeper_rows["shot_index"].to_numpy()
        has_goalkeeper[indices] = True
        goalkeeper_x[indices] = goalkeeper_rows["player_x"].to_numpy()
        goalkeeper_y[indices] = goalkeeper_rows["player_y"].to_numpy()

    has_freeze_frame = np.zeros(shot_count, dtype=bool)
    if shot_count:
        has_freeze_frame[np.unique(shot_index)] = True

    return {
        columns.HAS_FREEZE_FRAME: has_freeze_frame,
        columns.HAS_GOALKEEPER: has_goalkeeper & has_freeze_frame,
        columns.GOALKEEPER_DISTANCE_TO_GOAL: np.where(
            has_goalkeeper, distance_to_goal(goalkeeper_x, goalkeeper_y), 0.0
        ),
        columns.GOALKEEPER_Y_OFFSET: np.where(
            has_goalkeeper, np.abs(goalkeeper_y - GOAL_CENTER_Y), 0.0
        ),
        columns.SHOT_TO_GOALKEEPER_DISTANCE: np.where(
            has_goalkeeper,
            np.hypot(goalkeeper_x - shot_locations[:, 0], goalkeeper_y - shot_locations[:, 1]),
            0.0,
        ),
        columns.NEAREST_OPPONENT_DISTANCE: nearest_opponent,
        columns.OPPONENTS_IN_SHOT_TRIANGLE: opponents_in_triangle,
        columns.TEAMMATES_IN_SHOT_TRIANGLE: teammates_in_triangle,
        columns.OPPONENTS_BETWEEN_SHOT_AND_GOAL: opponents_between,
    }


def _in_shot_triangle(
    shot_x: FloatArray, shot_y: FloatArray, point_x: FloatArray, point_y: FloatArray
) -> NDArray[np.bool_]:
    """Whether each point lies inside the triangle (shot, low post, high post).

    Checked by same-sign of the three edge cross products; degenerate (point on
    an edge) counts as inside, which does not matter for integer counts.
    """
    post_low_x = np.full_like(shot_x, GOAL_X)
    post_low_y = np.full_like(shot_x, POST_LOW_Y)
    post_high_x = np.full_like(shot_x, GOAL_X)
    post_high_y = np.full_like(shot_x, POST_HIGH_Y)
    cross_a = _edge_cross(shot_x, shot_y, post_low_x, post_low_y, point_x, point_y)
    cross_b = _edge_cross(post_low_x, post_low_y, post_high_x, post_high_y, point_x, point_y)
    cross_c = _edge_cross(post_high_x, post_high_y, shot_x, shot_y, point_x, point_y)
    all_non_negative = (cross_a >= 0) & (cross_b >= 0) & (cross_c >= 0)
    all_non_positive = (cross_a <= 0) & (cross_b <= 0) & (cross_c <= 0)
    return all_non_negative | all_non_positive


def _edge_cross(
    ax: FloatArray,
    ay: FloatArray,
    bx: FloatArray,
    by: FloatArray,
    px: FloatArray,
    py: FloatArray,
) -> FloatArray:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _per_shot_reduce(
    shot_index: NDArray[np.int64],
    values: FloatArray,
    shot_count: int,
    reducer: np.ufunc,
    identity: float,
) -> FloatArray:
    output = np.full(shot_count, identity, dtype=np.float64)
    if len(shot_index):
        reducer.at(output, shot_index, values)
    return output


def _per_shot_count(shot_index: NDArray[np.int64], shot_count: int) -> NDArray[np.int64]:
    return np.bincount(shot_index, minlength=shot_count).astype(np.int64)
