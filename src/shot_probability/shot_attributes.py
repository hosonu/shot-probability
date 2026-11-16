import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shot_probability import columns

SHOT_EVENT_TYPE = "Shot"
PENALTY_SHOT_TYPE = "Penalty"
RAW_ATTRIBUTE_COLUMNS = [
    "id",
    "shot_type",
    "shot_technique",
    "shot_first_time",
    "shot_aerial_won",
    "play_pattern",
    "under_pressure",
    "shot_statsbomb_xg",
]


def build_shot_attributes(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Per-shot attributes recorded by StatsBomb on the shot event itself.

    StatsBomb writes boolean qualifiers only when true, so a missing value means False.
    Categorical qualifiers that are missing become ``columns.MISSING_CATEGORY``.

    Deliberately excluded: ``shot_end_location`` (the outcome itself), and
    ``shot_one_on_one`` / ``shot_open_goal`` (analyst judgements that restate the
    freeze-frame by hand). ``shot_statsbomb_xg`` is kept only as a reference column.
    """
    # reindex because StatsBomb omits columns that no event in the match uses.
    shots = events.loc[events["type"] == SHOT_EVENT_TYPE].reindex(columns=RAW_ATTRIBUTE_COLUMNS)
    return pd.DataFrame(
        {
            columns.MATCH_ID: np.full(len(shots), match_id, dtype=np.int64),
            columns.EVENT_ID: shots["id"].to_numpy(),
            columns.IS_PENALTY: (shots["shot_type"] == PENALTY_SHOT_TYPE).to_numpy(),
            columns.SHOT_TECHNIQUE: _category(shots["shot_technique"]),
            columns.IS_FIRST_TIME: _flag(shots["shot_first_time"]),
            columns.IS_AERIAL_WON: _flag(shots["shot_aerial_won"]),
            columns.PLAY_PATTERN: _category(shots["play_pattern"]),
            columns.IS_UNDER_PRESSURE: _flag(shots["under_pressure"]),
            columns.STATSBOMB_XG: shots["shot_statsbomb_xg"].to_numpy(dtype=np.float64),
        }
    )


def _flag(values: pd.Series) -> NDArray[np.bool_]:
    return values.astype(object).eq(True).to_numpy(dtype=bool)


def _category(values: pd.Series) -> NDArray[np.object_]:
    return values.astype(object).fillna(columns.MISSING_CATEGORY).astype(str).to_numpy(dtype=object)
