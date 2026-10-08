import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

from shot_probability import columns
from shot_probability.config import DataConfig
from shot_probability.geometry import angle_to_goal, distance_to_goal
from shot_probability.runtime import configure_logging

logger = logging.getLogger(__name__)

SHOT_EVENT_TYPE = "Shot"
GOAL_OUTCOME = "Goal"
SET_PIECE_SHOT_TYPES = frozenset({"Corner", "Free Kick", "Penalty", "Kick Off"})
BODY_PART_MAPPING = {
    "Right Foot": columns.BODY_PART_FOOT,
    "Left Foot": columns.BODY_PART_FOOT,
    "Head": columns.BODY_PART_HEAD,
}
RAW_SHOT_COLUMNS = ["location", "shot_body_part", "shot_outcome", "shot_type"]


def extract_shots(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    # reindex because StatsBomb omits shot_* columns for matches without any shot.
    return (
        events.loc[events["type"] == SHOT_EVENT_TYPE]
        .reindex(columns=RAW_SHOT_COLUMNS)
        .assign(**{columns.MATCH_ID: match_id})
    )


def to_shot_table(raw_shots: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build the modelling table and count shots dropped for a missing location."""
    located_shots = raw_shots.dropna(subset=["location"])
    dropped_count = len(raw_shots) - len(located_shots)
    coordinates = pd.DataFrame(
        located_shots["location"].tolist(),
        columns=["x", "y"],
        index=located_shots.index,
        dtype=float,
    )
    x_values = coordinates["x"].to_numpy()
    y_values = coordinates["y"].to_numpy()

    shot_table = pd.DataFrame(
        {
            columns.MATCH_ID: located_shots[columns.MATCH_ID].astype("int64"),
            columns.DISTANCE_TO_GOAL: distance_to_goal(x_values, y_values),
            columns.ANGLE_TO_GOAL: angle_to_goal(x_values, y_values),
            columns.BODY_PART: located_shots["shot_body_part"]
            .map(BODY_PART_MAPPING)
            .fillna(columns.BODY_PART_OTHER),
            columns.IS_SET_PIECE: located_shots["shot_type"].isin(SET_PIECE_SHOT_TYPES),
            columns.GOAL: (located_shots["shot_outcome"] == GOAL_OUTCOME).astype("int64"),
        },
        index=located_shots.index,
    )
    return shot_table.reset_index(drop=True), dropped_count


def fetch_match_shots(match_id: int) -> pd.DataFrame:
    return extract_shots(sb.events(match_id=match_id), match_id)


def fetch_raw_shots(config: DataConfig) -> pd.DataFrame:
    matches = sb.matches(competition_id=config.competition_id, season_id=config.season_id)
    match_ids: list[int] = sorted(int(match_id) for match_id in matches["match_id"])
    logger.info("Fetching events for %d matches", len(match_ids))
    with ThreadPoolExecutor(max_workers=config.fetch_workers) as executor:
        per_match_shots = list(executor.map(fetch_match_shots, match_ids))
    return pd.concat(per_match_shots, ignore_index=True)


def write_shot_table(shot_table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shot_table.to_parquet(path, index=False)


def main() -> None:
    configure_logging()
    # Open data is intentionally accessed without credentials; the warning repeats for every match.
    warnings.simplefilter("ignore", category=NoAuthWarning)
    config = DataConfig()
    raw_shots = fetch_raw_shots(config)
    shot_table, dropped_count = to_shot_table(raw_shots)
    write_shot_table(shot_table, config.shots_path)
    logger.info(
        "Wrote %s: rows=%d dropped=%d goal_rate=%.4f",
        config.shots_path,
        len(shot_table),
        dropped_count,
        shot_table[columns.GOAL].mean(),
    )


if __name__ == "__main__":
    main()
