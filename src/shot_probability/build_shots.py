import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

from shot_probability import columns
from shot_probability.config import DataConfig
from shot_probability.freeze_frame import build_freeze_frame_features
from shot_probability.geometry import angle_to_goal, distance_to_goal
from shot_probability.pre_shot import build_pre_shot_context
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
RAW_SHOT_COLUMNS = ["id", "location", "shot_body_part", "shot_outcome", "shot_type"]


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
            columns.EVENT_ID: located_shots["id"],
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


def fetch_match(
    match_id: int, sequence_length: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = sb.events(match_id=match_id)
    return (
        extract_shots(events, match_id),
        build_pre_shot_context(events, match_id, sequence_length),
        build_freeze_frame_features(events, match_id),
    )


def fetch_raw_tables(
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = sb.matches(competition_id=config.competition_id, season_id=config.season_id)
    match_ids: list[int] = sorted(int(match_id) for match_id in matches["match_id"])
    logger.info("Fetching events for %d matches", len(match_ids))
    with ThreadPoolExecutor(max_workers=config.fetch_workers) as executor:
        per_match_tables = list(
            executor.map(lambda match_id: fetch_match(match_id, config.sequence_length), match_ids)
        )
    raw_shots = pd.concat([shots for shots, _, _ in per_match_tables], ignore_index=True)
    context = pd.concat([context for _, context, _ in per_match_tables], ignore_index=True)
    freeze_frame = pd.concat([frame for _, _, frame in per_match_tables], ignore_index=True)
    return raw_shots, context, freeze_frame


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def main() -> None:
    configure_logging()
    # Open data is intentionally accessed without credentials; the warning repeats for every match.
    warnings.simplefilter("ignore", category=NoAuthWarning)
    config = DataConfig()
    raw_shots, context, freeze_frame = fetch_raw_tables(config)
    shot_table, dropped_count = to_shot_table(raw_shots)
    write_table(shot_table, config.shots_path)
    write_table(context, config.context_path)
    write_table(freeze_frame, config.freeze_frame_path)
    logger.info(
        "Wrote %s: rows=%d dropped=%d goal_rate=%.4f",
        config.shots_path,
        len(shot_table),
        dropped_count,
        shot_table[columns.GOAL].mean(),
    )


if __name__ == "__main__":
    main()
