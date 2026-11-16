import logging
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

from shot_probability import columns
from shot_probability.build_shots import RawShotTables, fetch_raw_tables, to_shot_table, write_table
from shot_probability.config import LeagueConfig, MultiLeagueDataConfig
from shot_probability.runtime import configure_logging

logger = logging.getLogger(__name__)

KEY_COLUMNS = [columns.MATCH_ID, columns.EVENT_ID]
MULTI_LEAGUE_SHOT_COLUMNS: list[str] = [*columns.SHOT_TABLE_COLUMNS, columns.LEAGUE]


def resolve_competition_ids(
    competitions: pd.DataFrame, leagues: tuple[LeagueConfig, ...], season_name: str
) -> dict[str, tuple[int, int]]:
    """Map each league name to StatsBomb's ``(competition_id, season_id)``.

    Fails unless exactly one row of ``sb.competitions()`` matches, so a renamed or
    duplicated competition cannot silently pull the wrong data.
    """
    resolved: dict[str, tuple[int, int]] = {}
    for league in leagues:
        matching = competitions.loc[
            (competitions["competition_name"] == league.competition_name)
            & (competitions["country_name"] == league.country_name)
            & (competitions["season_name"] == season_name)
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Expected one competition for {league.competition_name} "
                f"({league.country_name}) {season_name}, found {len(matching)}"
            )
        row = matching.iloc[0]
        resolved[league.competition_name] = (int(row["competition_id"]), int(row["season_id"]))
    return resolved


def tag_league(raw_tables: RawShotTables, league_name: str) -> tuple[pd.DataFrame, int]:
    """Shot table with the ``league`` column, plus the count dropped for a missing location."""
    shot_table, dropped_count = to_shot_table(raw_tables.shots)
    return shot_table.assign(**{columns.LEAGUE: league_name}), dropped_count


def summarize_leagues(shots: pd.DataFrame, freeze_frame: pd.DataFrame) -> pd.DataFrame:
    """Matches, shots, goal rate and share of shots without a freeze frame, per league."""
    with_frame = shots.merge(
        freeze_frame[[*KEY_COLUMNS, columns.HAS_FREEZE_FRAME]], on=KEY_COLUMNS, how="left"
    )
    missing_frame = ~with_frame[columns.HAS_FREEZE_FRAME].fillna(False).astype(bool)
    return (
        with_frame.assign(missing_freeze_frame=missing_frame)
        .groupby(columns.LEAGUE, sort=False)
        .agg(
            matches=(columns.MATCH_ID, "nunique"),
            shots=(columns.EVENT_ID, "size"),
            goals=(columns.GOAL, "sum"),
            goal_rate=(columns.GOAL, "mean"),
            missing_freeze_frame_rate=("missing_freeze_frame", "mean"),
        )
        .reset_index()
    )


def find_legacy_mismatches(
    multi_league_tables: dict[str, pd.DataFrame],
    legacy_tables: dict[str, pd.DataFrame],
    legacy_league: str,
) -> list[str]:
    """Names of tables whose legacy-league rows differ from the published single-league files.

    Comparison is exact (every digit), after sorting both sides by match and event id.
    """
    legacy_shots = multi_league_tables["shots"]
    legacy_match_ids = legacy_shots.loc[
        legacy_shots[columns.LEAGUE] == legacy_league, columns.MATCH_ID
    ].unique()
    mismatches = []
    for table_name, legacy_table in legacy_tables.items():
        candidate = multi_league_tables[table_name]
        candidate = candidate.loc[candidate[columns.MATCH_ID].isin(legacy_match_ids)]
        candidate = candidate.drop(columns=[columns.LEAGUE], errors="ignore")
        try:
            pd.testing.assert_frame_equal(
                _sorted_by_key(candidate), _sorted_by_key(legacy_table), check_exact=True
            )
        except AssertionError:
            mismatches.append(table_name)
    return mismatches


def _sorted_by_key(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def read_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {table_name: pd.read_parquet(path) for table_name, path in paths.items()}


def build_multi_league_tables(config: MultiLeagueDataConfig) -> dict[str, pd.DataFrame]:
    competition_ids = resolve_competition_ids(sb.competitions(), config.leagues, config.season_name)
    shot_tables, contexts, freeze_frames, attributes = [], [], [], []
    for league_name, (competition_id, season_id) in competition_ids.items():
        logger.info(
            "League %s: competition_id=%d season_id=%d", league_name, competition_id, season_id
        )
        raw_tables = fetch_raw_tables(
            competition_id, season_id, config.fetch_workers, config.sequence_length
        )
        shot_table, dropped_count = tag_league(raw_tables, league_name)
        logger.info("League %s: dropped %d shots without a location", league_name, dropped_count)
        shot_tables.append(shot_table)
        contexts.append(raw_tables.context)
        freeze_frames.append(raw_tables.freeze_frame)
        attributes.append(raw_tables.attributes)
    return {
        "shots": pd.concat(shot_tables, ignore_index=True)[MULTI_LEAGUE_SHOT_COLUMNS],
        "context": pd.concat(contexts, ignore_index=True),
        "freeze_frame": pd.concat(freeze_frames, ignore_index=True),
        "attributes": pd.concat(attributes, ignore_index=True),
    }


def main() -> None:
    configure_logging()
    warnings.simplefilter("ignore", category=NoAuthWarning)
    config = MultiLeagueDataConfig()
    output_paths = {
        "shots": config.shots_path,
        "context": config.context_path,
        "freeze_frame": config.freeze_frame_path,
        "attributes": config.attributes_path,
    }
    for table_name, table in build_multi_league_tables(config).items():
        write_table(table, output_paths[table_name])

    # Compare what was written, after the parquet round trip, with the published files.
    written = read_tables(output_paths)
    summary = summarize_leagues(written["shots"], written["freeze_frame"])
    logger.info("Per-league summary:\n%s", summary.to_string(index=False))

    legacy_paths = {
        "shots": config.legacy_shots_path,
        "context": config.legacy_context_path,
        "freeze_frame": config.legacy_freeze_frame_path,
    }
    mismatches = find_legacy_mismatches(written, read_tables(legacy_paths), config.legacy_league)
    if mismatches:
        raise ValueError(
            f"{config.legacy_league} rows differ from the published files: {mismatches}"
        )
    logger.info(
        "%s rows reproduce %s exactly",
        config.legacy_league,
        ", ".join(map(str, legacy_paths.values())),
    )


if __name__ == "__main__":
    main()
