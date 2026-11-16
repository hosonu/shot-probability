import pandas as pd
import pytest

from shot_probability import columns
from shot_probability.build_multi_league import (
    find_legacy_mismatches,
    resolve_competition_ids,
    summarize_leagues,
)
from shot_probability.config import LeagueConfig
from synthetic import make_benchmark_table

LEAGUES = (LeagueConfig("La Liga", "Spain"), LeagueConfig("Serie A", "Italy"))


def make_competitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition_id": [11, 11, 12, 99],
            "season_id": [27, 4, 27, 27],
            "competition_name": ["La Liga", "La Liga", "Serie A", "Serie A"],
            "country_name": ["Spain", "Spain", "Italy", "Brazil"],
            "season_name": ["2015/2016", "2018/2019", "2015/2016", "2015/2016"],
        }
    )


def test_ids_are_resolved_from_names_country_and_season() -> None:
    resolved = resolve_competition_ids(make_competitions(), LEAGUES, "2015/2016")

    assert resolved == {"La Liga": (11, 27), "Serie A": (12, 27)}


def test_unknown_league_fails_instead_of_fetching_something_else() -> None:
    with pytest.raises(ValueError, match="found 0"):
        resolve_competition_ids(make_competitions(), LEAGUES, "1999/2000")


def make_multi_league_tables() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    table = make_benchmark_table(shot_count=40)
    table[columns.LEAGUE] = (table[columns.MATCH_ID] < 4).map({True: "La Liga", False: "Serie A"})
    key = [columns.MATCH_ID, columns.EVENT_ID]
    freeze_frame = table[key + columns.FREEZE_FRAME_SCALAR_COLUMNS]
    multi = {
        "shots": table[[*columns.SHOT_TABLE_COLUMNS, columns.LEAGUE]],
        "freeze_frame": freeze_frame,
    }
    is_legacy = table[columns.LEAGUE] == "La Liga"
    # The published files hold only the legacy league, in a different row order.
    legacy = {
        "shots": table.loc[is_legacy, columns.SHOT_TABLE_COLUMNS].iloc[::-1],
        "freeze_frame": freeze_frame.loc[is_legacy],
    }
    return multi, legacy


def test_identical_legacy_rows_report_no_mismatch() -> None:
    multi, legacy = make_multi_league_tables()

    assert find_legacy_mismatches(multi, legacy, "La Liga") == []


def test_a_single_changed_digit_is_reported() -> None:
    multi, legacy = make_multi_league_tables()
    shots = multi["shots"].copy()
    first_legacy_row = shots.index[shots[columns.LEAGUE] == "La Liga"][0]
    shots.loc[first_legacy_row, columns.DISTANCE_TO_GOAL] += 1e-12
    multi["shots"] = shots

    assert find_legacy_mismatches(multi, legacy, "La Liga") == ["shots"]


def test_summary_counts_shots_goals_and_missing_freeze_frames_per_league() -> None:
    multi, _ = make_multi_league_tables()
    freeze_frame = multi["freeze_frame"].copy()
    freeze_frame.loc[freeze_frame.index[:2], columns.HAS_FREEZE_FRAME] = False

    summary = summarize_leagues(multi["shots"], freeze_frame).set_index(columns.LEAGUE)

    assert summary.loc["La Liga", "shots"] == 20
    assert summary.loc["Serie A", "shots"] == 20
    assert summary.loc["La Liga", "missing_freeze_frame_rate"] == pytest.approx(0.1)
    assert summary.loc["Serie A", "missing_freeze_frame_rate"] == 0.0
    assert summary["goals"].sum() == multi["shots"][columns.GOAL].sum()
