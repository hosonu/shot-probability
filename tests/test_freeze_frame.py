import numpy as np
import pandas as pd
import pytest

from shot_probability import columns
from shot_probability.freeze_frame import build_freeze_frame_features


def make_player(
    x: float, y: float, teammate: bool = False, position: str = "Centre Back"
) -> dict[str, object]:
    return {
        "location": [x, y],
        "teammate": teammate,
        "position": {"id": 0, "name": position},
    }


def make_shot_event(
    event_id: str, location: list[float] | None, freeze_frame: list[dict[str, object]] | None
) -> dict[str, object]:
    return {
        "id": event_id,
        "type": "Shot",
        "location": location,
        "shot_freeze_frame": freeze_frame,
    }


def build(events: pd.DataFrame) -> pd.DataFrame:
    return build_freeze_frame_features(events, match_id=1)


def test_single_shot_extracts_goalkeeper_distance_offset_and_shot_distance() -> None:
    frame = [
        make_player(118.0, 42.0, teammate=False, position="Goalkeeper"),
        make_player(90.0, 30.0, teammate=False),
        make_player(95.0, 50.0, teammate=True),
    ]
    events = pd.DataFrame([make_shot_event("shot-1", [100.0, 40.0], frame)])

    features = build(events)

    assert features[columns.HAS_FREEZE_FRAME].iloc[0]
    assert features[columns.HAS_GOALKEEPER].iloc[0]
    goalkeeper_distance = features[columns.GOALKEEPER_DISTANCE_TO_GOAL].iloc[0]
    assert goalkeeper_distance == pytest.approx(np.hypot(2.0, 2.0))
    assert features[columns.GOALKEEPER_Y_OFFSET].iloc[0] == pytest.approx(2.0)
    assert features[columns.SHOT_TO_GOALKEEPER_DISTANCE].iloc[0] == pytest.approx(
        np.hypot(18.0, 2.0)
    )


def test_nearest_opponent_distance_ignores_the_goalkeeper() -> None:
    frame = [
        make_player(100.5, 40.0, teammate=False, position="Goalkeeper"),
        make_player(95.0, 40.0, teammate=False),
        make_player(80.0, 40.0, teammate=True),
    ]
    events = pd.DataFrame([make_shot_event("shot-1", [90.0, 40.0], frame)])

    features = build(events)

    assert features[columns.NEAREST_OPPONENT_DISTANCE].iloc[0] == pytest.approx(5.0)


def test_opponents_and_teammates_inside_the_shot_triangle_are_counted_separately() -> None:
    frame = [
        make_player(119.0, 40.0, teammate=False, position="Goalkeeper"),
        make_player(115.0, 40.0, teammate=False),
        make_player(115.0, 39.0, teammate=False),
        make_player(115.0, 41.0, teammate=True),
        make_player(50.0, 10.0, teammate=False),
    ]
    events = pd.DataFrame([make_shot_event("shot-1", [100.0, 40.0], frame)])

    features = build(events)

    assert features[columns.OPPONENTS_IN_SHOT_TRIANGLE].iloc[0] == 2
    assert features[columns.TEAMMATES_IN_SHOT_TRIANGLE].iloc[0] == 1


def test_opponents_between_shot_and_goal_excludes_goalkeeper_and_players_behind() -> None:
    frame = [
        make_player(119.0, 40.0, teammate=False, position="Goalkeeper"),
        make_player(110.0, 20.0, teammate=False),
        make_player(105.0, 60.0, teammate=False),
        make_player(80.0, 40.0, teammate=False),
    ]
    events = pd.DataFrame([make_shot_event("shot-1", [100.0, 40.0], frame)])

    features = build(events)

    assert features[columns.OPPONENTS_BETWEEN_SHOT_AND_GOAL].iloc[0] == 2


def test_missing_frame_gets_sentinel_zeros_and_flags_off() -> None:
    events = pd.DataFrame(
        [
            make_shot_event("shot-1", [100.0, 40.0], None),
            make_shot_event(
                "shot-2",
                [100.0, 40.0],
                [make_player(110.0, 30.0, teammate=False)],
            ),
        ]
    )

    features = build(events).sort_values(columns.EVENT_ID).reset_index(drop=True)

    assert features[columns.HAS_FREEZE_FRAME].tolist() == [False, True]
    assert not features[columns.HAS_GOALKEEPER].iloc[0]
    assert features[columns.GOALKEEPER_DISTANCE_TO_GOAL].iloc[0] == 0.0
    assert features[columns.NEAREST_OPPONENT_DISTANCE].iloc[0] == 0.0
    nearest = features[columns.NEAREST_OPPONENT_DISTANCE].iloc[1]
    assert nearest == pytest.approx(np.hypot(10.0, 10.0))


def test_match_without_shots_yields_an_empty_table_with_the_full_schema() -> None:
    events = pd.DataFrame(
        [{"id": "pass-1", "type": "Pass", "location": [50.0, 40.0], "shot_freeze_frame": None}]
    )

    features = build(events)

    assert features.empty
    assert set(columns.FREEZE_FRAME_SCALAR_COLUMNS) <= set(features.columns)


def test_has_freeze_frame_flag_is_false_when_the_frame_is_absent() -> None:
    events = pd.DataFrame([make_shot_event("shot-1", [100.0, 40.0], None)])

    features = build(events)

    assert not features[columns.HAS_FREEZE_FRAME].iloc[0]
    assert not features[columns.HAS_GOALKEEPER].iloc[0]
