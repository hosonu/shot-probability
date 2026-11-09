import numpy as np
import pandas as pd
import pytest

from shot_probability import columns
from shot_probability.pre_shot import build_pre_shot_context

SEQUENCE_LENGTH = 3
ATTACKING_TEAM = "Home"
DEFENDING_TEAM = "Away"


def make_event(
    event_index: int,
    event_type: str,
    seconds: float,
    location: list[float] | None,
    team: str = ATTACKING_TEAM,
    possession: int = 1,
) -> dict[str, object]:
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "possession": possession,
        "team": team,
        "type": event_type,
        "timestamp": f"00:00:{seconds:06.3f}",
        "location": location,
    }


def make_possession_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_event(1, "Pass", 1.0, [60.0, 40.0]),
            make_event(2, "Pressure", 2.0, [62.0, 40.0], team=DEFENDING_TEAM),
            make_event(3, "Ball Receipt*", 3.0, [80.0, 30.0]),
            make_event(4, "Carry", 4.0, [80.0, 30.0]),
            make_event(5, "Pass", 6.0, [95.0, 35.0]),
            make_event(6, "Shot", 8.0, [108.0, 38.0]),
            make_event(7, "Goal Keeper", 9.0, [1.0, 40.0], team=DEFENDING_TEAM),
            make_event(8, "Pass", 12.0, [30.0, 20.0]),
        ]
    )


def build(events: pd.DataFrame) -> pd.DataFrame:
    return build_pre_shot_context(events, match_id=7, sequence_length=SEQUENCE_LENGTH)


def lag_types(context: pd.DataFrame) -> list[object]:
    return (
        context[columns.sequence_columns(columns.SEQUENCE_EVENT_TYPE, SEQUENCE_LENGTH)]
        .iloc[0]
        .tolist()
    )


def test_context_holds_the_latest_events_before_the_shot_newest_first() -> None:
    context = build(make_possession_events())

    assert lag_types(context) == ["Pass", "Carry", "Ball Receipt*"]
    assert context[columns.LAST_EVENT_TYPE].tolist() == ["Pass"]
    assert context[columns.EVENT_ID].tolist() == ["event-6"]


def test_row_order_of_the_input_does_not_matter() -> None:
    events = make_possession_events()
    shuffled_events = events.sample(frac=1.0, random_state=0)

    pd.testing.assert_frame_equal(build(events), build(shuffled_events))


def test_context_is_identical_when_every_event_after_the_shot_is_deleted() -> None:
    events = make_possession_events()
    events_until_shot = events.loc[events["index"] <= 6]

    pd.testing.assert_frame_equal(build(events), build(events_until_shot))


def test_context_is_identical_when_every_event_after_the_shot_is_corrupted() -> None:
    events = make_possession_events()
    corrupted_events = events.copy()
    is_after_shot = corrupted_events["index"] > 6
    corrupted_events.loc[is_after_shot, "type"] = "Pass"
    corrupted_events.loc[is_after_shot, "team"] = ATTACKING_TEAM
    corrupted_events.loc[is_after_shot, "timestamp"] = "00:00:08.500"
    corrupted_events["location"] = corrupted_events["location"].where(
        ~is_after_shot, pd.Series([[119.0, 40.0]] * len(corrupted_events))
    )

    pd.testing.assert_frame_equal(build(events), build(corrupted_events))


def test_later_possessions_do_not_leak_into_an_earlier_shot() -> None:
    events = make_possession_events()
    later_possession = pd.DataFrame(
        [
            make_event(9 + offset, "Pass", 20.0 + offset, [50.0, 40.0], possession=2)
            for offset in range(4)
        ]
    )

    pd.testing.assert_frame_equal(build(events), build(pd.concat([events, later_possession])))


def test_earlier_possessions_are_not_part_of_the_context() -> None:
    previous_possession = pd.DataFrame(
        [make_event(0, "Pass", 0.0, [10.0, 10.0], team=DEFENDING_TEAM, possession=0)]
    )
    events = pd.concat([previous_possession, make_possession_events()])

    pd.testing.assert_frame_equal(build(events), build(make_possession_events()))


def test_a_second_shot_sees_the_first_shot_but_never_itself_or_later_events() -> None:
    events = pd.concat(
        [
            make_possession_events(),
            pd.DataFrame(
                [
                    make_event(9, "Ball Recovery", 13.0, [100.0, 40.0]),
                    make_event(10, "Shot", 14.0, [105.0, 40.0]),
                    make_event(11, "Goal Keeper", 15.0, [1.0, 40.0], team=DEFENDING_TEAM),
                ]
            ),
        ]
    )
    context = build(events)
    second_shot = context.loc[context[columns.EVENT_ID] == "event-10"]

    assert lag_types(second_shot) == ["Ball Recovery", "Pass", "Goal Keeper"]
    assert second_shot[columns.N_PRIOR_EVENTS].iloc[0] == 9


def test_missing_lags_are_padded_when_the_possession_is_short() -> None:
    short_possession = make_possession_events().loc[lambda frame: frame["index"].isin([5, 6])]

    context = build(short_possession)

    assert lag_types(context) == ["Pass", None, None]
    sequence_x = context[columns.sequence_columns(columns.SEQUENCE_X, SEQUENCE_LENGTH)].iloc[0]
    assert sequence_x.iloc[0] == 95.0
    assert sequence_x.iloc[1:].isna().all()
    assert context[columns.N_PRIOR_EVENTS].iloc[0] == 1


def test_shot_that_opens_the_possession_has_no_context() -> None:
    opening_shot = pd.DataFrame([make_event(1, "Shot", 5.0, [100.0, 40.0])])

    context = build(opening_shot)

    assert context[columns.N_PRIOR_EVENTS].iloc[0] == 0
    assert context[columns.LAST_EVENT_TYPE].iloc[0] == columns.NO_PRIOR_EVENT
    assert context[columns.SECONDS_SINCE_PREVIOUS_EVENT].iloc[0] == 0.0
    assert context[columns.PROGRESSION_IN_WINDOW].iloc[0] == 0.0
    assert lag_types(context) == [None, None, None]


def test_scalar_features_count_every_earlier_event_even_beyond_the_window() -> None:
    context = build(make_possession_events())

    assert context[columns.N_PRIOR_EVENTS].iloc[0] == 5
    assert context[columns.N_PRIOR_PASSES].iloc[0] == 2
    assert context[columns.N_PRIOR_CARRIES].iloc[0] == 1
    assert context[columns.N_PRIOR_DEFENDING_EVENTS].iloc[0] == 1
    assert context[columns.POSSESSION_SECONDS_BEFORE_SHOT].iloc[0] == pytest.approx(7.0)
    assert context[columns.SECONDS_SINCE_PREVIOUS_EVENT].iloc[0] == pytest.approx(2.0)


def test_sequence_tokens_carry_team_side_and_time_since_their_own_predecessor() -> None:
    context = build(make_possession_events())
    attacking = context[columns.sequence_columns(columns.SEQUENCE_IS_ATTACKING, SEQUENCE_LENGTH)]
    seconds = context[columns.sequence_columns(columns.SEQUENCE_SECONDS, SEQUENCE_LENGTH)]

    assert attacking.iloc[0].tolist() == [True, True, True]
    assert seconds.iloc[0].tolist() == pytest.approx([2.0, 1.0, 1.0])


def test_progression_is_the_distance_gained_since_the_oldest_located_event() -> None:
    context = build(make_possession_events())

    oldest = np.hypot(120.0 - 80.0, 40.0 - 30.0)
    shot = np.hypot(120.0 - 108.0, 40.0 - 38.0)
    assert context[columns.PROGRESSION_IN_WINDOW].iloc[0] == pytest.approx(oldest - shot)


def test_events_without_location_do_not_break_the_context() -> None:
    events = make_possession_events()
    events["location"] = events["location"].where(events["index"] != 5, None)

    context = build(events)

    sequence_x = context[columns.sequence_columns(columns.SEQUENCE_X, SEQUENCE_LENGTH)]
    assert pd.isna(sequence_x.iloc[0, 0])
    assert lag_types(context) == ["Pass", "Carry", "Ball Receipt*"]


def test_match_without_shots_yields_an_empty_table_with_the_full_schema() -> None:
    events = make_possession_events().loc[lambda frame: frame["type"] != "Shot"]

    context = build(events)

    assert context.empty
    assert set(columns.CONTEXT_SCALAR_COLUMNS) <= set(context.columns)


def test_attacking_side_follows_the_shooter_even_when_it_is_not_the_possession_team() -> None:
    events = pd.DataFrame(
        [
            make_event(1, "Pass", 1.0, [60.0, 40.0], team=DEFENDING_TEAM),
            make_event(2, "Pressure", 2.0, [62.0, 40.0], team=ATTACKING_TEAM),
            make_event(3, "Shot", 3.0, [100.0, 40.0], team=DEFENDING_TEAM),
        ]
    )

    context = build(events)

    attacking = context[columns.sequence_columns(columns.SEQUENCE_IS_ATTACKING, SEQUENCE_LENGTH)]
    assert attacking.iloc[0].tolist() == [False, True, False]
    assert context[columns.N_PRIOR_DEFENDING_EVENTS].iloc[0] == 1
    assert context[columns.N_PRIOR_PASSES].iloc[0] == 1
