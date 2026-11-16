import numpy as np
import pandas as pd

from shot_probability import columns
from shot_probability.shot_attributes import build_shot_attributes


def make_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["pass", "penalty", "free-kick", "volley", "header"],
            "type": ["Pass", "Shot", "Shot", "Shot", "Shot"],
            "shot_type": [None, "Penalty", "Free Kick", "Open Play", "Open Play"],
            "shot_technique": [None, "Normal", "Normal", "Volley", None],
            "shot_first_time": [None, None, None, True, None],
            "shot_aerial_won": [None, None, None, None, True],
            "play_pattern": ["Regular Play", "Other", "From Free Kick", "From Corner", None],
            "under_pressure": [True, None, None, True, None],
            "shot_statsbomb_xg": [np.nan, 0.76, 0.05, 0.12, 0.30],
        }
    )


def test_penalty_is_separated_from_other_set_pieces() -> None:
    attributes = build_shot_attributes(make_events(), match_id=3)

    assert attributes[columns.EVENT_ID].tolist() == ["penalty", "free-kick", "volley", "header"]
    assert attributes[columns.IS_PENALTY].tolist() == [True, False, False, False]
    assert (attributes[columns.MATCH_ID] == 3).all()


def test_missing_boolean_qualifiers_mean_false() -> None:
    attributes = build_shot_attributes(make_events(), match_id=3)

    assert attributes[columns.IS_FIRST_TIME].tolist() == [False, False, True, False]
    assert attributes[columns.IS_AERIAL_WON].tolist() == [False, False, False, True]
    assert attributes[columns.IS_UNDER_PRESSURE].tolist() == [False, False, True, False]


def test_missing_categories_get_an_explicit_level() -> None:
    attributes = build_shot_attributes(make_events(), match_id=3)

    assert attributes[columns.SHOT_TECHNIQUE].tolist() == [
        "Normal",
        "Normal",
        "Volley",
        columns.MISSING_CATEGORY,
    ]
    assert attributes[columns.PLAY_PATTERN].tolist()[-1] == columns.MISSING_CATEGORY


def test_statsbomb_xg_is_kept_only_as_a_reference_column() -> None:
    attributes = build_shot_attributes(make_events(), match_id=3)

    assert attributes[columns.STATSBOMB_XG].tolist() == [0.76, 0.05, 0.12, 0.30]
    for group in columns.FEATURE_GROUPS.values():
        assert columns.STATSBOMB_XG not in group.names


def test_match_without_optional_columns_still_has_the_full_schema() -> None:
    events = pd.DataFrame({"id": ["shot"], "type": ["Shot"], "shot_type": ["Open Play"]})

    attributes = build_shot_attributes(events, match_id=1)

    assert set(columns.SHOT_ATTRIBUTE_COLUMNS) <= set(attributes.columns)
    assert not attributes[columns.IS_UNDER_PRESSURE].iloc[0]
    assert attributes[columns.PLAY_PATTERN].iloc[0] == columns.MISSING_CATEGORY
