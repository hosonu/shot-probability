import pandas as pd

from shot_probability import columns
from shot_probability.build_shots import extract_shots, to_shot_table


def make_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["Pass", "Shot", "Shot", "Shot", "Shot"],
            "location": [[50.0, 40.0], [108.0, 40.0], [100.0, 30.0], None, [110.0, 38.0]],
            "shot_body_part": [None, "Right Foot", "Head", "Left Foot", "Other"],
            "shot_outcome": [None, "Goal", "Saved", "Goal", "Blocked"],
            "shot_type": [None, "Penalty", "Open Play", "Open Play", "Corner"],
        }
    )


def test_extract_shots_keeps_only_shots_and_tags_match() -> None:
    raw_shots = extract_shots(make_events(), match_id=7)

    assert len(raw_shots) == 4
    assert (raw_shots[columns.MATCH_ID] == 7).all()


def test_extract_shots_handles_match_without_shots() -> None:
    passes_only = pd.DataFrame({"type": ["Pass"], "location": [[10.0, 10.0]]})

    assert extract_shots(passes_only, match_id=1).empty


def test_to_shot_table_schema_labels_and_dropped_count() -> None:
    shot_table, dropped_count = to_shot_table(extract_shots(make_events(), match_id=7))

    assert dropped_count == 1
    assert list(shot_table.columns) == columns.SHOT_TABLE_COLUMNS
    assert shot_table[columns.GOAL].tolist() == [1, 0, 0]
    assert shot_table[columns.BODY_PART].tolist() == ["foot", "head", "other"]
    assert shot_table[columns.IS_SET_PIECE].tolist() == [True, False, True]
