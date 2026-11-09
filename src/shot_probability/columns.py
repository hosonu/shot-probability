from dataclasses import dataclass

MATCH_ID = "match_id"
EVENT_ID = "event_id"
DISTANCE_TO_GOAL = "distance_to_goal"
ANGLE_TO_GOAL = "angle_to_goal"
BODY_PART = "body_part"
IS_SET_PIECE = "is_set_piece"
GOAL = "goal"

HAS_FREEZE_FRAME = "has_freeze_frame"
HAS_GOALKEEPER = "has_goalkeeper"
GOALKEEPER_DISTANCE_TO_GOAL = "goalkeeper_distance_to_goal"
GOALKEEPER_Y_OFFSET = "goalkeeper_y_offset"
SHOT_TO_GOALKEEPER_DISTANCE = "shot_to_goalkeeper_distance"
NEAREST_OPPONENT_DISTANCE = "nearest_opponent_distance"
OPPONENTS_IN_SHOT_TRIANGLE = "opponents_in_shot_triangle"
TEAMMATES_IN_SHOT_TRIANGLE = "teammates_in_shot_triangle"
OPPONENTS_BETWEEN_SHOT_AND_GOAL = "opponents_between_shot_and_goal"

N_PRIOR_EVENTS = "n_prior_events"
N_PRIOR_PASSES = "n_prior_passes"
N_PRIOR_CARRIES = "n_prior_carries"
N_PRIOR_DEFENDING_EVENTS = "n_prior_defending_events"
POSSESSION_SECONDS_BEFORE_SHOT = "possession_seconds_before_shot"
SECONDS_SINCE_PREVIOUS_EVENT = "seconds_since_previous_event"
PROGRESSION_IN_WINDOW = "progression_in_window"
LAST_EVENT_TYPE = "last_event_type"

BODY_PART_FOOT = "foot"
BODY_PART_HEAD = "head"
BODY_PART_OTHER = "other"
NO_PRIOR_EVENT = "none"

SEQUENCE_EVENT_TYPE = "type"
SEQUENCE_X = "x"
SEQUENCE_Y = "y"
SEQUENCE_SECONDS = "seconds_since_previous"
SEQUENCE_IS_ATTACKING = "is_attacking_team"
SEQUENCE_FIELDS: tuple[str, ...] = (
    SEQUENCE_EVENT_TYPE,
    SEQUENCE_X,
    SEQUENCE_Y,
    SEQUENCE_SECONDS,
    SEQUENCE_IS_ATTACKING,
)


@dataclass(frozen=True)
class FeatureSet:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]

    @property
    def names(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.boolean]


SHOT_FEATURES = FeatureSet(
    numeric=(DISTANCE_TO_GOAL, ANGLE_TO_GOAL),
    categorical=(BODY_PART,),
    boolean=(IS_SET_PIECE,),
)
CONTEXT_FEATURES = FeatureSet(
    numeric=(
        *SHOT_FEATURES.numeric,
        N_PRIOR_EVENTS,
        N_PRIOR_PASSES,
        N_PRIOR_CARRIES,
        N_PRIOR_DEFENDING_EVENTS,
        POSSESSION_SECONDS_BEFORE_SHOT,
        SECONDS_SINCE_PREVIOUS_EVENT,
        PROGRESSION_IN_WINDOW,
    ),
    categorical=(*SHOT_FEATURES.categorical, LAST_EVENT_TYPE),
    boolean=SHOT_FEATURES.boolean,
)
CONTEXT_SCALAR_COLUMNS: list[str] = [
    column for column in CONTEXT_FEATURES.names if column not in SHOT_FEATURES.names
]

FREEZE_FRAME_FEATURES = FeatureSet(
    numeric=(
        *SHOT_FEATURES.numeric,
        GOALKEEPER_DISTANCE_TO_GOAL,
        GOALKEEPER_Y_OFFSET,
        SHOT_TO_GOALKEEPER_DISTANCE,
        NEAREST_OPPONENT_DISTANCE,
        OPPONENTS_IN_SHOT_TRIANGLE,
        TEAMMATES_IN_SHOT_TRIANGLE,
        OPPONENTS_BETWEEN_SHOT_AND_GOAL,
    ),
    categorical=SHOT_FEATURES.categorical,
    boolean=(*SHOT_FEATURES.boolean, HAS_FREEZE_FRAME, HAS_GOALKEEPER),
)
FREEZE_FRAME_SCALAR_COLUMNS: list[str] = [
    column for column in FREEZE_FRAME_FEATURES.names if column not in SHOT_FEATURES.names
]

SHOT_TABLE_COLUMNS: list[str] = [MATCH_ID, EVENT_ID, *SHOT_FEATURES.names, GOAL]


def sequence_column(field: str, lag: int) -> str:
    """Column holding `field` of the event `lag` steps before the shot (lag 1 is the latest)."""
    return f"context_{field}_{lag}"


def sequence_columns(field: str, sequence_length: int) -> list[str]:
    return [sequence_column(field, lag) for lag in range(1, sequence_length + 1)]
