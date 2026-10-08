import numpy as np

from shot_probability.geometry import angle_to_goal, distance_to_goal


def test_penalty_spot_distance_is_twelve() -> None:
    assert distance_to_goal(np.array([108.0]), np.array([40.0]))[0] == 12.0


def test_penalty_spot_angle_matches_hand_derivation() -> None:
    # Worked by hand: 2 * atan(4 / 12) = atan2(96, 128).
    expected_angle = 2 * np.arctan(4 / 12)
    assert np.isclose(angle_to_goal(np.array([108.0]), np.array([40.0]))[0], expected_angle)


def test_angle_is_symmetric_around_goal_centre() -> None:
    upper = angle_to_goal(np.array([100.0]), np.array([50.0]))
    lower = angle_to_goal(np.array([100.0]), np.array([30.0]))
    assert np.isclose(upper, lower)


def test_angle_shrinks_with_distance() -> None:
    angles = angle_to_goal(np.array([80.0, 100.0]), np.array([40.0, 40.0]))
    assert angles[0] < angles[1]
