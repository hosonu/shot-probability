import numpy as np
from numpy.typing import NDArray

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
POST_LOW_Y = 36.0
POST_HIGH_Y = 44.0

FloatArray = NDArray[np.float64]


def distance_to_goal(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.hypot(GOAL_X - x, GOAL_CENTER_Y - y)


def angle_to_goal(x: FloatArray, y: FloatArray) -> FloatArray:
    """Angle in radians subtended by the two posts, via atan2(|cross|, dot) of the post vectors.

    Both post vectors share the x component `dx`, so the cross product reduces to
    `dx * (POST_HIGH_Y - POST_LOW_Y)`.
    """
    dx = GOAL_X - x
    cross = np.abs(dx * (POST_HIGH_Y - POST_LOW_Y))
    dot = dx**2 + (POST_LOW_Y - y) * (POST_HIGH_Y - y)
    return np.arctan2(cross, dot)
