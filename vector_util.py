"""
Contains basic definitions for data and function types.
"""
import sys
from typing import Tuple, Callable, Collection

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise Exception("Must be using Python 3.6 or higher")

Vector = Tuple[float, ...]
NamedVector = Tuple[str, Vector]

Distance_Function = Callable[[Vector, Vector], float]
Centroid_Function = Callable[[Collection[Vector]], Vector]


def pq_distance(x, y: Vector, q: int) -> float:
    """
    Calculates the distance of two vectors based on the formula:
    x_i, y_i - the i_th component of the vectors x and y
    q-th root of (sum of all ( |x_i - y_i| ** q))

    :param x: a vector
    :param y: another vector
    :param q: The exponent / degree of root of the formula
    :return: The distance (positive scalar)
    """
    powered_dist_sum = 0
    dimensions = len(x)
    for i in range(0, dimensions):
        powered_dist_sum += abs(x[i] - y[i]) ** q
    return powered_dist_sum ** (1.0 / q)


def euclidean_distance(x, y: Vector) -> float:
    """
    Returns the euclidean distance between to points.
    :param x: a point
    :param y: another point
    :return: The distance (positive scalar)
    """
    return pq_distance(x, y, 2)


def block_distance(x, y: Vector) -> float:
    """
    Returns the block distance between to points.
    :param x: a point
    :param y: another point
    :return: The distance (positive scalar)
    """
    return pq_distance(x, y, 1)


def simple_centroid(points: Collection[Vector]) -> Vector:
    """
    Calculates and returns a center of points based on the linear average value of scalars
    for each given point per dimension.

    :param points: a set of vectors for which a centroid is searched
    :return: the centroid of the points
    """
    num_points = len(points)
    num_dimensions = len(next(iter(points)))
    sums = []
    for _ in range(0, num_dimensions):
        sums.append(0)
    for point in points:
        for i in range(0, len(point)):
            sums[i] += point[i]
    for i in range(0, num_dimensions):
        sums[i] = float(sums[i]) / float(num_points)
    return tuple(sums.copy())
