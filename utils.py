import sys
import numpy as np
from scipy.spatial import distance


def euclidean_dist(point1, point2):
    if len(point1) != len(point2):
        raise AssertionError("dimension not match")
    return distance.euclidean(point1, point2)


def manhattan_dist(point1, point2):
    if len(point1) != len(point2):
        raise AssertionError("dimension not match")
    return distance.cityblock(point1, point2)


def cosine_dist(point1, point2):
    if len(point1) != len(point2):
        raise AssertionError("dimension not match")
    return np.arccos(1.0 - distance.cosine(point1, point2))


def diversity(points, dist):
    div = sys.float_info.max
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                div = min(div, dist(points[i], points[j]))
    return div
