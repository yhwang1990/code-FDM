import sys
import numpy as np
from scipy.spatial import distance


class Element:
    def __init__(self, idx, color, features):
        self.idx = idx
        self.color = color
        self.features = features


def euclidean_dist(elem1, elem2):
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.euclidean(elem1.features, elem2.features)


def manhattan_dist(elem1, elem2):
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.cityblock(elem1.features, elem2.features)


def cosine_dist(elem1, elem2):
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return np.arccos(1.0 - distance.cosine(elem1.features, elem2.features))


def diversity(elements, dist):
    div = sys.float_info.max
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i != j:
                div = min(div, dist(elements[i], elements[j]))
    return div
