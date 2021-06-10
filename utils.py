import sys
import numpy as np
from scipy.spatial import distance
from typing import Any, Callable, Dict, List


class Element:
    def __init__(self, idx: int, color: int, features: List[float]):
        self.idx = idx
        self.color = color
        self.features = features


class ElementSparse:
    def __init__(self, idx: int, color: int, features: Dict):
        self.idx = idx
        self.color = color
        self.features = features


def euclidean_dist(elem1: Element, elem2: Element) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.euclidean(elem1.features, elem2.features)


def euclidean_dist_sparse(elem1: ElementSparse, elem2: ElementSparse) -> float:
    sum_len = 0.0
    for j in elem1.features.keys():
        if j in elem2.features.keys():
            sum_len += (elem1.features[j] - elem2.features[j]) * (elem1.features[j] - elem2.features[j])
        else:
            sum_len += elem1.features[j] * elem1.features[j]
    for j in elem2.features.keys():
        if j not in elem2.features.keys():
            sum_len += elem2.features[j] * elem2.features[j]
    return np.sqrt(sum_len)


def manhattan_dist(elem1: Element, elem2: Element) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return distance.cityblock(elem1.features, elem2.features)


def manhattan_dist_sparse(elem1: ElementSparse, elem2: ElementSparse) -> float:
    sum_len = 0.0
    for j in elem1.features.keys():
        if j in elem2.features.keys():
            sum_len += np.abs(elem1.features[j] - elem2.features[j])
        else:
            sum_len += elem1.features[j]
    for j in elem2.features.keys():
        if j not in elem2.features.keys():
            sum_len += elem2.features[j]
    return sum_len


def cosine_dist(elem1: Element, elem2: Element) -> float:
    if len(elem1.features) != len(elem2.features):
        raise AssertionError("dimension not match")
    return np.arccos(1.0 - distance.cosine(elem1.features, elem2.features))


def cosine_dist_sparse(elem1: ElementSparse, elem2: ElementSparse) -> float:
    inner_prod = 0
    length1 = 0
    length2 = 0
    for j in elem1.features.keys():
        if j in elem2.features.keys():
            inner_prod += elem1.features[j] * elem2.features[j]
        length1 += elem1.features[j] * elem1.features[j]
    for j in elem2.features.keys():
        length2 += elem2.features[j] * elem2.features[j]
    if length1 == 0 or length2 == 0:
        return 0.0
    cosine = inner_prod / (np.sqrt(length1) * np.sqrt(length2))
    if cosine > 1.0:
        return 0.0
    return np.arccos(cosine)


def diversity(elements: List[Element], dist: Callable[[Any, Any], float]) -> float:
    div = sys.float_info.max
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i != j:
                div = min(div, dist(elements[i], elements[j]))
    return div


# if __name__ == "__main__":
#     a = ElementSparse(0, 0, {0: 1.0, 1: 1.0})
#     b = ElementSparse(0, 0, {0: 1.0, 2: 1.0})
#     c = ElementSparse(0, 0, {1: 1.0, 3: 1.0})
#     print(euclidean_dist_sparse(a, b))
#     print(manhattan_dist_sparse(a, b))
#     print(cosine_dist_sparse(a, b))
#     print(euclidean_dist_sparse(c, b))
#     print(manhattan_dist_sparse(c, b))
#     print(cosine_dist_sparse(c, b))
