import sys
import numpy as np


class Element:
    def __init__(self, idx, color, features):
        self.idx = idx
        self.color = color
        self.features = features


def GMM(elements, k, sol, dist):
    dist_array = np.full(len(elements), sys.maxsize)
    if len(sol) == 0:
        sol.append(0)
        for i in range(len(elements)):
            dist_array[i] = dist(elements[0].features, elements[i].features)
    else:
        for i in range(len(elements)):
            for j in sol:
                dist_array[i] = min(dist_array[i], dist(elements[i].features, elements[j].features))
    while len(sol) < k:
        min_idx = np.argmin(dist_array)
        sol.append(min_idx)
        for i in range(len(elements)):
            dist_array[i] = min(dist_array[i], dist(elements[i].features, elements[min_idx].features))
