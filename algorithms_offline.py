import csv
import sys
import numpy as np
import utils


class Element:
    def __init__(self, idx, color, features):
        self.idx = idx
        self.color = color
        self.features = features


def GMM(X, k, S, div, dist):
    dist_array = np.full(len(X), sys.float_info.max)
    if len(S) == 0:
        S.append(0)
        div.append(sys.float_info.max)
        for i in range(len(X)):
            dist_array[i] = dist(X[0].features, X[i].features)
    else:
        for i in range(len(S)):
            div.append(sys.float_info.max)
        for i in range(len(X)):
            for j in S:
                dist_array[i] = min(dist_array[i], dist(X[i].features, X[j].features))
    while len(S) < k:
        max_idx = np.argmax(dist_array)
        max_dist = np.max(dist_array)
        S.append(max_idx)
        div.append(max_dist)
        for i in range(len(X)):
            dist_array[i] = min(dist_array[i], dist(X[i].features, X[max_idx].features))


if __name__ == "__main__":
    elements = []
    with open("datasets/blobs_n100_m2.csv") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = [float(row[2]), float(row[3])]
            elem = Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    sol = []
    diversity = []
    GMM(X=elements, k=10, S=sol, div=diversity, dist=utils.cosine_dist)
    print(sol)
    print(diversity[1:])

    solution = []
    for idx in sol:
        solution.append(elements[idx].features)
    print(utils.diversity(solution, utils.cosine_dist))
