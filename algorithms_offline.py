import csv
import sys
import numpy as np

import utils


def GMM(X, k, init, dist):
    S = []
    div = []
    dist_array = np.full(len(X), sys.float_info.max)
    if len(init) == 0:
        S.append(0)
        div.append(sys.float_info.max)
        for i in range(len(X)):
            dist_array[i] = dist(X[0], X[i])
    else:
        for i in range(len(init)):
            S.append(init[i])
            div.append(sys.float_info.max)
        for i in range(len(X)):
            for j in S:
                dist_array[i] = min(dist_array[i], dist(X[i], X[j]))
    while len(S) < k:
        max_idx = np.argmax(dist_array)
        max_dist = np.max(dist_array)
        S.append(max_idx)
        div.append(max_dist)
        for i in range(len(X)):
            dist_array[i] = min(dist_array[i], dist(X[i], X[max_idx]))
    return S, div


def GMM_color(X, color, k, init, dist):
    S = []
    div = []
    dist_array = np.full(len(X), sys.float_info.max)
    if len(init) == 0:
        first = -1
        for i in range(len(X)):
            if X[i].color == color:
                first = i
                break
        S.append(first)
        div.append(sys.float_info.max)
        for i in range(len(X)):
            if X[i].color == color:
                dist_array[i] = dist(X[first], X[i])
            else:
                dist_array[i] = 0.0
    else:
        for i in range(len(init)):
            S.append(init[i])
            div.append(sys.float_info.max)
        for i in range(len(X)):
            for j in S:
                if X[i].color == color:
                    dist_array[i] = min(dist_array[i], dist(X[i], X[j]))
                else:
                    dist_array[i] = 0.0

    while len(S) < k:
        max_idx = np.argmax(dist_array)
        max_dist = np.max(dist_array)
        S.append(max_idx)
        div.append(max_dist)
        for i in range(len(X)):
            if X[i].color == color:
                dist_array[i] = min(dist_array[i], dist(X[i], X[max_idx]))
    return S, div


def FairSwap(X, k0, k1, dist):
    S, div = GMM(X, k=k0 + k1, init=[], dist=dist)
    X0 = []
    X1 = []
    for i in S:
        if X[i].color == 0:
            X0.append(i)
        else:
            X1.append(i)
    if len(X0) < k0:
        S0, div0 = GMM_color(X, color=0, k=k0, init=X0, dist=dist)
        S1 = X1.copy()
        min_idx = -1
        min_dist = sys.float_info.max
        while len(S1) > k1:
            for i in S1:
                min_dist_i = sys.float_info.max
                for j in S0:
                    min_dist_i = min(min_dist_i, dist(X[i], X[j]))
                if min_dist_i < min_dist:
                    min_idx = i
                    min_dist = min_dist_i
            S1.remove(min_idx)
            min_idx = -1
            min_dist = sys.float_info.max
        S0.extend(S1)
        return S0
    elif len(X1) < k1:
        S1, div1 = GMM_color(X, color=1, k=k1, init=X1, dist=dist)
        S0 = X0.copy()
        min_idx = -1
        min_dist = sys.float_info.max
        while len(S0) > k0:
            for i in S0:
                min_dist_i = sys.float_info.max
                for j in S1:
                    min_dist_i = min(min_dist_i, dist(X[i], X[j]))
                if min_dist_i < min_dist:
                    min_idx = i
                    min_dist = min_dist_i
            S0.remove(min_idx)
            min_idx = -1
            min_dist = sys.float_info.max
        S0.extend(S1)
        return S0
    else:
        return S


if __name__ == "__main__":
    elements = []
    with open("datasets/blobs_n100_m2.csv") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = [float(row[2]), float(row[3])]
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    solf = FairSwap(X=elements, k0=5, k1=5, dist=utils.euclidean_dist)
    print(solf)
    solution = []
    for i in solf:
        solution.append(elements[i])
    print(utils.diversity(solution, utils.euclidean_dist))

    sol, div = GMM(X=elements, k=10, init=[], dist=utils.euclidean_dist)
    print(sol)
    print(div[1:])

    sol0, div0 = GMM_color(X=elements, color=0, k=10, init=[], dist=utils.euclidean_dist)
    print(sol0)
    print(div0[1:])

    sol1, div1 = GMM_color(X=elements, color=1, k=10, init=[], dist=utils.euclidean_dist)
    print(sol1)
    print(div1[1:])

