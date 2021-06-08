import csv
import itertools
import math
import sys
import numpy as np
import time
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


def FairSwap(X, k, dist):
    t0 = time.perf_counter()
    S, div = GMM(X, k=k[0] + k[1], init=[], dist=dist)
    S_group0 = []
    S_group1 = []
    for i in S:
        if X[i].color == 0:
            S_group0.append(i)
        else:
            S_group1.append(i)
    if len(S_group0) < k[0]:
        S0, div0 = GMM_color(X, color=0, k=k[0], init=S_group0, dist=dist)
        S1 = S_group1.copy()
        min_idx = -1
        min_dist = sys.float_info.max
        while len(S1) > k[1]:
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
        t1 = time.perf_counter()
        return S0, (t1 - t0)
    elif len(S_group1) < k[1]:
        S1, div1 = GMM_color(X, color=1, k=k[1], init=S_group1, dist=dist)
        S0 = S_group0.copy()
        min_idx = -1
        min_dist = sys.float_info.max
        while len(S0) > k[0]:
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
        t1 = time.perf_counter()
        return S0, (t1 - t0)
    else:
        t1 = time.perf_counter()
        return S, (t1 - t0)


def FairGMM(X, num_colors, k, dist):
    sum_k = sum(k)
    num_enum = 1
    for c in range(num_colors):
        num_enum *= math.comb(sum_k, k[c])
    print(num_enum)
    if num_enum > 1e6:
        return list(), 0, 0

    t0 = time.perf_counter()
    S = []
    for c in range(num_colors):
        Sc, divc = GMM_color(X, color=c, k=sum_k, init=[], dist=dist)
        S.append(Sc)
    f_seqs = []
    for c in range(num_colors):
        f_seqs.append(list(itertools.combinations(S[c], k[c])))
    f_sols = f_seqs[0].copy()
    for c in range(num_colors - 1):
        f_sols = list(itertools.product(f_sols, f_seqs[c + 1]))
        for i in range(len(f_sols)):
            f_sols[i] = list(np.concatenate(f_sols[i]).flat)
    max_div = 0
    max_sol = []
    for f_sol in f_sols:
        div_f_sol = diversity(X, idxs=f_sol, dist=dist)
        if div_f_sol > max_div:
            print(div_f_sol)
            max_sol = f_sol
            max_div = div_f_sol
    t1 = time.perf_counter()
    return max_sol, max_div, (t1 - t0)


def diversity(X, idxs, dist):
    div_val = sys.float_info.max
    for id1 in idxs:
        for id2 in idxs:
            if id1 != id2:
                div_val = min(div_val, dist(X[id1], X[id2]))
    return div_val


if __name__ == "__main__":
    elements = []
    with open("datasets/blobs_n100_m2.csv") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = [float(row[2]), float(row[3])]
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    solf, elapsed_time = FairSwap(X=elements, k=[2, 8], dist=utils.euclidean_dist)
    print(solf, elapsed_time)
    solution = []
    for idx in solf:
        solution.append(elements[idx])
    print(utils.diversity(solution, utils.euclidean_dist))

    sol, div_sol = GMM(X=elements, k=10, init=[], dist=utils.euclidean_dist)
    print(sol, div_sol[-1])

    sol0, div_sol0 = GMM_color(X=elements, color=0, k=10, init=[], dist=utils.euclidean_dist)
    print(sol0, div_sol0[-1])

    sol1, div_sol1 = GMM_color(X=elements, color=1, k=10, init=[], dist=utils.euclidean_dist)
    print(sol1, div_sol1[-1])

    # sol2, div2, elapsed_time2 = FairGMM(X=elements, num_colors=2, k=[5, 5], dist=utils.euclidean_dist)
    # print(sol2, div2, elapsed_time2)
    # solf2 = [0, 52, 51, 82, 45, 6, 41, 13, 23, 96]
    # solution.clear()
    # for i in solf2:
    #     solution.append(elements[i])
    # print(utils.diversity(solution, utils.euclidean_dist))
