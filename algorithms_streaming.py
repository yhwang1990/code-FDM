import csv
import math
import sys
import time
import numpy as np
import networkx as nx
import utils


class Instance:
    def __init__(self, k, mu, m):
        self.k = k
        self.mu = mu
        self.div = sys.float_info.max
        self.idxs = set()
        if m > 1:
            self.group_idxs = []
            for c in range(m):
                self.group_idxs.append(set())


def StreamDivMax(X, k, dist, eps, dmax, dmin):
    t0 = time.perf_counter()
    zmax = math.floor(math.log2(dmin) / math.log2(1 - eps))
    zmin = math.ceil(math.log2(dmax) / math.log2(1 - eps))
    print(zmin, zmax)
    Ins = []
    for z in range(zmin, zmax + 1):
        ins = Instance(k, mu=math.pow(1 - eps, z), m=1)
        Ins.append(ins)
    for x in X:
        for ins in Ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for y_idx in ins.idxs:
                    div_x = min(div_x, dist(x, X[y_idx]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.div = min(ins.div, div_x)
    max_inst = None
    max_div = 0
    for ins in Ins:
        # print(ins.mu, ins.div, ins.idxs)
        if len(ins.idxs) == k and ins.div > max_div:
            max_inst = ins
            max_div = ins.div
    t1 = time.perf_counter()
    return max_inst.idxs, max_inst.div, (t1 - t0)


def StreamFairDivMax1(X, k, dist, eps, dmax, dmin):
    if len(k) != 2:
        print("The length of k must be 2")
        return list(), 0, 0
    t0 = time.perf_counter()
    # Initialization
    zmax = math.floor(math.log2(dmin) / math.log2(1 - eps))
    zmin = math.ceil(math.log2(dmax) / math.log2(1 - eps))
    print(zmin, zmax)
    all_ins = []
    group_ins = [list(), list()]
    for z in range(zmin, zmax + 1):
        ins = Instance(k=k[0] + k[1], mu=math.pow(1 - eps, z), m=2)
        all_ins.append(ins)
        for c in range(0, 2):
            gins = Instance(k=k[c], mu=math.pow(1 - eps, z), m=1)
            group_ins[c].append(gins)
    # Stream processing
    for x in X:
        for ins in all_ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
                ins.group_idxs[x.color].add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in ins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.group_idxs[x.color].add(x.idx)
                    ins.div = min(ins.div, div_x)
        for gins in group_ins[x.color]:
            if len(gins.idxs) == 0:
                gins.idxs.add(x.idx)
            elif len(gins.idxs) < gins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in gins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < gins.mu:
                        flag_x = False
                        break
                if flag_x:
                    gins.idxs.add(x.idx)
                    gins.div = min(gins.div, div_x)
    t1 = time.perf_counter()
    # Post-processing (1): Find the lower index
    lower = 0
    upper = len(all_ins) - 1
    while lower < upper - 1:
        mid = (lower + upper) // 2
        if len(all_ins[mid].idxs) == k[0] + k[1] and len(group_ins[0][mid].idxs) == k[0] and len(
                group_ins[1][mid].idxs) == k[1]:
            upper = mid
        else:
            lower = mid
    # print(upper)
    # Post-processing (2): Balance each instance so that it is a fair solution.
    sol_div = 0
    sol_idx = -1
    for ins_id in range(upper + 1):
        if len(all_ins[ins_id].group_idxs[0].union(group_ins[0][ins_id].idxs)) < k[0] or len(
                all_ins[ins_id].group_idxs[1].union(group_ins[1][ins_id].idxs)) < k[1]:
            continue
        elif len(all_ins[ins_id].idxs) < k[0] + k[1]:
            while len(all_ins[ins_id].group_idxs[0]) < k[0]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[0][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[0]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].idxs:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[0].add(max_idx)
            while len(all_ins[ins_id].group_idxs[1]) < k[1]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[1][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[1]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].idxs:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[1].add(max_idx)
            while len(all_ins[ins_id].group_idxs[0]) > k[0]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[0]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].idxs:
                        if idx1 != idx2:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[0].remove(min_idx)
            while len(all_ins[ins_id].group_idxs[1]) > k[1]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[1]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].idxs:
                        if idx1 != idx2:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[1].remove(min_idx)
            ins_div = diversity(X, all_ins[ins_id].idxs, dist)
            if ins_div > sol_div:
                sol_idx = ins_id
                sol_div = ins_div
        else:
            while len(all_ins[ins_id].group_idxs[0]) < k[0]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[0][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[0]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].group_idxs[0]:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[0].add(max_idx)
            while len(all_ins[ins_id].group_idxs[1]) < k[1]:
                max_div = 0.0
                max_idx = -1
                for idx1 in group_ins[1][ins_id].idxs:
                    if idx1 not in all_ins[ins_id].group_idxs[1]:
                        div1 = sys.float_info.max
                        for idx2 in all_ins[ins_id].group_idxs[1]:
                            div1 = min(div1, dist(X[idx1], X[idx2]))
                        if div1 > max_div:
                            max_div = div1
                            max_idx = idx1
                all_ins[ins_id].idxs.add(max_idx)
                all_ins[ins_id].group_idxs[1].add(max_idx)
            while len(all_ins[ins_id].group_idxs[0]) > k[0]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[0]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].group_idxs[1]:
                        div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[0].remove(min_idx)
            while len(all_ins[ins_id].group_idxs[1]) > k[1]:
                min_div = sys.float_info.max
                min_idx = -1
                for idx1 in all_ins[ins_id].group_idxs[1]:
                    div1 = sys.float_info.max
                    for idx2 in all_ins[ins_id].group_idxs[0]:
                        div1 = min(div1, dist(X[idx1], X[idx2]))
                    if div1 < min_div:
                        min_div = div1
                        min_idx = idx1
                all_ins[ins_id].idxs.remove(min_idx)
                all_ins[ins_id].group_idxs[1].remove(min_idx)
            ins_div = diversity(X, all_ins[ins_id].idxs, dist)
            if ins_div > sol_div:
                sol_idx = ins_id
                sol_div = ins_div
    t2 = time.perf_counter()
    return all_ins[sol_idx].idxs, sol_div, (t1 - t0), (t2 - t1)


def StreamFairDivMax2(X, k, m, dist, eps, dmax, dmin):
    t0 = time.perf_counter()
    # Initialization
    sum_k = sum(k)
    zmax = math.floor(math.log2(dmin) / math.log2(1 - eps))
    zmin = math.ceil(math.log2(dmax) / math.log2(1 - eps))
    print(zmin, zmax)
    all_ins = []
    group_ins = []
    for c in range(m):
        group_ins.append(list())
    for z in range(zmin, zmax + 1):
        ins = Instance(k=sum_k, mu=math.pow(1 - eps, z), m=m)
        all_ins.append(ins)
        for c in range(m):
            gins = Instance(k=sum_k, mu=math.pow(1 - eps, z), m=1)
            group_ins[c].append(gins)
    # Stream processing
    for x in X:
        for ins in all_ins:
            if len(ins.idxs) == 0:
                ins.idxs.add(x.idx)
                ins.group_idxs[x.color].add(x.idx)
            elif len(ins.idxs) < ins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in ins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < ins.mu:
                        flag_x = False
                        break
                if flag_x:
                    ins.idxs.add(x.idx)
                    ins.group_idxs[x.color].add(x.idx)
                    ins.div = min(ins.div, div_x)
        for gins in group_ins[x.color]:
            if len(gins.idxs) == 0:
                gins.idxs.add(x.idx)
            elif len(gins.idxs) < gins.k:
                div_x = sys.float_info.max
                flag_x = True
                for idx_y in gins.idxs:
                    div_x = min(div_x, dist(x, X[idx_y]))
                    if div_x < gins.mu:
                        flag_x = False
                        break
                if flag_x:
                    gins.idxs.add(x.idx)
                    gins.div = min(gins.div, div_x)
    t1 = time.perf_counter()
    # for ins_id in range(len(all_ins)):
    #     print(ins_id, all_ins[ins_id].mu)
    #     print(all_ins[ins_id].idxs)
    #     for c in range(m):
    #         print(group_ins[c][ins_id].idxs)
    # post-processing
    sol = None
    sol_div = 0.0
    for ins_id in range(len(all_ins)):
        hasValidSol = True
        for c in range(m):
            if len(group_ins[c][ins_id].idxs) < k[c]:
                hasValidSol = False
                break
        if not hasValidSol:
            # print("No valid solution")
            continue
        S_all = set()
        S_all.update(all_ins[ins_id].idxs)
        for c in range(m):
            S_all.update(group_ins[c][ins_id].idxs)
        # print(S_all)
        G1 = nx.Graph()
        for idx1 in S_all:
            G1.add_node(idx1)
            for idx2 in S_all:
                if idx1 < idx2 and dist(X[idx1], X[idx2]) < all_ins[ins_id].mu / (m + 1):
                    G1.add_edge(idx1, idx2)
        P = []
        for p in nx.connected_components(G1):
            P.append(set(p))
        # print(P)
        dict_par = dict()
        for j in range(len(P)):
            for s_idx in P[j]:
                dict_par[s_idx] = j
        # print(dict_par)
        S_prime = set()
        num_elem_col = np.zeros(m)
        for c in range(m):
            if len(all_ins[ins_id].group_idxs[c]) <= k[c]:
                S_prime.update(all_ins[ins_id].group_idxs[c])
                num_elem_col[c] = len(all_ins[ins_id].group_idxs[c])
            else:
                for s_idx in all_ins[ins_id].group_idxs[c]:
                    S_prime.add(s_idx)
                    num_elem_col[c] += 1
                    if num_elem_col[c] == k[c]:
                        break
        # print(S_prime, num_elem_col)
        if len(S_prime) < sum_k:
            X1 = set()
            X2 = set()
            P_prime = set()
            for s_idx in S_prime:
                P_prime.add(dict_par[s_idx])
            for s_idx in S_all:
                s_col = X[s_idx].color
                s_par = dict_par[s_idx]
                if s_idx not in S_prime and num_elem_col[s_col] < k[s_col]:
                    X1.add(s_idx)
                if s_idx not in S_prime and s_par not in P_prime:
                    X2.add(s_idx)
            cand = X1.intersection(X2)
            # print(cand, X1, X2)
            while len(cand) > 0:
                max_idx = -1
                max_div = 0.0
                for s_idx1 in cand:
                    s_div1 = sys.float_info.max
                    for s_idx2 in S_prime:
                        s_div1 = min(s_div1, dist(X[s_idx1], X[s_idx2]))
                    if s_div1 > max_div:
                        max_idx = s_idx1
                        max_div = s_div1
                max_col = X[max_idx].color
                max_par = dict_par[max_idx]
                S_prime.add(max_idx)
                num_elem_col[max_col] += 1
                # print(max_idx, max_col, max_par, S_prime, num_elem_col)
                if num_elem_col[max_col] == k[max_col]:
                    for s_idx in group_ins[max_col][ins_id].idxs:
                        X1.discard(s_idx)
                for s_idx in P[max_par]:
                    X2.discard(s_idx)
                cand = X1.intersection(X2)
                # print(cand, X1, X2)
        if len(S_prime) == sum_k:
            div_s = diversity(X, S_prime, dist)
            # print(S_prime, div_s)
            if div_s > sol_div:
                sol = S_prime
                sol_div = div_s
    t2 = time.perf_counter()
    return sol, sol_div, (t1 - t0), (t2 - t1)


def diversity(X, idxs, dist):
    div_val = sys.float_info.max
    for idx1 in idxs:
        for idx2 in idxs:
            if idx1 != idx2:
                div_val = min(div_val, dist(X[idx1], X[idx2]))
    return div_val


if __name__ == "__main__":
    elements = []
    with open("datasets/blobs_n100_m2.csv") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = [float(row[2]), float(row[3])]
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)

    # solf, div_solf, elapsed_time = StreamDivMax(X=elements, k=5, dist=utils.euclidean_dist, eps=0.1, dmax=15.0,
    #                                             dmin=5.0)
    # print(solf, div_solf, elapsed_time)
    # solution = []
    # for i in solf:
    #     solution.append(elements[i])
    # print(utils.diversity(solution, utils.euclidean_dist))
    solf2, div_solf2, stream_time, post_time = StreamFairDivMax1(X=elements, k=[2, 3], dist=utils.euclidean_dist,
                                                                 eps=0.1, dmax=17.0, dmin=8.0)
    print(solf2, div_solf2, stream_time, post_time)
    # solution.clear()
    # for i in solf2:
    #     solution.append(elements[i])
    # print(utils.diversity(solution, utils.euclidean_dist))

    solf3, div_solf3, stream_time3, post_time3 = StreamFairDivMax2(X=elements, k=[2, 3], m=2, dist=utils.euclidean_dist,
                                                                   eps=0.1, dmax=17.0, dmin=8.0)
    print(solf3, div_solf3, stream_time3, post_time3)
    # solution = []
    # for i in solf3:
    #     solution.append(elements[i])
    # print(utils.diversity(solution, utils.euclidean_dist))
