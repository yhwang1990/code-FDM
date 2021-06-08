import csv
import math
import sys
import time
import utils


class Instance:
    def __init__(self, k, mu, num_colors):
        self.k = k
        self.mu = mu
        self.div = sys.float_info.max
        self.idxs = set()
        if num_colors > 1:
            self.group_idxs = []
            for c in range(num_colors):
                self.group_idxs.append(set())


def StreamDivMax(X, k, dist, eps, dmax, dmin):
    t0 = time.perf_counter()
    zmax = math.floor(math.log2(dmin) / math.log2(1 - eps))
    zmin = math.ceil(math.log2(dmax) / math.log2(1 - eps))
    print(zmin, zmax)
    Ins = []
    for z in range(zmin, zmax + 1):
        ins = Instance(k, mu=math.pow(1 - eps, z), num_colors=1)
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
        print(ins.mu, ins.div, ins.idxs)
        if len(ins.idxs) == k and ins.div > max_div:
            max_inst = ins
            max_div = ins.div
    t1 = time.perf_counter()
    return max_inst.idxs, max_inst.div, (t1 - t0)


def StreamFairDivMax1(X, k, dist, eps, dmax, dmin):
    t0 = time.perf_counter()
    # Initialization
    zmax = math.floor(math.log2(dmin) / math.log2(1 - eps))
    zmin = math.ceil(math.log2(dmax) / math.log2(1 - eps))
    print(zmin, zmax)
    all_ins = []
    group_ins = [list(), list()]
    for z in range(zmin, zmax + 1):
        ins = Instance(k=k[0] + k[1], mu=math.pow(1 - eps, z), num_colors=2)
        all_ins.append(ins)
        for c in range(0, 2):
            gins = Instance(k=k[c], mu=math.pow(1 - eps, z), num_colors=1)
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
        if len(all_ins[mid].idxs) == k[0] + k[1] and len(group_ins[0][mid].idxs) == k[0] and len(group_ins[1][mid].idxs) == k[1]:
            upper = mid
        else:
            lower = mid
    print(upper)
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
    # solf, divf, elapsed_time = StreamDivMax(X=elements, k=10, dist=utils.euclidean_dist, eps=0.1, dmax=6.0, dmin=3.0)
    # print(solf, divf, elapsed_time)
    # solution = []
    # for i in solf:
    #     solution.append(elements[i])
    # print(utils.diversity(solution, utils.euclidean_dist))
    solf, divf, stream_time, post_time = StreamFairDivMax1(X=elements, k=[2, 8], dist=utils.euclidean_dist, eps=0.1, dmax=6.0,
                                                 dmin=3.0)
    print(solf, divf, stream_time, post_time)
    solution = []
    for i in solf:
        solution.append(elements[i])
    print(utils.diversity(solution, utils.euclidean_dist))
