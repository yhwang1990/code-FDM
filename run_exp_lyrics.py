import csv
import random

import numpy as np

import algorithms_offline as algo
import algorithms_streaming as algs
import utils

output = open("results_lyrics.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])

output.flush()

values_eps = [0.02, 0.04, 0.06, 0.08, 0.1]
values_k = range(5, 51, 5)

# read the Lyrics dataset grouped by genre (m=15)
elements = []
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = dict()
        for i in range(2, len(row)):
            if float(row[i]) >= 0.0001:
                features[i - 2] = float(row[i])
        elem = utils.ElementSparse(int(row[0]), int(row[1]), features)
        elements.append(elem)

range_d = {15: 1.3, 20: 1.15, 25: 1.07, 30: 1.03, 35: 0.97, 40: 0.93, 45: 0.9, 50: 0.87}
k_20 = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m = 15

# experiments on varying epsilon
for eps in values_eps:
    alg2 = np.zeros([4, 10])
    for run in range(5):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=k_20, m=m, dist=utils.cosine_dist_sparse, eps=eps, dmax=1.57, dmin=range_d[20])
        print(sol)
    writer.writerow(["Lyrics", "Genre", m, 20, "Alg2", eps, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

# experiments on varying k
for k in values_k:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    alg2 = np.zeros([4, 10])
    fair_flow = np.zeros([2, 10])
    for run in range(10):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.cosine_dist_sparse)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.cosine_dist_sparse, eps=0.05, dmax=1.57, dmin=range_d[k])
        print(sol)
    writer.writerow(["Lyrics", "Genre", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Lyrics", "Genre", m, k, "Alg2", 0.05, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

k_20_p = [5, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
alg2 = np.zeros([4, 10])
fair_flow = np.zeros([2, 10])
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=k_20_p, m=m, dist=utils.cosine_dist_sparse)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=k_20_p, m=m, dist=utils.cosine_dist_sparse, eps=0.05, dmax=1.57, dmin=range_d[20])
    print(sol)
writer.writerow(["Lyrics", "Genre_P", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Lyrics", "Genre_P", m, 20, "Alg2", 0.05, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.close()
