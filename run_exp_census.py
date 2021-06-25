import csv
import random

import numpy as np

import algorithms_offline as algo
import algorithms_streaming as algs
import utils

output = open("results_census.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])

output.flush()

values_eps = [0.25, 0.2, 0.15, 0.1, 0.05]
values_k = range(5, 51, 5)

num_runs = 2

# read the Census dataset grouped by sex (m=2)
elements = []
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)

range_d_sex = {5: [40.0, 60.0], 10: [32.0, 50.0], 15: [27.5, 45.0], 20: [25.0, 42.5], 25: [24.0, 40.0],
               30: [24.0, 40.0], 35: [24.0, 40.0], 40: [22.0, 38.0], 45: [21.0, 37.5], 50: [20.0, 37.5]}
m = 2

# experiments on varying epsilon
for eps in values_eps:
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=[10, 10], dist=utils.manhattan_dist, eps=eps, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[10, 10], m=m, dist=utils.manhattan_dist, eps=eps, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
        print(sol)
    writer.writerow(["Census", "Sex", m, 20, "Alg1", eps, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Census", "Sex", m, 20, "Alg2", eps, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
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
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fair_gmm = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_sex[k][1], dmin=range_d_sex[k][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_sex[k][1], dmin=range_d_sex[k][0])
        print(sol)
    writer.writerow(["Census", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["Census", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Census", "Sex", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["Census", "Sex", m, k, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Census", "Sex", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

# read the Census dataset grouped by age (m=7)
elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)

range_d_age = {10: [13.0, 50.0], 15: [11.0, 45.0], 20: [10.0, 42.5], 25: [9.0, 40.0],  30: [8.0, 40.0],
               35: [8.0, 40.0], 40: [8.0, 38.0], 45: [7.0, 37.5], 50: [7.0, 37.5]}
m = 7

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
    alg2 = np.zeros([4, num_runs])
    fair_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_age[k][1], dmin=range_d_age[k][0])
        print(sol)
    writer.writerow(["Census", "Age", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Census", "Age", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

# read the Census dataset grouped by sex+age (m=14)
elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)

range_d_both = [10.0, 42.5]
m = 14

alg2 = np.zeros([4, num_runs])
fair_flow = np.zeros([2, num_runs])
k_20 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
for run in range(num_runs):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=k_20, m=m, dist=utils.manhattan_dist)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=k_20, m=m, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_both[1], dmin=range_d_both[0])
    print(sol)
writer.writerow(["Census", "Both", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Census", "Both", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.close()
