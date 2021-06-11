import csv
import random

import numpy as np

import algorithms_offline as algo
import algorithms_streaming as algs
import utils

output = open("results_adult.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])

output.flush()

values_eps = [0.25, 0.2, 0.15, 0.1, 0.05]
values_k = range(5, 51, 5)

# read the Adult dataset grouped by sex (m=2)
elements = []
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)

range_d_sex = {5: [5.0, 11.7], 10: [4.2, 8.8], 15: [3.7, 7.5], 20: [3.2, 6.5], 25: [3.0, 6.2],
               30: [2.7, 5.7], 35: [2.5, 5.2], 40: [2.4, 5.0], 45: [2.3, 4.9], 50: [2.2, 4.7]}
m = 2

# experiments on varying epsilon
# for eps in values_eps:
#     alg1 = np.zeros([4, 10])
#     alg2 = np.zeros([4, 10])
#     for run in range(10):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=[10, 10], dist=utils.euclidean_dist, eps=eps, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
#         print(sol)
#         sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[10, 10], m=m, dist=utils.euclidean_dist, eps=eps, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
#         print(sol)
#     writer.writerow(["Adult", "Sex", m, 20, "Alg1", eps, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
#     writer.writerow(["Adult", "Sex", m, 20, "Alg2", eps, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
#     output.flush()

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
    alg1 = np.zeros([4, 10])
    alg2 = np.zeros([4, 10])
    fair_swap = np.zeros([2, 10])
    fair_flow = np.zeros([2, 10])
    fair_gmm = np.zeros([2, 10])
    for run in range(10):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_sex[k][1], dmin=range_d_sex[k][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_sex[k][1], dmin=range_d_sex[k][0])
        print(sol)
    writer.writerow(["Adult", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["Adult", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Adult", "Sex", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["Adult", "Sex", m, k, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Adult", "Sex", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

# read the Adult dataset grouped by race (m=5)
elements.clear()
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)

range_d_race = {5: [3.5, 11.7], 10: [2.6, 8.8], 15: [2.1, 7.5], 20: [1.8, 6.5], 25: [1.7, 6.2],
                30: [1.5, 5.7], 35: [1.4, 5.2], 40: [1.3, 5.0], 45: [1.2, 4.9], 50: [1.1, 4.7]}
m = 5

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
    fair_gmm = np.zeros([2, 10])
    for run in range(10):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_race[k][1], dmin=range_d_race[k][0])
        print(sol)
    writer.writerow(["Adult", "Race", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Adult", "Race", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["Adult", "Race", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

# read the Adult dataset grouped by race (m=5)
elements.clear()
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)

range_d_both = [1.3, 6.5]
m = 10

alg2 = np.zeros([4, 10])
fair_flow = np.zeros([2, 10])
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[2] * m, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[2] * m, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_both[1], dmin=range_d_both[0])
    print(sol)
writer.writerow(["Adult", "Both", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Adult", "Both", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.flush()

m = 2
group_k = [13, 7]
alg1 = np.zeros([4, 10])
alg2 = np.zeros([4, 10])
fair_swap = np.zeros([2, 10])
fair_flow = np.zeros([2, 10])
fair_gmm = np.zeros([2, 10])
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k, dist=utils.euclidean_dist)
    print(sol)
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_sex[20][1], dmin=range_d_sex[20][0])
    print(sol)
writer.writerow(["Adult", "Sex_P", m, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
writer.writerow(["Adult", "Sex_P", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Adult", "Sex_P", m, 20, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
writer.writerow(["Adult", "Sex_P", m, 20, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
writer.writerow(["Adult", "Sex_P", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.flush()

m = 5
group_k = [15, 1, 1, 2, 1]
alg2 = np.zeros([4, 10])
fair_flow = np.zeros([2, 10])
fair_gmm = np.zeros([2, 10])
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=range_d_race[20][1], dmin=range_d_race[20][0])
    print(sol)
writer.writerow(["Adult", "Race_P", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Adult", "Race_P", m, 20, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
writer.writerow(["Adult", "Race_P", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.close()
