import csv
import itertools
import random

import numpy as np

import algorithms_offline as algo
import algorithms_streaming as algs
import utils

output = open("results_synthetic.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "n", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])

num_runs = 5

# experiments for varying m
elements = []
varying_m = {2: 3.6, 4: 3.5, 6: 3.3, 8: 3.3, 10: 3.2, 12: 3.2, 14: 3.2, 16: 3.1, 18: 3.1, 20: 3.0}
for m in range(2, 21, 2):
    elements.clear()
    with open("datasets/blobs_n100000_m" + str(m) + ".csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    if m == 2:
        alg1 = np.zeros([4, num_runs])
        alg2 = np.zeros([4, num_runs])
        fair_swap = np.zeros([2, num_runs])
        fair_flow = np.zeros([2, num_runs])
        for run in range(num_runs):
            random.Random(run).shuffle(elements)
            for new_idx in range(len(elements)):
                elements[new_idx].idx = new_idx
            sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=[10, 10], dist=utils.euclidean_dist)
            print(sol)
            sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[10, 10], m=2, dist=utils.euclidean_dist)
            print(sol)
            sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=[10, 10], dist=utils.euclidean_dist, eps=0.1, dmax=6.0, dmin=varying_m[2])
            print(sol)
            sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[10, 10], m=2, dist=utils.euclidean_dist, eps=0.1, dmax=6.0, dmin=varying_m[2])
            print(sol)
        writer.writerow(["Blobs", 100000, 2, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
        writer.writerow(["Blobs", 100000, 2, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
        writer.writerow(["Blobs", 100000, 2, 20, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
        writer.writerow(["Blobs", 100000, 2, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
        output.flush()
    else:
        alg2 = np.zeros([4, num_runs])
        fair_flow = np.zeros([2, num_runs])
        group_k = [0] * m
        remainder = 20 % m
        for c in range(m):
            if c < remainder:
                group_k[c] = 20 // m + 1
            else:
                group_k[c] = 20 // m
        for run in range(num_runs):
            random.Random(run).shuffle(elements)
            for new_idx in range(len(elements)):
                elements[new_idx].idx = new_idx
            sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
            print(sol)
            sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=6.0, dmin=varying_m[m])
            print(sol)
        writer.writerow(["Blobs", 100000, m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
        writer.writerow(["Blobs", 100000, m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
        output.flush()

num_elem = [1000, 10000, 100000, 1000000, 10000000]
varying_n_m2 = {1000: [2.7, 4.5], 10000: [3.2, 5.2], 100000: [3.7, 5.8], 1000000: [3.9, 6.0], 10000000: [4.2, 6.5]}
varying_n_m10 = {1000: [1.6, 4.5], 10000: [3.8, 5.2], 100000: [3.3, 5.8], 1000000: [3.7, 6.0], 10000000: [4.0, 6.5]}
for n in num_elem:
    elements.clear()
    with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in itertools.islice(csvreader, n):
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    num_runs = 0
    if n <= 100000:
        num_runs = 5
    else:
        num_runs = 2
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=[10, 10], dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[10, 10], m=2, dist=utils.euclidean_dist)
        print(sol)
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=[10, 10], dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m2[n][1], dmin=varying_n_m2[n][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[10, 10], m=2, dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m2[n][1], dmin=varying_n_m2[n][0])
        print(sol)
    writer.writerow(["Blobs", n, 2, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["Blobs", n, 2, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Blobs", n, 2, 20, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Blobs", n, 2, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()

    elements.clear()
    with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in itertools.islice(csvreader, n):
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    num_runs = 0
    if n <= 100000:
        num_runs = 5
    else:
        num_runs = 2
    alg2 = np.zeros([4, num_runs])
    fair_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[2] * 10, m=10, dist=utils.euclidean_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[2] * 10, m=10, dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m10[n][1], dmin=varying_n_m10[n][0])
        print(sol)
    writer.writerow(["Blobs", n, 10, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Blobs", n, 10, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
output.close()
