import csv
import random
import numpy as np
import algorithms_offline as algo
import algorithms_streaming as algs
import utils
import itertools

output = open("results_adult.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

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
num_runs = 10
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
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements, k=group_k, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_sex[k][1],
                                                                                             dmin=range_d_sex[k][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_sex[k][1],
                                                                                             dmin=range_d_sex[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.1,
                                                                                              dmax=range_d_sex[k][1],
                                                                                            dmin=range_d_sex[k][0],
                                                                                             metric_name='euclidean')
        print(sol)
    writer.writerow(["Adult", "Sex", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])

    writer.writerow(["Adult", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["Adult", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Adult", "Sex", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["Adult", "Sex", m, k, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
                     np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["Adult", "Sex", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                     np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
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
    alg2 = np.zeros([4, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fair_gmm = np.zeros([2, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])

    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_race[k][1],
                                                                                             dmin=range_d_race[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_race[k][1],
                                                                                             dmin=range_d_race[k][0],
                                                                                             metric_name='euclidean')
        print(sol)
    writer.writerow(["Adult", "Race", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow(["Adult", "Race", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Adult", "Race", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["Adult", "Race", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                     np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
# read the Adult dataset grouped by sex+race (m=10)
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
alg2 = np.zeros([4, num_runs])
fair_flow = np.zeros([2, num_runs])
fair_greedy_flow = np.zeros([2, num_runs])

for run in range(num_runs):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[2] * m, m=m, dist=utils.euclidean_dist)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[2] * m, m=m,
                                                                                         dist=utils.euclidean_dist,
                                                                                         eps=0.1, dmax=range_d_both[1],
                                                                                         dmin=range_d_both[0])
    print(sol)
    sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=[2] * m, m=m,
                                                                                  dist=utils.euclidean_dist,
                                                                                  eps=0.1,
                                                                                  dmax=range_d_both[1],
                                                                                  dmin=range_d_both[0],
                                                                                  metric_name='euclidean')
    print(sol)
writer.writerow(["Adult", "Both", m, 20, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
writer.writerow(["Adult", "Both", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Adult", "Both", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                 np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.flush()
output.close()

output = open("results_celeba.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

values_eps = [0.25, 0.2, 0.15, 0.1, 0.05]
values_k = range(5, 51, 5)
num_runs = 10
elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
range_d_sex = {5: [11.0, 18.0], 10: [10.0, 16.0], 15: [9.0, 15.0], 20: [9.0, 15.0], 25: [9.0, 14.0],
               30: [9.0, 14.0], 35: [8.0, 13.0], 40: [8.0, 13.0], 45: [8.0, 13.0], 50: [8.0, 13.0]}
m = 2
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
    fair_greedy_flow = np.zeros([2, num_runs])
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
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_sex[k][1],
                                                                                             dmin=range_d_sex[k][0],
                                                                                             metric_name='cityblock')
        print(sol)
    writer.writerow(["CelebA", "Sex", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow(["CelebA", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["CelebA", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["CelebA", "Sex", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["CelebA", "Sex", m, k, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
                     np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["CelebA", "Sex", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                     np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)

range_d_age = {5: [11.0, 18.0], 10: [10.0, 16.0], 15: [9.0, 15.0], 20: [9.0, 15.0], 25: [9.0, 14.0],
               30: [9.0, 14.0], 35: [8.0, 13.0], 40: [8.0, 13.0], 45: [8.0, 13.0], 50: [8.0, 13.0]}
m = 2
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
    fair_greedy_flow = np.zeros([2, num_runs])
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
        sol, alg1[0][run], alg1[1][run], alg1[2][run], alg1[3][run] = algs.StreamFairDivMax1(X=elements, k=group_k,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_age[k][1],
                                                                                             dmin=range_d_age[k][0])
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_age[k][1],
                                                                                             dmin=range_d_age[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_age[k][1],
                                                                                             dmin=range_d_age[k][0],
                                                                                             metric_name='cityblock')
        print(sol)
    writer.writerow(["CelebA", "Age", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow(["CelebA", "Age", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
    writer.writerow(["CelebA", "Age", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["CelebA", "Age", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["CelebA", "Age", m, k, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]),
                     np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
    writer.writerow(["CelebA", "Age", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                     np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)

range_d_both = {5: [11.0, 18.0], 10: [9.0, 16.0], 15: [9.0, 15.0], 20: [9.0, 15.0], 25: [8.0, 14.0],
                30: [8.0, 14.0], 35: [8.0, 13.0], 40: [8.0, 13.0], 45: [8.0, 13.0], 50: [7.0, 13.0]}
m = 4
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
    fair_gmm = np.zeros([2, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])

    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_gmm[0][run], fair_gmm[1][run] = algo.FairGMM(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_both[k][1],
                                                                                             dmin=range_d_both[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.manhattan_dist,
                                                                                             eps=0.1,
                                                                                             dmax=range_d_both[k][1],
                                                                                             dmin=range_d_both[k][0],
                                                                                             metric_name='cityblock')
        print(sol)
    writer.writerow(["CelebA", "Both", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow(["CelebA", "Both", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["CelebA", "Both", m, k, "FairGMM", "-", np.average(fair_gmm[0]), "-", "-", "-", np.average(fair_gmm[1])])
    writer.writerow(["CelebA", "Both", m, k, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]),
                     np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
output.close()

output = open("results_census.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

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
    fair_greedy_flow = np.zeros([2, num_runs])
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
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                  dist=utils.manhattan_dist,
                                                                                  eps=0.1,
                                                                                      dmax=range_d_sex[k][1],
                                                                                      dmin=range_d_sex[k][0],
                                                                                 metric_name='cityblock')
        print(sol)
    writer.writerow(["Census", "Sex", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
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
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=group_k, m=m, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_age[k][1], dmin=range_d_age[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                  dist=utils.manhattan_dist,
                                                                                  eps=0.1,
                                                                                      dmax=range_d_age[k][1],
                                                                                      dmin=range_d_age[k][0],
                                                                                 metric_name='cityblock')
        print(sol)
    writer.writerow(["Census", "Age", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
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
fair_greedy_flow = np.zeros([2, num_runs])
k_20 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
for run in range(num_runs):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=k_20, m=m, dist=utils.manhattan_dist)
    print(sol)
    sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=k_20, m=m, dist=utils.manhattan_dist, eps=0.1, dmax=range_d_both[1], dmin=range_d_both[0])
    print(sol)
    sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=k_20, m=m,
                                                                                  dist=utils.manhattan_dist,
                                                                                  eps=0.1,
                                                                                  dmax=range_d_both[1],
                                                                                  dmin=range_d_both[0],
                                                                                  metric_name='cityblock')
    print(sol)
writer.writerow(["Census", "Both", m, 20, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
writer.writerow(["Census", "Both", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
writer.writerow(["Census", "Both", m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
output.flush()
output.close()


output = open("results_lyrics.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

values_k = range(5, 51, 5)
num_runs = 5
# read the Lyrics dataset grouped by genre (m=15)
elements = []
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
range_d = {15: 1.3, 20: 1.15, 25: 1.07, 30: 1.03, 35: 0.97, 40: 0.93, 45: 0.9, 50: 0.87}
k_20 = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m = 15
# FairGreedyFlow vary k
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
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m,
                                                                                             dist=utils.cosine_dist,
                                                                                             eps=0.05,
                                                                                             dmax=1.57, dmin=range_d[k],
                                                                                            metric_name='cosine')
        print(sol)
    writer.writerow(["Lyrics", "Genre", m, k, "FairGreedyFlow", 0.05, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    output.flush()

elements.clear()
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = dict()
        for i in range(2, len(row)):
            if float(row[i]) >= 0.0001:
                features[i - 2] = float(row[i])
        elem = utils.ElementSparse(int(row[0]), int(row[1]), features)
        elements.append(elem)
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
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
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
output.close()


output = open("results_gmm.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "k", "div"])
output.flush()

elements.clear()
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
sol, divs = algo.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
for k in range(5, 51, 5):
    writer.writerow(["Adult", k, divs[k - 1]])
output.flush()

elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
sol, divs = algo.GMM(X=elements, k=50, init=[], dist=utils.manhattan_dist)
for k in range(5, 51, 5):
    writer.writerow(["CelebA", k, divs[k - 1]])
output.flush()

elements.clear()
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = dict()
        for i in range(2, len(row)):
            if float(row[i]) >= 0.0001:
                features[i - 2] = float(row[i])
        elem = utils.ElementSparse(int(row[0]), int(row[1]), features)
        elements.append(elem)
sol, divs = algo.GMM(X=elements, k=50, init=[], dist=utils.cosine_dist_sparse)
for k in range(5, 51, 5):
    writer.writerow(["Lyrics", k, divs[k - 1]])
output.flush()

elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
sol, divs = algo.GMM(X=elements, k=50, init=[], dist=utils.manhattan_dist)
for k in range(5, 51, 5):
    writer.writerow(["Census", k, divs[k - 1]])
output.flush()

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
    sol, divs = algo.GMM(X=elements, k=20, init=[], dist=utils.euclidean_dist)
    writer.writerow(["Blobs-m" + str(m), 20, divs[19]])
    output.flush()

num_elem = [1000, 10000, 100000, 1000000, 10000000]
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
    sol, divs = algo.GMM(X=elements, k=20, init=[], dist=utils.euclidean_dist)
    writer.writerow(["Blobs-n" + str(n), 20, divs[19]])
    output.flush()
output.close()


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
        fair_greedy_flow = np.zeros([2, num_runs])
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
            sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=[10, 10], m=2,
                                                                                          dist=utils.euclidean_dist,
                                                                                          eps=0.1, dmax=6.0, dmin=varying_m[2],
                                                                                          metric_name='euclidean')
            print(sol)
        writer.writerow(["Blobs", 100000, 2, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-", np.average(fair_swap[1])])
        writer.writerow(["Blobs", 100000, 2, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
        writer.writerow(["Blobs", 100000, 2, 20, "Alg1", 0.1, np.average(alg1[0]), np.average(alg1[1]), np.average(alg1[2]), np.average(alg1[3]), np.average(alg1[2]) + np.average(alg1[3])])
        writer.writerow(["Blobs", 100000, 2, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
        writer.writerow(["Blobs", 100000, 2, 20, "FairGreedyFlow", "-", np.average(fair_greedy_flow[0]), "-", "-", "-", np.average(fair_greedy_flow[1])])
        output.flush()
    else:
        alg2 = np.zeros([4, num_runs])
        fair_flow = np.zeros([2, num_runs])
        fair_greedy_flow = np.zeros([2, num_runs])
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
            sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=group_k, m=m, dist=utils.euclidean_dist, eps=0.1, dmax=6.0, dmin=varying_m[m],
                                                                                          metric_name='euclidean')
            print(sol)
        writer.writerow(["Blobs", 100000, m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
        writer.writerow(["Blobs", 100000, m, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
        writer.writerow(["Blobs", 100000, m, 20, "FairGreedyFlow", "-", np.average(fair_greedy_flow[0]), "-", "-", "-", np.average(fair_greedy_flow[1])])

        output.flush()

elements.clear()
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
    if n < 100000:
        num_runs = 5
    else:
        num_runs = 1
    alg1 = np.zeros([4, num_runs])
    alg2 = np.zeros([4, num_runs])
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
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
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=[10, 10], m=2, dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m2[n][1], dmin=varying_n_m2[n][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow(["Blobs", n, 2, 20, "FairGreedyFlow", "-", np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
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
    if n < 100000:
        num_runs = 5
    else:
        num_runs = 1
    alg2 = np.zeros([4, num_runs])
    fair_flow = np.zeros([2, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements, k=[2] * 10, m=10, dist=utils.euclidean_dist)
        print(sol)
        sol, alg2[0][run], alg2[1][run], alg2[2][run], alg2[3][run] = algs.StreamFairDivMax2(X=elements, k=[2] * 10, m=10, dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m10[n][1], dmin=varying_n_m10[n][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements, k=[2] * 10, m=10, dist=utils.euclidean_dist, eps=0.1, dmax=varying_n_m10[n][1], dmin=varying_n_m10[n][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow(["Blobs", n, 10, 20, "FairGreedyFlow", "-", np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow(["Blobs", n, 10, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-", np.average(fair_flow[1])])
    writer.writerow(["Blobs", n, 10, 20, "Alg2", 0.1, np.average(alg2[0]), np.average(alg2[1]), np.average(alg2[2]), np.average(alg2[3]), np.average(alg2[2]) + np.average(alg2[3])])
    output.flush()
output.close()

