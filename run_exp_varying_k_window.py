import csv
import random
import numpy as np
import algorithms_offline_window as algo
import algorithms_slidewindow as algsw
import utils
import copy

output = open("results_exp_varying_k_window.csv", "a")
writer = csv.writer(output)
writer.writerow(["window_size", "dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()

'''
    varying k(eps = 0.25, m = 2/5/10, w = 25000/100000),
    containing defalut(k = 20, eps = 0.25, m = 2/5/10, w = 25000/100000)
'''
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
range_d_sex = {5: [2.9, 6.9], 10: [2.7, 6.6], 15: [2.8, 7.0], 20: [2.5, 6.2], 25: [2.3, 5.8],
               30: [2.1, 5.2], 35: [2.0, 5.0], 40: [1.8, 4.6], 45: [1.7, 4.3], 50: [1.7, 4.3]}
m = 2
num_runs = 10
w = 25000
elements_w = []
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
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_k, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_k,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                             dist=utils.euclidean_dist,
                                                                                             eps=0.25,
                                                                                              dmax=range_d_sex[k][1],
                                                                                            dmin=range_d_sex[k][0],
                                                                                             metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Adult", "Sex", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]),  "-", "-", "-", np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Adult", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "Adult", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Adult", "Sex", m, k, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]),np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "Adult", "Sex", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# read the Adult dataset grouped by race (m=5)
elements_w.clear()
elements.clear()
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)
m = 5
range_d_race = {5: [3.5, 6], 10: [2.4, 4.1], 15: [2.1, 3.7], 20: [1.5, 2.5], 25: [1.5, 2.4],
                30: [1.5, 2.4], 35: [1.3, 2.2], 40: [1.3, 2.2], 45: [1.2, 2.0], 50: [1.2, 2.1]}
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
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_race[k][1],
                                                                       dmin=range_d_race[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.euclidean_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_race[k][1],
                                                                                      dmin=range_d_race[k][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Adult", "Race", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Adult", "Race", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Adult", "Race", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()
# read the Adult dataset grouped by sex+race (m=10)
elements_w.clear()
elements.clear()
with open("datasets/adult.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)
range_d_both = {5: [2.0, 6.5], 10: [2.0, 2.7], 15: [2.0, 2.7], 20: [1.5, 2.3], 25: [1.5, 2.3],
               30: [1.2, 1.6], 35: [1.2, 1.6], 40: [1.0, 1.4], 45: [1.0, 1.4], 50: [1.0, 1.4]}
m = 10
for k in values_k[1:]:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.euclidean_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_both[k][1],
                                                                       dmin=range_d_both[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.euclidean_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_both[k][1],
                                                                                      dmin=range_d_both[k][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Adult", "Sex + Race", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Adult", "Sex + Race", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Adult", "Sex + Race", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()
# celebA
elements_w.clear()
elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
range_d_sex = {5: [8.5, 21.3], 10: [7, 17.5], 15: [6.5, 16.3], 20: [6.5, 16.3], 25: [6, 15],
               30: [6, 15], 35: [5.5, 13.8], 40: [5.5, 13.8], 45: [5.5, 13.8], 50: [5.5, 13.8]}
m = 2
w = 100000
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
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_k, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_sex[k][1],
                                                                                      dmin=range_d_sex[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "CelebA", "Sex", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
         np.average(fair_greedy_flow[1])])
    writer.writerow([w, "CelebA", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "CelebA", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "CelebA", "Sex", m, k, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]),np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "CelebA", "Sex", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

elements_w.clear()
elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)
range_d_age = {5: [8.5, 21.3], 10: [7, 17.5], 15: [6.5, 16.3], 20: [6.5, 16.3], 25: [6, 15],
               30: [6, 15], 35: [5.5, 13.8], 40: [5.5, 13.8], 45: [5.5, 13.8], 50: [5.5, 13.8]}
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
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_k, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_age[k][1],
                                                                       dmin=range_d_age[k][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_age[k][1],
                                                                       dmin=range_d_age[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_age[k][1],
                                                                                      dmin=range_d_age[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "CelebA", "Age", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "CelebA", "Age", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "CelebA", "Age", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "CelebA", "Age", m, k, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]),np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "CelebA", "Age", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

elements_w.clear()
elements.clear()
num_runs = 10
w = 100000
elements = []
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)
range_d_both = {5: [8.5, 21.3], 10: [7, 17.5], 15: [6.5, 16.3], 20: [6.5, 16.3], 25: [6, 15],
               30: [6, 15], 35: [5.5, 13.8], 40: [5.5, 13.8], 45: [5.5, 13.8], 50: [5.5, 13.8]}
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
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_both[k][1],
                                                                       dmin=range_d_both[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_both[k][1],
                                                                                      dmin=range_d_both[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "CelebA", "Both", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "CelebA", "Both", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "CelebA", "Both", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# census
# read the Census dataset grouped by sex (m=2)
num_runs = 10
elements_w.clear()
elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
elements = elements[-200000:]
range_d_sex = {5: [22, 41], 10: [20, 37], 15: [17, 34], 20: [16, 32], 25: [15, 31],
               30: [15, 31], 35: [14, 30], 40: [14, 30], 45: [13, 29], 50: [13, 29]}

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
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_k, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_sex[k][1],
                                                                       dmin=range_d_sex[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_sex[k][1],
                                                                                      dmin=range_d_sex[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Sex", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "Census", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Sex", m, k, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]),np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "Census", "Sex", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# read the Census dataset grouped by age (m=7)
elements_w.clear()
elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)
elements = elements[-200000:]
range_d_age = {5: [4, 7], 10: [4, 7], 15: [6, 8.5], 20: [5.5, 8], 25: [5.0, 7.5],
               30: [5.0, 7.5], 35: [5.0, 7.5], 40: [4, 6.5], 45: [3.8, 6.3], 50: [3.8, 6.3]}

m = 7
# experiments on varying k
for k in values_k[1:]:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])

    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_age[k][1],
                                                                       dmin=range_d_age[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_age[k][1],
                                                                                      dmin=range_d_age[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Age", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Age", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Age", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]),np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# read the Census dataset grouped by sex+age (m=14)
elements_w.clear()
elements.clear()
with open("datasets/census.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)
elements = elements[-200000:]
range_d_both = {5: [20, 37], 10: [4, 7], 15: [6, 8.5], 20: [5.5, 8], 25: [4.5, 7.0],
               30: [4, 6], 35: [4.5, 6.5], 40: [4, 6], 45: [3.5, 5.5], 50: [3.5, 5.5]}
m = 14
for k in values_k[2:]:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.manhattan_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=range_d_both[k][1], dmin=range_d_both[k][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=range_d_both[k][1],
                                                                                      dmin=range_d_both[k][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Both", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Both", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Both", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# lyrics
# read the Lyrics dataset grouped by genre (m=15)
num_runs = 10
elements_w.clear()
elements.clear()
elements = []
values_k = range(5, 51, 5)
w = 100000
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = dict()
        for i in range(2, len(row)):
            if float(row[i]) >= 0.0001:
                features[i - 2] = float(row[i])
        elem = utils.ElementSparse(int(row[0]), int(row[1]), features)
        elements.append(elem)
range_d = {15: [1.3, 1.57], 20: [0.95, 1.15], 25: [1, 1.15],
               30: [1, 1.15], 35: [1.05, 1.2], 40: [1, 1.15], 45: [0.95, 1.1], 50: [0.9, 1.00]}
m = 15
# experiments on varying k
for k in values_k[2:]:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m, dist=utils.cosine_dist_sparse)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.cosine_dist_sparse,
                                                                       eps=0.1,dmax=range_d[k][1], dmin=range_d[k][0])
        print(sol)
    writer.writerow([w, "Lyrics", "Genre", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Lyrics", "Genre", m, k, "swdm2", 0.1, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# FairGreedyFlow
with open("datasets/lyrics.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
range_d = {15: [1.3, 1.57], 20: [0.95, 1.15], 25: [1, 1.15],
               30: [1, 1.15], 35: [1.05, 1.2], 40: [1, 1.15], 45: [0.95, 1.1], 50: [0.9, 1.00]}
m = 15
# experiments on varying k
for k in values_k[2:]:
    if k < m:
        continue
    group_k = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_k[c] = k // m + 1
        else:
            group_k[c] = k // m
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.cosine_dist,
                                                                                      eps=0.1,
                                                                                      dmax=range_d[k][1],
                                                                                      dmin=range_d[k][0],
                                                                                      metric_name='cosine')
        print(sol)
    writer.writerow([w, "Lyrics", "Genre", m, k, "FairGreedyFlow", 0.1, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    output.flush()