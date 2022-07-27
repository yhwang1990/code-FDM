import csv
import random
import numpy as np
import algorithms_offline_window as algo
import algorithms_slidewindow as algsw
import utils
import copy
output = open("result_varying_m_w_window.csv", "a")
writer = csv.writer(output)
writer.writerow(["window_size", "dataset", "group", "m", "k", "algorithm", "param_eps", "div", "num_elem", "time1", "time2", "time3"])
output.flush()
'''
    varying m(k = 20,eps = 0.25, m = 2,4,...,20, w = 100000, n = 500000),
'''
# experiments for varying m
w = 100000
num_runs = 10
elements = []
elements_w = []
varying_m = {2: [3.1,5.1], 4: [3.0,5.0], 6: [3.0,4.8], 8: [3.0,4.8], 10: [2.8,4.5], 12: [2.8,4.5], 14: [2.8,4.5], 16: [2.5,4], 18: [2.5,4], 20: [2,3.3]}
for m in range(2, 21, 2):
    elements.clear()
    elements_w.clear()
    with open("datasets/blobs_n500000_m" + str(m) + ".csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    k = 20
    group_m = [0] * m
    remainder = k % m
    for c in range(m):
        if c < remainder:
            group_m[c] = k // m + 1
        else:
            group_m[c] = k // m
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
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_m, dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_m, m=2, dist=utils.euclidean_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_m,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_m[m][1],
                                                                       dmin=varying_m[m][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_m,
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_m[m][1],
                                                                       dmin=varying_m[m][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_m, m=m,
                                                                                      dist=utils.euclidean_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_m[m][1],
                                                                                      dmin=varying_m[m][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Blobs", "-", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Blobs", "-", m, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "Blobs", "-", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Blobs", "-", m, 20, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]), np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "Blobs", "-", m, 20, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()
'''
    varying w(k = 20,eps = 0.25, m = 2/10, w = 1000,4000,...,1024000)
'''
# synthetic,m=2
elements_all = []
varying_n_m2 = {1000: [2.5, 4.0], 4000: [2.5, 4.0], 16000: [2.7, 4.2], 64000: [2.8, 4.3], 256000: [2.8, 4.3], 1024000: [4.0, 6.1]}
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements_all.append(elem)
values_window = [1000, 4000, 16000, 64000, 256000, 1024000]
elements.clear()
num_runs = 10
for w in values_window:
    elements.clear()
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    elements = copy.deepcopy(elements_all[-2*w:])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=[10, 10], dist=utils.euclidean_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=[10, 10], m=2,
                                                                  dist=utils.euclidean_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=[10, 10],
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n_m2[w][1], dmin=varying_n_m2[w][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=[10, 10],
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n_m2[w][1], dmin=varying_n_m2[w][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=[10, 10], m=2,
                                                                                      dist=utils.euclidean_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_n_m2[w][1],
                                                                                      dmin=varying_n_m2[w][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Blobs", "-", 2, 20, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Blobs", "-", 2, 20, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "Blobs", "-", 2, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Blobs", "-", 2, 20, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]), np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "Blobs", "-", 2, 20, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

varying_n_m10 = {1000: [1.5, 2.5], 4000: [2.0, 3.5], 16000: [2.3, 3.8], 64000: [2.5, 4.0], 256000: [2.5, 4.0], 1024000: [2.5, 4.0]} # 3.1, 6.1
# synthetic,m=10
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements_all.append(elem)
values_window = [1000, 4000, 16000, 64000, 256000, 1024000]
elements.clear()
for w in values_window:
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    elements = copy.deepcopy(elements_all[-2*w:])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=[2,2,2,2,2,2,2,2,2,2], m=10,
                                                                  dist=utils.euclidean_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=[2,2,2,2,2,2,2,2,2,2],
                                                                       dist=utils.euclidean_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n_m10[w][1], dmin=varying_n_m10[w][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=[2,2,2,2,2,2,2,2,2,2], m=10,
                                                                                      dist=utils.euclidean_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_n_m10[w][1],
                                                                                      dmin=varying_n_m10[w][0],
                                                                                      metric_name='euclidean')
        print(sol)
    writer.writerow([w, "Blobs", "-", 10, 20, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Blobs", "-", 10, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Blobs", "-", 10, 20, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# census
elements_all.clear()
num_runs = 10
values_window = [1000, 4000, 16000, 64000, 256000, 1024000]
# read the Census dataset grouped by sex (m=2)
with open("datasets/census_age.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements_all.append(elem)
m = 2
k = 20
group_k = [0] * m
remainder = k % m
for c in range(m):
    if c < remainder:
        group_k[c] = k // m + 1
    else:
        group_k[c] = k // m
varying_n = {1000: [17, 26.7], 4000: [17, 26.7], 16000: [17, 26.7], 64000: [17, 26.7], 256000: [17, 26.7], 1024000: [17, 26.7]}
elements.clear()
for w in values_window:
    fair_swap = np.zeros([2, num_runs])
    fair_flow = np.zeros([2, num_runs])
    swdm1 = np.zeros([4, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    elements.clear()
    elements = copy.deepcopy(elements_all[-2*w:])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_swap[0][run], fair_swap[1][run] = algo.FairSwap(X=elements_w, k=group_k, dist=utils.manhattan_dist)
        print(sol)
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=2,
                                                                  dist=utils.manhattan_dist)
        print(sol)
        sol, swdm1[0][run], swdm1[1][run], swdm1[2][run], swdm1[3][run] = algsw.SFDM1(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n[w][1], dmin=varying_n[w][0])
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n[w][1], dmin=varying_n[w][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_n[w][1],
                                                                                      dmin=varying_n[w][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Sex", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Sex", m, k, "FairSwap", "-", np.average(fair_swap[0]), "-", "-", "-",
                     np.average(fair_swap[1])])
    writer.writerow([w, "Census", "Sex", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Sex", m, k, "swdm1", 0.25, np.average(swdm1[0]), np.average(swdm1[3]), np.average(swdm1[1]),
                     np.average(swdm1[2]), np.average(swdm1[1]) + np.average(swdm1[2])])
    writer.writerow([w, "Census", "Sex", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# read the Census dataset grouped by age (m=7)
elements_all.clear()
num_runs = 10
values_window = [1000, 4000, 16000, 64000, 256000, 1024000]
with open("datasets/census_age.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements_all.append(elem)
m = 7
k = 20
group_k = [0] * m
remainder = k % m
for c in range(m):
    if c < remainder:
        group_k[c] = k // m + 1
    else:
        group_k[c] = k // m
varying_n = {1000: [2.8, 4.5], 4000: [4.2, 6.7], 16000: [4.3,6.8], 64000: [4.3,6.8], 256000: [4.7,7.4], 1024000: [5.0,7.9]}
elements.clear()
elements_w.clear()
for w in values_window:
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    fair_greedy_flow = np.zeros([2, num_runs])
    elements.clear()
    elements_w.clear()
    elements = copy.deepcopy(elements_all[-2*w:])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=group_k, m=m,
                                                                  dist=utils.manhattan_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=group_k,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n[w][1], dmin=varying_n[w][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=group_k, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_n[w][1],
                                                                                      dmin=varying_n[w][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Age", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Age", m, k, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Age", m, k, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()

# read the Census dataset grouped by sex+age (m=14)
elements_all.clear()
num_runs = 10
with open("datasets/census_age.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements_all.append(elem)
m = 14
k_20 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
values_window = [1000, 4000, 16000, 64000, 256000, 1024000]
varying_n = {1000: [2.2, 3.5], 4000: [2.8, 4.5], 16000: [4.3, 6.8], 64000: [5.0, 7.9], 256000: [5.5, 8.7], 1024000: [6.0, 9.4]}
elements.clear()
elements_w.clear()
for w in values_window:
    fair_flow = np.zeros([2, num_runs])
    swdm2 = np.zeros([4, num_runs])
    elements.clear()
    elements_w.clear()
    elements = copy.deepcopy(elements_all[-2*w:])
    fair_greedy_flow = np.zeros([2, num_runs])
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        elements_w = copy.deepcopy(elements[-w:])
        for new_idx in range(len(elements_w)):
            elements_w[new_idx].idx = new_idx
        sol, fair_flow[0][run], fair_flow[1][run] = algo.FairFlow(X=elements_w, k=k_20, m=m,
                                                                  dist=utils.manhattan_dist)
        print(sol)
        sol, swdm2[0][run], swdm2[1][run], swdm2[2][run], swdm2[3][run] = algsw.SFDM2(X=elements, w=w, k=k_20,
                                                                       dist=utils.manhattan_dist,
                                                                       eps=0.25,
                                                                       dmax=varying_n[w][1], dmin=varying_n[w][0])
        print(sol)
        sol, fair_greedy_flow[0][run], fair_greedy_flow[1][run] = algo.FairGreedyFlow(X=elements_w, k=k_20, m=m,
                                                                                      dist=utils.manhattan_dist,
                                                                                      eps=0.25,
                                                                                      dmax=varying_n[w][1],
                                                                                      dmin=varying_n[w][0],
                                                                                      metric_name='cityblock')
        print(sol)
    writer.writerow([w, "Census", "Both", m, k, "FairGreedyFlow", 0.25, np.average(fair_greedy_flow[0]), "-", "-", "-",
                     np.average(fair_greedy_flow[1])])
    writer.writerow([w, "Census", "Both", m, 20, "FairFlow", "-", np.average(fair_flow[0]), "-", "-", "-",
                     np.average(fair_flow[1])])
    writer.writerow([w, "Census", "Both", m, 20, "swdm2", 0.25, np.average(swdm2[0]), np.average(swdm2[3]), np.average(swdm2[1]),
                     np.average(swdm2[2]), np.average(swdm2[1]) + np.average(swdm2[2])])
    output.flush()
output.close()
