import csv
import itertools
import random
import sys

import algorithms_offline
import utils

# estimate d_max and d_min for blobs datasets
elements = []
with open("datasets/blobs_n100000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(5):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=20, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=20, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m2", d_min, d_max)

for num_colors in range(4, 21, 2):
    elements.clear()
    with open("datasets/blobs_n100000_m" + str(num_colors) + ".csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in csvreader:
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    div_min = sys.float_info.max
    for run in range(5):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        for c in range(num_colors):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=20, init=[], dist=utils.euclidean_dist)
            print(solution, value[-1])
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(div_min, div_max)
    d_min = div_min * 0.75
    d_max = div_max * 1.25
    print("n100000_m" + str(num_colors), d_min, d_max)

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
    div_min = sys.float_info.max
    div_max = 0
    num_runs = 0
    if n <= 100000:
        num_runs = 5
    else:
        num_runs = 2
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        solution, value = algorithms_offline.GMM(X=elements, k=20, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
        for c in range(2):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=20, init=[], dist=utils.euclidean_dist)
            print(solution, value[-1])
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(div_min, div_max)
    d_min = div_min * 0.75
    d_max = div_max * 1.25
    print("n" + str(n) + "_m2", d_min, d_max)

    elements.clear()
    with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
        csvreader = csv.reader(fileobj, delimiter=',')
        for row in itertools.islice(csvreader, n):
            features = []
            for i in range(2, len(row)):
                features.append(float(row[i]))
            elem = utils.Element(int(row[0]), int(row[1]), features)
            elements.append(elem)
    div_min = sys.float_info.max
    num_runs = 0
    if n <= 100000:
        num_runs = 5
    else:
        num_runs = 2
    for run in range(num_runs):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        for c in range(10):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=20, init=[], dist=utils.euclidean_dist)
            print(solution, value[-1])
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(div_min, div_max)
    d_min = div_min * 0.75
    d_max = div_max * 1.25
    print("n" + str(n) + "_m10", d_min, d_max)
