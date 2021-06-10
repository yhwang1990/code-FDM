import csv
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n100000_m5.csv", "r") as fileobj:
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(5):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m5", d_min, d_max)

elements.clear()
with open("datasets/blobs_n100000_m10.csv", "r") as fileobj:
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m10", d_min, d_max)

elements.clear()
with open("datasets/blobs_n100000_m15.csv", "r") as fileobj:
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(15):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m15", d_min, d_max)

elements.clear()
with open("datasets/blobs_n100000_m20.csv", "r") as fileobj:
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(20):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m20", d_min, d_max)

elements.clear()
with open("datasets/blobs_n100000_m25.csv", "r") as fileobj:
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
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(25):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m25", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(1000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n1000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(1000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n1000_m10", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(10000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n10000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(10000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n10000_m10", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(100000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(100000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n100000_m10", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(1000000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n1000000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(1000000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n1000000_m10", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m2.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(10000000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(2):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n10000000_m2", d_min, d_max)

elements.clear()
with open("datasets/blobs_n10000000_m10.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for idx in range(10000000):
        row = csvreader.next()
        features = []
        for i in range(2, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
div_min = sys.float_info.max
div_max = 0
for run in range(10):
    random.Random(run).shuffle(elements)
    for new_idx in range(len(elements)):
        elements[new_idx].idx = new_idx
    solution, value = algorithms_offline.GMM(X=elements, k=50, init=[], dist=utils.euclidean_dist)
    print(solution, value[-1])
    div_min = min(div_min, value[-1])
    div_max = max(div_max, value[-1])
    for c in range(10):
        solution, value = algorithms_offline.GMMC(X=elements, color=c, k=50, init=[], dist=utils.euclidean_dist)
        print(solution, value[-1])
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
print(div_min, div_max)
d_min = div_min * 0.75
d_max = div_max * 1.25
print("n10000000_m10", d_min, d_max)
