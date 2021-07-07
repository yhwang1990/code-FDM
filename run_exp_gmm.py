import csv
import itertools

import algorithms_offline as algo
import utils

output = open("results_gmm.csv", "a")
writer = csv.writer(output)
writer.writerow(["dataset", "k", "div"])

output.flush()

elements = []
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
