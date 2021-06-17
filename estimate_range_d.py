import csv
import random
import sys

import algorithms_offline
import utils


# estimate d_max and d_min for Adult
# elements = []
# with open("datasets/adult.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, 10):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[1]), features)
#         elements.append(elem)
# for k in range(5, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(10):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         solution, value = algorithms_offline.GMM(X=elements, k=k, init=[], dist=utils.euclidean_dist)
#         div_min = min(div_min, value[-1])
#         div_max = max(div_max, value[-1])
#         for c in range(2):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.euclidean_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# elements.clear()
# with open("datasets/adult.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, 10):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[2]), features)
#         elements.append(elem)
# for k in range(5, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(10):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         for c in range(5):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.euclidean_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# elements.clear()
# with open("datasets/adult.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, 10):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[3]), features)
#         elements.append(elem)
# for k in range(10, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(10):
#         random.Random(run).shuffle(elements)
#         for c in range(10):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.euclidean_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# # estimate d_max and d_min for Lyrics
# elements = []
# with open("datasets/lyrics.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = dict()
#         for i in range(2, len(row)):
#             if float(row[i]) >= 0.0001:
#                 features[i - 2] = float(row[i])
#         elem = utils.ElementSparse(int(row[0]), int(row[1]), features)
#         elements.append(elem)
# for k in range(15, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(5):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         solution, value = algorithms_offline.GMM(X=elements, k=k, init=[], dist=utils.cosine_dist_sparse)
#         div_min = min(div_min, value[-1])
#         div_max = max(div_max, value[-1])
#         for c in range(15):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.cosine_dist_sparse)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# # estimate d_max and d_min for Census
# elements = []
# with open("datasets/census.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, len(row)):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[1]), features)
#         elements.append(elem)
# for k in range(5, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(5):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         solution, value = algorithms_offline.GMM(X=elements, k=k, init=[], dist=utils.manhattan_dist)
#         div_min = min(div_min, value[-1])
#         div_max = max(div_max, value[-1])
#         for c in range(2):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# elements.clear()
# with open("datasets/census.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, len(row)):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[2]), features)
#         elements.append(elem)
# for k in range(10, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(5):
#         random.Random(run).shuffle(elements)
#         for new_idx in range(len(elements)):
#             elements[new_idx].idx = new_idx
#         for c in range(7):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)
#
# elements.clear()
# with open("datasets/census.csv", "r") as fileobj:
#     csvreader = csv.reader(fileobj, delimiter=',')
#     for row in csvreader:
#         features = []
#         for i in range(4, len(row)):
#             features.append(float(row[i]))
#         elem = utils.Element(int(row[0]), int(row[3]), features)
#         elements.append(elem)
# for k in range(15, 51, 5):
#     div_min = sys.float_info.max
#     div_max = 0
#     for run in range(5):
#         random.Random(run).shuffle(elements)
#         for c in range(14):
#             solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
#             div_min = min(div_min, value[-1])
#             div_max = max(div_max, value[-1])
#     print(k, div_min, div_max)

# estimate d_max and d_min for CelebA
elements = []
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[1]), features)
        elements.append(elem)
for k in range(5, 51, 5):
    div_min = sys.float_info.max
    div_max = 0
    for run in range(5):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        solution, value = algorithms_offline.GMM(X=elements, k=k, init=[], dist=utils.manhattan_dist)
        div_min = min(div_min, value[-1])
        div_max = max(div_max, value[-1])
        for c in range(2):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(k, div_min, div_max)

elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[2]), features)
        elements.append(elem)
for k in range(5, 51, 5):
    div_min = sys.float_info.max
    div_max = 0
    for run in range(5):
        random.Random(run).shuffle(elements)
        for new_idx in range(len(elements)):
            elements[new_idx].idx = new_idx
        for c in range(2):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(k, div_min, div_max)

elements.clear()
with open("datasets/celeba.csv", "r") as fileobj:
    csvreader = csv.reader(fileobj, delimiter=',')
    for row in csvreader:
        features = []
        for i in range(4, len(row)):
            features.append(float(row[i]))
        elem = utils.Element(int(row[0]), int(row[3]), features)
        elements.append(elem)
for k in range(5, 51, 5):
    div_min = sys.float_info.max
    div_max = 0
    for run in range(5):
        random.Random(run).shuffle(elements)
        for c in range(4):
            solution, value = algorithms_offline.GMMC(X=elements, color=c, k=k, init=[], dist=utils.manhattan_dist)
            div_min = min(div_min, value[-1])
            div_max = max(div_max, value[-1])
    print(k, div_min, div_max)
