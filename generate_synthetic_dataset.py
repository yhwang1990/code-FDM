import argparse
import numpy as np
from sklearn.datasets import make_blobs

global args


def generate_blobs():
    X, y = make_blobs(n_samples=args.size, centers=10, n_features=args.dim, random_state=17)
    g = np.random.randint(args.group, size=args.size, dtype=np.uint8)
    with open("datasets/blobs_n" + str(args.size) + "_m" + str(args.group) + ".csv", "a") as fileobj:
        for i in range(args.size):
            fileobj.write(str(i) + ",")
            fileobj.write(str(g[i]) + ",")
            for j in range(args.dim - 1):
                fileobj.write(format(X[i][j], ".6f") + ",")
            fileobj.write(format(X[i][args.dim - 1], ".6f") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", default=1000, type=int)
    parser.add_argument("-m", "--group", default=2, type=int)
    parser.add_argument("-d", "--dim", default=2, type=int)
    args = parser.parse_args()
    generate_blobs()
