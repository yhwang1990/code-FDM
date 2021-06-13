import argparse
import numpy as np
from sklearn.datasets import make_blobs


def generate_blobs(size: int, group: int, dim: int) -> None:
    X, y = make_blobs(n_samples=size, centers=10, n_features=dim, random_state=17)
    g = np.random.randint(group, size=size, dtype=np.uint8)
    with open("datasets/blobs_n" + str(size) + "_m" + str(group) + ".csv", "a") as fileobj:
        for i in range(size):
            fileobj.write(str(i) + ",")
            fileobj.write(str(g[i]) + ",")
            for j in range(dim - 1):
                fileobj.write(format(X[i][j], ".6f") + ",")
            fileobj.write(format(X[i][dim - 1], ".6f") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", default=1000, type=int)
    parser.add_argument("-m", "--group", default=2, type=int)
    parser.add_argument("-d", "--dim", default=2, type=int)
    args = parser.parse_args()
    generate_blobs(args.size, args.group, args.dim)
