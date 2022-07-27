import sys
import time
import math
import numpy as np
import utils
from typing import Any, Callable, Iterable, List, Set, Union
import networkx as nx

ElemList = Union[List[utils.Element], List[utils.ElementSparse]]

def GMM(X: ElemList, idxs: List[int], k: int, dist: Callable[[Any, Any], float]) -> (Set[int], List[float]):
    S, div = [], []
    dist_array = np.full(len(idxs), sys.float_info.max)
    S.append(idxs[0])
    div.append(sys.float_info.max)
    for i in range(len(idxs)):
        dist_array[i] = dist(X[idxs[0]], X[idxs[i]])
    while len(S) < k:
        max_idx = np.argmax(dist_array)
        max_dist = np.max(dist_array)
        S.append(idxs[max_idx])
        div.append(max_dist)
        for i in range(len(idxs)):
            dist_array[i] = min(dist_array[i], dist(X[idxs[i]], X[idxs[max_idx]]))
    return set(S), div[k-1]


def diversity(X: ElemList, idxs: Iterable[int], dist: Callable[[Any, Any], float]) -> float:
    div_val = sys.float_info.max
    for idx1 in idxs:
        for idx2 in idxs:
            if idx1 != idx2:
                div_val = min(div_val, dist(X[idx1], X[idx2]))
    return div_val


def SlideWindowDivMax(X: ElemList, k: int, w: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (Set[int], float, float, float):
    t0 = time.perf_counter()
    Lambda = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    Mu = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    A = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    A_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    B = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    B_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    Div = [[0.0 for j in range(len(Mu))] for i in range(len(Lambda))]
    for x in X:
        for i in range(len(Lambda)):
            for j in range(len(Mu)):
                if len(B[i][j]) == 0:
                    B[i][j].add(x.idx)
                    B_apo[i][j][x.idx] = x.idx
                    Div[i][j] = sys.float_info.max
                else:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in B[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if len(B[i][j]) < k and div_x >= Mu[j]:
                        B[i][j].add(x.idx)
                        B_apo[i][j][x.idx] = x.idx
                        Div[i][j] = min(Div[i][j], div_x)
                    elif div_x < Mu[j]:
                        B_apo[i][j][div_idx] = x.idx
                if A[i][j]:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in A[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if div_x < Mu[j]:
                        A_apo[i][j][div_idx] = x.idx
            div_B_lambda = 0
            for j in range(len(B[i])):
                if len(B[i][j]) == k:
                    div_B_lambda = max(div_B_lambda, Div[i][j])
            if div_B_lambda > Lambda[i]:
                for j in range(len(B[i])):
                    if x.idx in B[i][j]:
                        B[i][j].remove(x.idx)
                    if x.idx in B_apo[i][j].keys():
                        B_apo[i][j].pop(x.idx)
                    A[i][j] = B[i][j].copy()
                    A_apo[i][j] = B_apo[i][j].copy()
                    B[i][j] = {x.idx}
                    B_apo[i][j] = {x.idx: x.idx}
                    Div[i][j] = sys.float_info.max
    t1 = time.perf_counter()
    #  Post-processing
    w_start = max(0, len(X)-w+1)
    Sol = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    Divs = [[0 for j in range(len(Mu))] for i in range(len(Lambda))]
    for i in range(len(Lambda)):
        for j in range(len(Mu)):
            if len(A[i][j]) > 0 and min(A[i][j]) >= w_start:
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(A[i][j].union(B[i][j])), k=k, dist=dist)
            elif min(B[i][j]) >= w_start:
                elem_A = B[i][j].copy()
                for value in A_apo[i][j].values():
                    if value >= w_start:
                        elem_A.add(value)
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(elem_A), k=k, dist=dist)
    max_index = np.array(Divs).argmax()
    t2 = time.perf_counter()
    return np.array(Sol).flatten()[max_index], np.array(Divs).flatten()[max_index], (t1 - t0)/len(X), t2 - t1


def SFDM1(X: ElemList, k: List[int], w: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (Set[int], float, float, float, int):
    t0 = time.perf_counter()
    m, sum_k = len(k), sum(k)
    Lambda = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    Mu = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    A = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    A_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    B = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    B_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    Div = [[0.0 for j in range(len(Mu))] for i in range(len(Lambda))]
    A_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    A_apo_m = [[[{} for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    B_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    B_apo_m = [[[{} for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    Div_m = [[[0.0 for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    for x in X:
        for i in range(len(Lambda)):
            for j in range(len(Mu)):
                if len(B[i][j]) == 0:
                    B[i][j].add(x.idx)
                    B_apo[i][j][x.idx] = x.idx
                    Div[i][j] = sys.float_info.max
                else:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in B[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if len(B[i][j]) < sum_k and div_x >= Mu[j]:
                        B[i][j].add(x.idx)
                        B_apo[i][j][x.idx] = x.idx
                        Div[i][j] = min(Div[i][j], div_x)
                    elif div_x < Mu[j]:
                        B_apo[i][j][div_idx] = x.idx
                if A[i][j]:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in A[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if div_x < Mu[j]:
                        A_apo[i][j][div_idx] = x.idx
            div_B_lambda = 0
            for j in range(len(B[i])):
                if len(B[i][j]) == sum_k:
                    div_B_lambda = max(div_B_lambda, Div[i][j])
            if div_B_lambda > Lambda[i]:
                for j in range(len(B[i])):
                    if x.idx in B[i][j]:
                        B[i][j].remove(x.idx)
                    if x.idx in B_apo[i][j].keys():
                        B_apo[i][j].pop(x.idx)
                    A[i][j] = B[i][j].copy()
                    A_apo[i][j] = B_apo[i][j].copy()
                    B[i][j] = {x.idx}
                    B_apo[i][j] = {x.idx: x.idx}
                    Div[i][j] = sys.float_info.max
        c = x.color
        for i in range(len(Lambda)):
            for j in range(len(Mu)):
                if len(B_m[i][j][c]) == 0:
                    B_m[i][j][c].add(x.idx)
                    B_apo_m[i][j][c][x.idx] = x.idx
                    Div_m[i][j][c] = sys.float_info.max
                else:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in B_m[i][j][c]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if len(B_m[i][j][c]) < k[c] and div_x >= Mu[j]:
                        B_m[i][j][c].add(x.idx)
                        B_apo_m[i][j][c][x.idx] = x.idx
                        Div_m[i][j][c] = min(Div_m[i][j][c], div_x)
                    elif div_x < Mu[j]:
                        B_apo_m[i][j][c][div_idx] = x.idx
                if A_m[i][j][c]:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in A_m[i][j][c]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if div_x < Mu[j]:
                        A_apo_m[i][j][c][div_idx] = x.idx
            div_B_lambda = 0
            for j in range(len(B_m[i])):
                if len(B_m[i][j][c]) == k[c]:
                    div_B_lambda = max(div_B_lambda, Div_m[i][j][c])
            if div_B_lambda > Lambda[i]:
                for j in range(len(B[i])):
                    if x.idx in B_m[i][j][c]:
                        B_m[i][j][c].remove(x.idx)
                    if x.idx in B_apo_m[i][j][c].keys():
                        B_apo_m[i][j][c].pop(x.idx)
                    A_m[i][j][c] = B_m[i][j][c].copy()
                    A_apo_m[i][j][c] = B_apo_m[i][j][c].copy()
                    B_m[i][j][c] = {x.idx}
                    B_apo_m[i][j][c] = {x.idx: x.idx}
                    Div_m[i][j][c] = sys.float_info.max
    t1 = time.perf_counter()
    #count
    stored_elements = set()
    for i in range(len(Lambda)):
        for j in range(len(Mu)):
            stored_elements.update(A[i][j])
            stored_elements.update(B[i][j])
            stored_elements.update(A_apo[i][j].values())
            stored_elements.update(B_apo[i][j].values())
            for c in range(m):
                stored_elements.update(A_m[i][j][c])
                stored_elements.update(B_m[i][j][c])
                stored_elements.update(A_apo_m[i][j][c].values())
                stored_elements.update(B_apo_m[i][j][c].values())
    num_elements = len(stored_elements)
    #  Post-processing
    w_start = max(0, len(X)-w+1)
    Sol = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    Divs = [[0 for j in range(len(Mu))] for i in range(len(Lambda))]
    Sol_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    Divs_m = [[[0 for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    for i in range(len(Lambda)):
        for j in range(len(Mu)):
            if len(A[i][j]) > 0 and min(A[i][j]) >= w_start:
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(A[i][j].union(B[i][j])), k=sum_k, dist=dist)
            elif min(B[i][j]) >= w_start:
                elem_A = B[i][j].copy()
                for value in A_apo[i][j].values():
                    if value >= w_start:
                        elem_A.add(value)
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(elem_A), k=sum_k, dist=dist)
            for c in range(m):
                idxs_c_in_Sol_i_j = [x.idx for x in X if x.color == c and x.idx in Sol[i][j]].copy()
                if len(Sol[i][j]) == sum_k and len(idxs_c_in_Sol_i_j) < k[c]:
                    if len(A_m[i][j][c]) > 0 and min(A_m[i][j][c]) >= w_start:
                        Sol_m[i][j][c], Divs_m[i][j][c] = GMM(X, idxs=list(A_m[i][j][c].union(B_m[i][j][c])), k=k[c], dist=dist)
                    elif min(B_m[i][j][c]) >= w_start:
                        elem_A = B_m[i][j][c].copy()
                        for value in A_apo_m[i][j][c].values():
                            if value >= w_start:
                                elem_A.add(value)
                        Sol_m[i][j][c], Divs_m[i][j][c] = GMM(X, idxs=list(elem_A), k=k[c], dist=dist)
                    while len(idxs_c_in_Sol_i_j) < k[c]:
                        max_div, max_idx = 0.0, -1
                        for idx1 in Sol_m[i][j][c]:
                            if idx1 not in idxs_c_in_Sol_i_j:
                                div1 = sys.float_info.max
                                for idx2 in idxs_c_in_Sol_i_j:
                                    div1 = min(div1, dist(X[idx1], X[idx2]))
                                if div1 > max_div:
                                    max_div, max_idx = div1, idx1
                        Sol[i][j].add(max_idx)
                        idxs_c_in_Sol_i_j.append(max_idx)
                    while len(Sol[i][j]) > sum_k:
                        min_div, min_idx = sys.float_info.max, -1
                        idxs_c_not_in_S_i_j = [x.idx for x in X if x.color != c and x.idx in Sol[i][j]]
                        for idx1 in idxs_c_not_in_S_i_j:
                            div1 = sys.float_info.max
                            for idx2 in idxs_c_in_Sol_i_j:
                                div1 = min(div1, dist(X[idx1], X[idx2]))
                            if div1 < min_div:
                                min_div, min_idx = div1, idx1
                        Sol[i][j].remove(min_idx)
    fair_div = []
    for idxs in np.array(Sol).flatten():
        if len(idxs) == sum_k:
            fair_div.append(diversity(X, idxs, dist))
        else:
            fair_div.append(-1)
    max_index = np.array(fair_div).argmax()
    t2 = time.perf_counter()
    print('sfdm1 Sol',Sol)
    return np.array(Sol).flatten()[max_index], fair_div[max_index], (t1 - t0)/len(X), t2 - t1, num_elements


def SFDM2(X: ElemList, k: List[int], w: int, dist: Callable[[Any, Any], float], eps: float, dmax: float, dmin: float) -> (Set[int], float, float, float, int):
    t0 = time.perf_counter()
    m, sum_k = len(k), sum(k)
    Lambda = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    Mu = [dmin / ((1 - eps) ** i) for i in range(math.floor(math.log(dmin / dmax, 1 - eps)) + 1)]
    A = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    A_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    B = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    B_apo = [[{} for j in range(len(Mu))] for i in range(len(Lambda))]
    Div = [[0.0 for j in range(len(Mu))] for i in range(len(Lambda))]
    A_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    A_apo_m = [[[{} for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    B_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    B_apo_m = [[[{} for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    Div_m = [[[0.0 for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    for x in X:
        for i in range(len(Lambda)):
            for j in range(len(Mu)):
                if len(B[i][j]) == 0:
                    B[i][j].add(x.idx)
                    B_apo[i][j][x.idx] = x.idx
                    Div[i][j] = sys.float_info.max
                else:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in B[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if len(B[i][j]) < sum_k and div_x >= Mu[j]:
                        B[i][j].add(x.idx)
                        B_apo[i][j][x.idx] = x.idx
                        Div[i][j] = min(Div[i][j], div_x)
                    elif div_x < Mu[j]:
                        B_apo[i][j][div_idx] = x.idx
                if A[i][j]:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in A[i][j]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if div_x < Mu[j]:
                        A_apo[i][j][div_idx] = x.idx
            div_B_lambda = 0
            for j in range(len(B[i])):
                if len(B[i][j]) == sum_k:
                    div_B_lambda = max(div_B_lambda, Div[i][j])
            if div_B_lambda > Lambda[i]:
                for j in range(len(B[i])):
                    if x.idx in B[i][j]:
                        B[i][j].remove(x.idx)
                    if x.idx in B_apo[i][j].keys():
                        B_apo[i][j].pop(x.idx)
                    A[i][j] = B[i][j].copy()
                    A_apo[i][j] = B_apo[i][j].copy()
                    B[i][j] = {x.idx}
                    B_apo[i][j] = {x.idx: x.idx}
                    Div[i][j] = sys.float_info.max
        c = x.color
        for i in range(len(Lambda)):
            for j in range(len(Mu)):
                if len(B_m[i][j][c]) == 0:
                    B_m[i][j][c].add(x.idx)
                    B_apo_m[i][j][c][x.idx] = x.idx
                    Div_m[i][j][c] = sys.float_info.max
                else:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in B_m[i][j][c]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if len(B_m[i][j][c]) < sum_k and div_x >= Mu[j]:
                        B_m[i][j][c].add(x.idx)
                        B_apo_m[i][j][c][x.idx] = x.idx
                        Div_m[i][j][c] = min(Div_m[i][j][c], div_x)
                    elif div_x < Mu[j]:
                        B_apo_m[i][j][c][div_idx] = x.idx
                if A_m[i][j][c]:
                    div_x, div_idx = sys.float_info.max, -1
                    for y_idx in A_m[i][j][c]:
                        cur = dist(x, X[y_idx])
                        if cur < div_x:
                            div_x = cur
                            div_idx = y_idx
                    if div_x < Mu[j]:
                        A_apo_m[i][j][c][div_idx] = x.idx
            div_B_lambda = 0
            for j in range(len(B_m[i])):
                if len(B_m[i][j][c]) == sum_k:
                    div_B_lambda = max(div_B_lambda, Div_m[i][j][c])
            if div_B_lambda > Lambda[i]:
                for j in range(len(B[i])):
                    if x.idx in B_m[i][j][c]:
                        B_m[i][j][c].remove(x.idx)
                    if x.idx in B_apo_m[i][j][c].keys():
                        B_apo_m[i][j][c].pop(x.idx)
                    A_m[i][j][c] = B_m[i][j][c].copy()
                    A_apo_m[i][j][c] = B_apo_m[i][j][c].copy()
                    B_m[i][j][c] = {x.idx}
                    B_apo_m[i][j][c] = {x.idx: x.idx}
                    Div_m[i][j][c] = sys.float_info.max
    t1 = time.perf_counter()
    #count
    stored_elements = set()
    for i in range(len(Lambda)):
        for j in range(len(Mu)):
            stored_elements.update(A[i][j])
            stored_elements.update(B[i][j])
            stored_elements.update(A_apo[i][j].values())
            stored_elements.update(B_apo[i][j].values())
            for c in range(m):
                stored_elements.update(A_m[i][j][c])
                stored_elements.update(B_m[i][j][c])
                stored_elements.update(A_apo_m[i][j][c].values())
                stored_elements.update(B_apo_m[i][j][c].values())
    num_elements = len(stored_elements)
    #  Post-processing
    sol, sol_div = None, 0.0
    w_start = max(0, len(X)-w+1)
    Sol = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    Divs = [[0 for j in range(len(Mu))] for i in range(len(Lambda))]
    Sol_m = [[[set() for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    Divs_m = [[[0 for c in range(m)] for j in range(len(Mu))] for i in range(len(Lambda))]
    Sol_fair = [[set() for j in range(len(Mu))] for i in range(len(Lambda))]
    for i in range(len(Lambda)):
        for j in range(len(Mu)):
            if len(A[i][j]) > 0 and min(A[i][j]) >= w_start:
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(A[i][j].union(B[i][j])), k=sum_k, dist=dist)
            elif min(B[i][j]) >= w_start:
                elem_A = B[i][j].copy()
                for value in A_apo[i][j].values():
                    if value >= w_start:
                        elem_A.add(value)
                Sol[i][j], Divs[i][j] = GMM(X, idxs=list(elem_A), k=sum_k, dist=dist)
            for c in range(m):
                if len(A_m[i][j][c]) > 0 and min(A_m[i][j][c]) >= w_start:
                    Sol_m[i][j][c], Divs_m[i][j][c] = GMM(X, idxs=list(A_m[i][j][c].union(B_m[i][j][c])), k=sum_k, dist=dist)
                elif min(B_m[i][j][c]) >= w_start:
                    elem_A = B_m[i][j][c].copy()
                    for value in A_apo_m[i][j][c].values():
                        if value >= w_start:
                            elem_A.add(value)
                    Sol_m[i][j][c], Divs_m[i][j][c] = GMM(X, idxs=list(elem_A), k=sum_k, dist=dist)
            hasValidSol = True
            for c in range(m):
                if len(Sol_m[i][j][c]) < k[c]:
                    hasValidSol = False
                    break
            if not hasValidSol:
                continue
            S_all = set()
            S_all.update(Sol[i][j])
            for c in range(m):
                S_all.update(Sol_m[i][j][c])
            G1 = nx.Graph()
            for idx1 in S_all:
                G1.add_node(idx1)
                for idx2 in S_all:
                    if idx1 < idx2 and dist(X[idx1], X[idx2]) < eps * Mu[j] / (m + 1):
                        G1.add_edge(idx1, idx2)
            P = []
            for p in nx.connected_components(G1):
                P.append(set(p))
            dict_par = dict()
            for z in range(len(P)):
                for s_idx in P[z]:
                    dict_par[s_idx] = z
            S_prime = set()
            num_elem_col = np.zeros(m)
            for c in range(m):
                sol_i_j_c = {x.idx for x in X if x.idx in Sol[i][j] and x.color == c}
                if len(sol_i_j_c) <= k[c]:
                    S_prime.update(sol_i_j_c)
                    num_elem_col[c] = len(sol_i_j_c)
                else:
                    for s_idx in sol_i_j_c:
                        S_prime.add(s_idx)
                        num_elem_col[c] += 1
                        if num_elem_col[c] == k[c]:
                            break
            X1 = set()
            X2 = set()
            P_prime = set()
            if len(S_prime) < sum_k:
                for s_idx in S_prime:
                    P_prime.add(dict_par[s_idx])
                for s_idx in S_all:
                    s_col = X[s_idx].color
                    s_par = dict_par[s_idx]
                    if s_idx not in S_prime and num_elem_col[s_col] < k[s_col]:
                        X1.add(s_idx)
                    if s_idx not in S_prime and s_par not in P_prime:
                        X2.add(s_idx)
                X12 = X1.intersection(X2)
                while len(X12) > 0:
                    max_idx = -1
                    max_div = 0.0
                    for s_idx1 in X12:
                        s_div1 = sys.float_info.max
                        for s_idx2 in S_prime:
                            s_div1 = min(s_div1, dist(X[s_idx1], X[s_idx2]))
                        if s_div1 > max_div:
                            max_idx = s_idx1
                            max_div = s_div1
                    max_col = X[max_idx].color
                    max_par = dict_par[max_idx]
                    S_prime.add(max_idx)
                    num_elem_col[max_col] += 1
                    # print(max_idx, max_col, max_par, S_prime, num_elem_col)
                    if num_elem_col[max_col] == k[max_col]:
                        for s_idx in Sol_m[i][j][max_col]:#group_ins[max_col][ins_id].idxs:
                            X1.discard(s_idx)
                    for s_idx in P[max_par]:
                        X2.discard(s_idx)
                    X12 = X1.intersection(X2)
            while len(S_prime) < sum_k and len(X1) > 0 and len(X2) > 0:
                GA = nx.DiGraph()
                GA.add_node(-1)
                GA.add_node(len(X))
                for s_idx in X1:
                    GA.add_node(s_idx)
                    GA.add_edge(-1, s_idx)
                for s_idx in X2:
                    GA.add_node(s_idx)
                    GA.add_edge(s_idx, len(X))
                for s_idx1 in S_prime:
                    GA.add_node(s_idx1)
                    for s_idx2 in X1:
                        if X[s_idx1].color == X[s_idx2].color:
                            GA.add_edge(s_idx1, s_idx2)
                        if dict_par[s_idx1] == dict_par[s_idx2]:
                            GA.add_edge(s_idx2, s_idx1)
                    for s_idx2 in X2:
                        if X[s_idx1].color == X[s_idx2].color:
                            GA.add_edge(s_idx1, s_idx2)
                        if dict_par[s_idx1] == dict_par[s_idx2]:
                            GA.add_edge(s_idx2, s_idx1)
                try:
                    s_path = nx.shortest_path(GA, source=-1, target=len(X))
                    for s_idx in s_path:
                        if -1 < s_idx < len(X):
                            if s_idx in S_prime:
                                S_prime.remove(s_idx)
                            else:
                                S_prime.add(s_idx)
                    if len(S_prime) == sum_k:
                        break
                    P_prime.clear()
                    X1.clear()
                    X2.clear()
                    for s_idx in S_prime:
                        P_prime.add(dict_par[s_idx])
                    for s_idx in S_all:
                        s_col = X[s_idx].color
                        s_par = dict_par[s_idx]
                        if s_idx not in S_prime and num_elem_col[s_col] < k[s_col]:
                            X1.add(s_idx)
                        if s_idx not in S_prime and s_par not in P_prime:
                            X2.add(s_idx)
                except nx.NetworkXNoPath:
                    break
            Sol_fair[i][j] = S_prime
            if len(S_prime) == sum_k:
                div_s = diversity(X, S_prime, dist)
                if div_s > sol_div:
                    sol = S_prime
                    sol_div = div_s
    t2 = time.perf_counter()
    print("Sol",Sol)
    print("Sol_fair",Sol_fair)
    #print([x.color for x in X if x.idx in sol])
    return sol, sol_div, (t1 - t0)/len(X), t2 - t1, num_elements