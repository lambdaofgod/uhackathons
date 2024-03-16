import numba
import numpy as np
import abc


@numba.jit(nopython=True)
def are_unique(args):
    n_args = len(args)
    n_unique = len(np.unique(np.array(args)))
    return n_args == n_unique


@numba.jit(nopython=True)
def _get_hyp(D, a, b, c, d):
    if not are_unique([a, b, c, d]):
        return 0
    S1 = (D[a, b] + D[d, c])
    S2 = (D[a, c] + D[b, d])
    S3 = (D[a, d] + D[b, c])
    S = np.array([S1, S2, S3])
    S.sort()
    M1 = S[-1]
    M2 = S[-2]
    return M1 - M2


@numba.jit(nopython=True)
def _get_tau(D, a, b, c, d):
    S1 = (D[a, b] + D[c, d])
    S2 = (D[a, c] + D[b, d])
    S3 = (D[a, d] + D[b, c])
    return (S1 - max(S2, S3))


@numba.jit(nopython=True)
def get_delta_hyperbolicity_ccl(graph_dists, sorted_indices, sorted_dists, debug=False):
    if debug:
        print("ccl")
    n_vertices = graph_dists.shape[0]
    max_hyp = 0
    for i_idx in range(0, n_vertices**2):
        # if i_idx % 2 == 1: continue
        i = sorted_indices[i_idx]
        a = (i // n_vertices)
        b = i % n_vertices
        for j_idx in range(i):
            # if j_idx % 2 == 1: continue
            j = sorted_indices[j_idx]
            c = (j // n_vertices)
            d = j % n_vertices

            if not are_unique([a, b, c, d]):
                continue

            hyp = _get_tau(graph_dists, a, b, c, d)
            if hyp > max_hyp:
                if debug:
                    print("new")
                    print([a, b, c, d])
                    print(hyp)
                max_hyp = hyp

            if graph_dists[a, b] <= max_hyp:
                if debug:
                    print("breaking")
                    print("i", i)
                    print("d(", a, b, ") =", graph_dists[a, b])
                    print([a, b, c, d])
                return max_hyp / 2

    return max_hyp / 2


@numba.jit(nopython=True)
def get_delta_hyperbolicity_naive(graph_dists, debug=False):
    if debug:
        print("naive")
    n_nodes = graph_dists.shape[0]

    max_hyp = 0
    idxs = None
    for a in range(n_nodes):
        for b in range(a):
            for c in range(b):
                for d in range(c):
                    hyp = _get_hyp(graph_dists, a, b, c, d)
                    if hyp > max_hyp:
                        max_hyp = hyp
                        if debug:
                            print("max at idxs", [a, b, c, d])
    return max_hyp / 2
