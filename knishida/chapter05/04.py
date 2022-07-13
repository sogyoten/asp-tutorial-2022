import numpy as np


def spacial_corr_matrix(X):
    X = np.array(X)
    [M, F, T] = X.shape
    x = np.empty((F, T, M), dtype=complex)
    for t in range(T):
        for f in range(F):
            for m in range(M):
                x[f, t, m] = X[m, f, t]

    R = np.empty((F, M, M), dtype=complex)
    for f in range(F):
        tmpR = np.empty((M, M), dtype=complex)
        for t in range(T):
            x_ = np.conjugate(x[f, t].T)
            tmpR += x[f, t].T @ x_.T
        R[f] = tmpR / T

    return R


if __name__ == "__main__":
    X = [
        [[1, -1j, -1, 1j], [2, -2j, -2, 2j], [3, -3j, -3, 3j]],
        [[4, -2j, 1, 0], [2, -1j, 0, 0], [1, -1j, 1, 0]],
    ]

    ans = spacial_corr_matrix(X)
    print(ans)
