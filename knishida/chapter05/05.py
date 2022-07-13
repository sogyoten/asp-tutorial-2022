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
    fs = 16000
    sec = 5

    n1 = np.random.rand(round(fs * sec))
    n2 = np.random.rand(round(fs * sec))
