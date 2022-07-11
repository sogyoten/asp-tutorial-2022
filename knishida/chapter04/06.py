import numpy as np


def synt_win(S, w):
    L = len(w)
    Q = int(np.floor(L / S))
    ws = np.zeros(L)
    for k in range(L):
        i = k - (Q - 1) * S
        ws[k] = w[k] / np.sum(w[i : k + (Q - 1) * S] ** 2)
    return ws


def istft(S, X, w):
    F = X.shape[0]
    T = X.shape[1]

    N = 2 * (F - 1)
    M = S * (T - 1) + N

    win = synt_win(S, w)

    x_ = np.zeros(M)
    z = np.zeros((T, N))
    for t in range(T):
        z[t] = np.fft.irfft(X[:, t])
        x_[t * S : t * S + N] = x_[t * S : t * S + N] + win * z[t]

    return x_
