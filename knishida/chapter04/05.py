import numpy as np


def synt_win(S, w):
    L = len(w)
    Q = int(np.floor(L / S))
    ws = np.zeros(L)
    for k in range(L):
        i = k - (Q - 1) * S
        ws[k] = w[k] / np.sum(w[i : k + (Q - 1) * S] ** 2)
    return ws
