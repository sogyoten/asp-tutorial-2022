import numpy as np


def zero_pad(L, S, x):
    a = np.pad(x, ((L - S, L - S)))
    add = (S - (len(a) % S)) % S
    a = np.pad(a, ((0, add)))
    return a


def frame_divide(L, S, x):
    a = zero_pad(L, S, x)
    T = (len(a) - (L - S)) // S
    b = np.zeros((T, L))
    for t in range(T):
        for l in range(L):
            b[t][l] = a[t * S + l]
    return b


if __name__ == "__main__":
    x = [1, 2, 3, 4]
    L = 8
    S = 4
    print(frame_divide(L, S, x))
