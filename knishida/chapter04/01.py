import numpy as np


def zero_pad(L, S, x):
    a = np.pad(x, ((L - S, L - S)))
    add = (S - (len(a) % S)) % S
    a = np.pad(a, ((0, add)))
    return a


if __name__ == "__main__":
    x = [1, 2, 3, 4]
    L = 9
    S = 3
    print(zero_pad(L, S, x))
