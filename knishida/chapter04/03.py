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


def STFT(L, S, w, x):
    a = zero_pad(L, S, x)
    T = (len(a) - (L - S)) // S
    b = frame_divide(L, S, x)
    c = np.zeros((T, L // 2 + 1), dtype=complex)
    for t in range(T):
        tmp = b[t] * w
        c[t] = np.fft.rfft(tmp)
    return c


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    L = 4
    S = 2
    w = [0, 1, 1, 0]
    print(STFT(L, S, w, x))
