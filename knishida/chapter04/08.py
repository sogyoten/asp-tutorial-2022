import numpy as np
import matplotlib.pyplot as plt


def zero_pad(L, S, x):
    a = np.pad(x, ((L - S, L - S)))
    add = (S - (len(a) % S)) % S
    a = np.pad(a, ((0, add)))
    return a


def frame_divide(L, S, x):
    a = zero_pad(L, S, x)
    T = (len(a) - L) // S + 1
    b = np.zeros((T, L))
    for t in range(T):
        b[t] = a[t * S : t * S + L]
    return b


def STFT(L, S, w, x):
    b = frame_divide(L, S, x)
    T = b.shape[0]
    c = np.zeros((L // 2 + 1, T), dtype=np.complex128)
    for t in range(T):
        c[:, t] = np.fft.rfft(w * b[t, :])
    return c


def Hamming(N):
    n = np.arange(N)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    return win


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


def istft1(S, X, w):
    F = X.shape[0]
    T = X.shape[1]

    N = 2 * (F - 1)
    M = S * (T - 1) + N

    win = synt_win(S, w)

    x_ = np.zeros(M)
    z = np.zeros((T, N))
    for t in range(T):
        z[t] = np.fft.irfft(X[:, t])
        x_[t * S : t * S + N] = x_[t * S : t * S + N] + z[t]

    return x_


if __name__ == "__main__":
    A = 1
    f = 440
    fs = 16000
    sec = 0.1

    t = np.arange(sec * fs) / fs
    y = A * np.sin(2 * np.pi * f * t)

    L = 1000
    w = Hamming(L)
    S = 500

    ans = STFT(L, S, w, y)

    ans_ = istft(S, ans, w)
    ans1_ = istft1(S, ans, w)

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot(ans_)
    ax1.set_title("default")
    ax2.plot(ans1_)
    ax2.set_title("ones window")
    plt.tight_layout()
    plt.show()
