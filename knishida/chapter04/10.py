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


def conv_to_real(rslt, fs, L, S):

    fn = fs // 2
    f_num = rslt.shape[0]
    F_r = np.arange(0, fn, fn // f_num)

    T_r = np.arange(rslt.shape[1])

    return F_r, T_r


if __name__ == "__main__":
    A = 1
    f = 440
    fs = 16000
    sec = 0.1

    t = np.arange(0, sec, 1 / fs)
    y = A * np.sin(2 * np.pi * f * t)

    L = 1000
    w = Hamming(L)
    S = 500

    ans = STFT(L, S, w, y)
    F_r, T_r = conv_to_real(ans, fs, L, S)

    # X, Y = np.mgrid[: ans.shape[1] + 1, : ans.shape[0] + 1]
    X, Y = np.mgrid[T_r, F_r]

    fig, ax = plt.subplots(figsize=(5, 5))
    spec = ax.pcolormesh(X, Y, 20 * np.log10(np.abs(ans)).T)

    fig.colorbar(spec, ax=ax, orientation="vertical")
    plt.show()
