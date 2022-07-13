import numpy as np


def lin_array(d, M, theta, f):
    a = np.empty(M, dtype=complex)
    p = np.zeros((M, 3))
    c = 334
    u = [np.sin(theta), np.cos(theta), 0]

    for m in range(M):
        p[m] = [(m - (M - 1) / 2) * d, 0, 0]
        a[m] = np.exp(1j * 2 * np.pi * f / c * np.dot(u, p[m].T))

    return a


if __name__ == "__main__":
    d = 0.05
    M = 3
    theta = 45 / 360 * 2 * np.pi
    f = 1000
    ans = lin_array(d, M, theta, f)
    print(ans)
