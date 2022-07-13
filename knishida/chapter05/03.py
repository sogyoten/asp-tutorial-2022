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


def circ_array(r, M, theta, f):
    a = np.empty(M, dtype=complex)
    p = np.zeros((M, 3))
    c = 334
    u = [np.sin(theta), np.cos(theta), 0]

    for m in range(M):
        p[m] = [
            r * np.sin(2 * np.pi / M * m),
            r * np.cos(2 * np.pi / M * m),
            0,
        ]
        a[m] = np.exp(1j * 2 * np.pi * f / c * np.dot(u, p[m].T))

    return a


def general_array(coords, theta, f):
    c = 334
    u = [np.sin(theta), np.cos(theta), 0]
    p = np.array(coords)
    M = len(p[:, 0])
    a = np.empty(M, dtype=complex)

    for m in range(M):
        a[m] = np.exp(1j * 2 * np.pi * f / c * np.dot(u, p[m].T))
    return a


if __name__ == "__main__":
    d = 0.05
    M = 3
    theta = 45 / 360 * 2 * np.pi
    f = 1000
    coords_lin = [[-0.05, 0, 0], [0, 0, 0], [0.05, 0, 0]]
    ans = lin_array(d, M, theta, f)
    ans_g = general_array(coords_lin, theta, f)
    print(f"lin:{ans}")
    print(f"gen:{ans_g}")

    coords_circ = [
        [d * np.sin(0), d * np.cos(0), 0],
        [d * np.sin(2 * np.pi / 3), d * np.cos(2 * np.pi / 3), 0],
        [d * np.sin(2 * np.pi * 2 / 3), d * np.cos(2 * np.pi * 2 / 3), 0],
    ]
    ans = circ_array(d, M, theta, f)
    ans_g = general_array(coords_circ, theta, f)
    print(f"circ:{ans}")
    print(f"gen :{ans_g}")
