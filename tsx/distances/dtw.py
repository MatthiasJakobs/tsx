import numpy as np

# Dynamic time warping (assume s and t to be numpy array)
# TODO: At the moment, s and t need to be 1d arrays
def dtw(s, t):
    m = np.array((len(s), len(t)))
    m[:, :] = 1e20
    m[0, 0] = 0

    for i in range(len(s)):
        for j in range(len(t)):
            cost = np.abs(s[i] - t[j])
            m[i, j] = cost + np.min(m[i-1, j], m[i, j-1], m[i-1, j-1])

    return m[-1, -1]


