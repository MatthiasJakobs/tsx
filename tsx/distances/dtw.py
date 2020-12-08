import numpy as np

# Dynamic time warping (assume s and t to be numpy array)
# assume: s, t are (at most) 2d arrays (batch_size, nr_values)
def dtw(s, t):
    if s.ndim == 1 and t.ndim == 1:
        s = np.expand_dims(s, 0)
        t = np.expand_dims(t, 0)

    if len(s) < len(t):
        s = np.tile(s, (len(t), *np.ones_like(s.shape[1:]))) # tile needs information on how often to replicate each dimension
    if len(s) > len(t):
        t = np.tile(t, (len(s), *np.ones_like(t.shape[1:]))) # tile needs information on how often to replicate each dimension

    distances = []
    for i in range(len(s)):
        distances.append(dtw_single(s[i], t[i]))

    if len(distances) == 1:
        return distances[0]
    return np.array(distances)
        

def dtw_single(s, t):
    m = np.zeros((len(s), len(t)))
    m[:, :] = 1e20
    m[0, 0] = 0

    for i in range(1, len(s)):
        for j in range(1, len(t)):
            cost = np.abs(s[i] - t[j])
            m[i, j] = cost + np.min([m[i-1, j], m[i, j-1], m[i-1, j-1]])

    return m[-1, -1]
