import numpy as np
from hashlib import md5

from tsx.distances import dtw

class ROC_Member:

    ''' Object representing a member of a Region of Competence

    Args:
        x (`np.ndarray`): Original time series values
        y (`np.ndarray`): Corresponding true forecasting values
        indices (`np.ndarray`): Indices indicating the salient region
        squared_error (`float`): Squared error achieved by the model

    Attributes:
        r (`np.ndarray`): Most salient subseries of `x`
        x (`np.ndarray`): Original time series values
        y (`np.ndarray`): Corresponding true forecasting values
        indices (`np.ndarray`): Indices indicating the salient region

    '''
    def __init__(self, x, y, indices, squared_error):
        self.x = x
        self.y = y
        self.r = x[indices]
        self.indices = indices
        self.squared_error = squared_error

    def __repr__(self):
        return ', '.join(str(v.round(4)) for v in self.r)

    def __hash__(self):
        representation = self.__repr__()
        return int(md5(representation.encode('utf-8')).hexdigest(), 16) & 0xffffffff

    def euclidean_distance(self, x):
        s_r = self.r.reshape(1, -1)
        if len(x.shape) <= 1:
            x = x.reshape(1, -1)
        _x = x[:, self.indices]
        dists = np.sum((_x - s_r)**2, axis=1)
        return dists.squeeze()

    def dtw_distance(self, x):
        if len(x.shape) <= 1:
            x = x.reshape(1, -1)
        _x = x[:, self.indices]
        dists = np.array([dtw(__x, self.r) for __x in _x])
        return dists.squeeze()

