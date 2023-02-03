#import torch
import numpy as np
#from torch.utils.data import Dataset

# Helper functions


class piecewiese_sinusoidal():

    def __init__(self, t0=96, n=1000, return_numpy=False, return_torch=False, random_state=0):
        self.t0 = t0
        self.n = n
        self.return_numpy = return_numpy
        self.return_torch = return_torch
        self.rng = self.to_random_state(random_state)

        #timepoints
        self.x = np.vstack(self.n*[np.expand_dims(np.arange(0,self.t0+24).astype(np.float),0)])

        #sinusoidal signal
        part1,part2,part3 = 60 * self.rng.rand(3,n)
        part4 = np.maximum(part1, part2)
        self.fx = np.hstack([
            np.expand_dims(part1,1)*np.sin(np.pi*self.x[0,0:12]/6)+72,
            np.expand_dims(part2,1)*np.sin(np.pi*self.x[0,12:24]/6)+72,
            np.expand_dims(part3,1)*np.sin(np.pi*self.x[0,24:t0]/6)+72,
            np.expand_dims(part4,1)*np.sin(np.pi*self.x[0,t0:t0+24]/6)+72
        ])

        self.fx = self.fx + self.rng.normal()

        self.masks = self._generate_square_subsequent_mask()

    def to_random_state(rs):
        if not isinstance(rs, np.random.RandomState):
            rs = np.random.RandomState(rs)
        return rs

    def load_piecewise_sinusoidal(self, index):
        if self.return_numpy:
            return self.x,self.fx,self.masks
        if self.return_torch:
            return 0 #TODO
        #TODO Exception werfen

    def _generate_square_subsequent_mask(self):
        mask = np.zeros((self.t0+24,self.t0+24))
        for i in range(0,self.t0):
            mask[i,self.t0:] = 1 
        for i in range(self.t0,self.t0+24):
            mask[i,i+1:] = 1
        mask = np.ma.filled(mask.astype(np.float), np.float(-np.inf)) #TODO funktioniert das so???
        return mask