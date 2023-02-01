from hashlib import md5

class ROC_Member:

    def __init__(self, r, quantized_values=None, loss=None):
        self.r = r
        assert len(self.r.shape) == 1, 'Multivariate RoCs not implemented'
        self.quantized_values = quantized_values
        self.loss = loss

    def __repr__(self):
        if self.quantized_values is not None:
            return ''.join(str(v) for v in self.quantized_values)
        else:
            return ','.join(str(v.round(4)) for v in self.r)

    def __hash__(self):
        representation = self.__repr__()
        return int(md5(representation.encode('utf-8')).hexdigest(), 16) & 0xffffffff
