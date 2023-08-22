from hashlib import md5

class ROC_Member:

    def __init__(self, x, y, indices, loss=None):
        self.x = x
        self.y = y
        self.r = x[indices]
        self.indices = indices
        self.loss = loss

    def __repr__(self):
        return ', '.join(str(v.round(4)) for v in self.r)

    def __hash__(self):
        representation = self.__repr__()
        return int(md5(representation.encode('utf-8')).hexdigest(), 16) & 0xffffffff

