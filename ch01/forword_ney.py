import numpy as numpy

# p15
class Sigmoid:
    def __init__(self):
        self.params = []
    def forword(self, x):
        return 1 / 1 + np.exp(-x)

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    def forword(self, x):
        W, b = self.params
        return np.dot(x, W) + b
