import numpy as np 

class RegularizationIC:
    def cost(self, x0):
        raise NotImplementedError("Child class should implement cost")
    def grad(self, x0):
        raise NotImplementedError("Child class should implement grad")

class TikhnovRegularizationIC(RegularizationIC):
    def __init__(self, reg_coeff, mean=None):
        self.reg_coeff = reg_coeff
        self.mean = mean 
    def cost(self, x0):
        if self.mean is not None:
            return self.reg_coeff * np.linalg.norm(x0 - self.mean)**2 / 2
        else:
            return self.reg_coeff * np.linalg.norm(x0)**2 / 2

    def grad(self, x0):
        if self.mean is not None:
            return self.reg_coeff * (x0 - self.mean)
        else:
            return self.reg_coeff * x0