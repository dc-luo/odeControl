import math
import numpy as np
import matplotlib.pyplot as plt
from ..ode import ODESystem, forwardEuler

class SIRModel(ODESystem):
    def __init__(self, beta, gamma):
        """
        SIR model for epidemics
        x[0] = s
        x[1] = i
        x[2] = r
        """
        self.beta = beta
        self.gamma = gamma
        self.dim = 3

    def forwardSystem(self, x, u, t):
        ds = -self.beta * x[0] * x[1] + u[0]
        di = self.beta * x[0] * x[1] - self.gamma * x[1] + u[1]
        dr = self.gamma * x[1] + u[2]
        return np.array([ds, di, dr])

    def jacobian_x(self, x, u, t):
        J = np.array([[-self.beta*x[1], -self.beta*x[0], 0],
                      [self.beta*x[1], self.beta*x[0] - self.gamma, 0],
                      [0, self.gamma, 0]])
        return J

    def adjointSystem(self, p, x, u, t):
        return self.jacobian_x(x, u, t).T @ p


    def jacobian_u(self, x, u, t):
        return np.eye(self.dim)
