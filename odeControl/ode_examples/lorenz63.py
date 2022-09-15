import numpy as np
from ..ode import ODESystem

class Lorenz63(ODESystem):
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3):
        """
        Lorenz 63 model with default parameters leading to chaos
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dim = 3

    def forwardSystem(self, x, u, t):
        dx0 = self.sigma * (x[1] - x[0])
        dx1 = x[0] * (self.rho - x[2]) - x[1]
        dx2 = x[0] * x[1] - self.beta * x[2]
        return np.array([dx0, dx1, dx2])

    def jacobian_x(self, x, u, t):
        J = np.array([[-self.sigma, self.sigma, 0],
                      [self.rho - x[2], -1, -x[0]],
                      [x[1], x[0], -self.beta]])
        return J

    def adjointSystem(self, p, x, u, t):
        return self.jacobian_x(x, u, t).T @ p


    def jacobian_u(self, x, u, t):
        return np.zeros((self.dim, u.shape[0]))
