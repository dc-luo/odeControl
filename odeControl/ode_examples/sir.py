import time
import math
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp 
from ..ode import ODESystem, forwardEuler, ODESystemJax

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
        # t0 = time.time()
        rhs = self.jacobian_x(x, u, t).T @ p
        # t1 = time.time()
        # print("Time to compute adjoint system: ", t1-t0)
        return rhs


    def jacobian_u(self, x, u, t):
        return np.eye(self.dim)

class SIRModelJax(ODESystemJax):
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
        super().__init__(self.forwardSystemForm)

    def forwardSystemForm(self, x, u, t):
        ds = -self.beta * x[0] * x[1] + u[0]
        di = self.beta * x[0] * x[1] - self.gamma * x[1] + u[1]
        dr = self.gamma * x[1] + u[2]
        return jnp.array([ds, di, dr])

