import math
import jax 
import jax.numpy as jnp
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from odeControl import ODESystemJax

class Lorenz63ODE(ODESystemJax):
    def __init__(self, sigma, rho, beta):
        self.dim = 3
        self.sigma = sigma 
        self.rho = rho 
        self.beta = beta 
        super().__init__(self.forwardSystemForm)

    def forwardSystemForm(self, x, u, t):
        rhs_vector = jnp.array([self.sigma * (x[1] - x[0]),
                                x[0] * (self.rho - x[2]) - x[1],
                                x[0] * x[1] - self.beta * x[2]])
        return rhs_vector

