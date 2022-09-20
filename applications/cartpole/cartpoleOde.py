import math
import jax 
import jax.numpy as jnp
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from odeControl import ODESystemJax

class CartPoleODE(ODESystemJax):
    def __init__(self, m_cart, m_ball, L):
        self.m_cart = m_cart
        self.m_ball = m_ball
        self.L = L 
        self.g = 9.81
        self.dim = 4
        super().__init__(self.forwardSystemForm)

    def forwardSystemForm(self, x, u, t):
        lhs_matrix = jnp.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, self.m_cart + self.m_ball, 0, -self.m_ball*self.L*jnp.cos(x[2])],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -self.m_ball * self.L * jnp.cos(x[2]), 0.0, self.m_ball*self.L]
                ])

        rhs_vector = jnp.array([x[1], 
                u[0] - self.m_ball * self.L * jnp.sin(x[2])*x[1]**2, 
                x[3], 
                self.m_ball * self.L * self.g * jnp.sin(x[2])
                ])
        return jnp.linalg.solve(lhs_matrix, rhs_vector)


    def cartesianCoordinates(self, x):
        """
        Turns the solution into cartesian coordinates of the cart and the 
        pendulum 
        """
        x_cart = x[0] 
        theta = x[2]
        rel_loc = self.L * np.array([-np.sin(theta), np.cos(theta)])
        loc_cart = np.array([x_cart, 0.0])
        loc_pendulum = loc_cart + rel_loc
        return loc_cart, loc_pendulum


