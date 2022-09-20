import math
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append("../../")
from odeControl import *

from cartpoleOde import CartPoleODE 
from renderCartPole import renderCartPole




if __name__ == "__main__":

    DIM_STATE = 4
    DIM_CONTROL = 1
    CART_MASS = 1.2
    BALL_MASS  = 1
    LENGTH = 1.0
    T_MAX = 10.0
    NT = 2000

    np.random.seed(111)
    cartpole = CartPoleODE(CART_MASS, BALL_MASS, LENGTH)
    
    t_all = np.linspace(0, T_MAX, NT+1)
    u_all = np.zeros((NT+1, DIM_CONTROL))
    x_target = np.zeros(DIM_STATE)
    u0 = np.zeros(DIM_CONTROL)
    x0 = np.array([0.0, 0.0, 0.01, 0.0])

    cumulative = [CumulativeL2Misfit(x_target, 1.0), CumulativeL2Penalty(u0, 1.0)]
    terminal = []
    ode_model = ODEControlModel(cartpole, t_all, forwardEuler, cumulative, terminal)
    x_all = ode_model.solveFwd(x0, u_all)
    cost = ode_model.cost(x_all, u_all)

    renderCartPole(cartpole, x_all, t_all)


