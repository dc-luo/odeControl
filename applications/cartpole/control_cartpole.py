import math
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append("../../")
from odeControl import *

from cartpoleOde import CartPoleODE 
from renderCartPole import renderCartPole


def finiteDifferenceControl(ode_model, x0, u_all, du_all):
    x_all = ode_model.solveFwd(x0, u_all)
    p_all = ode_model.solveAdj(x_all, u_all)
    c = ode_model.cost(x_all, u_all)
    adj_grad = ode_model.evalGradientControl(x_all, u_all, p_all)

    delta = 1e-4
    fd_der = np.zeros(u_all.shape)

    x_new_all = ode_model.solveFwd(x0, u_all + delta * du_all)
    c_new = ode_model.cost(x_new_all, u_all)
    fd_der = (c_new - c)/delta

    adj_der = np.sum(adj_grad * du_all)

    print("Adjoint control derivative: ")
    print(adj_der)

    print("Finite difference control derivative: ")
    print(fd_der)

    return adj_der, fd_der



class CumulativeStability(CumulativeCost):
    def __init__(self, theta_target, alpha=1.0):
        self.theta_target = theta_target
        self.alpha = alpha

    def eval(self, x, u, t):
        return self.alpha*np.linalg.norm(x[2] - self.theta_target)**2

    def jacobian_x(self, x, u, t):
        dQdx = np.zeros(4)
        dQdx[2] = 2*self.alpha*(x[2] - self.theta_target)
        return dQdx

    def jacobian_u(self, x, u, t):
        return np.zeros(u.shape)

    def eval_all(self, x_all, u_all, t_all):
        assert x_all.shape[0] == t_all.shape[0]
        diff_trajectory = self.alpha*np.linalg.norm(x_all[:, 2:3] - self.theta_target, axis=1)**2
        dts = t_all[1:] - t_all[:-1]
        # print(diff_trajectory)

        # Integration by trapezoidal rule
        integral = np.inner(diff_trajectory[:-1], 0.5*dts)
        integral += np.inner(diff_trajectory[1:], 0.5*dts)

        return integral

class TerminalStability(TerminalCost):
    def __init__(self, theta_target, alpha=1.0):
        self.theta_target = theta_target
        self.alpha = alpha

    def eval(self, x, u):
        return self.alpha*np.linalg.norm(x[2] - self.theta_target)**2

    def jacobian_x(self, x, u):
        dxQ = np.zeros(4)
        dxQ[2] = 2*self.alpha*(x[2] - self.theta_target)
        return dxQ

    def jacobian_u(self, x, u):
        return np.zeros(u.shape)


if __name__ == "__main__":

    DIM_STATE = 4
    DIM_CONTROL = 1
    CART_MASS = 1.0
    BALL_MASS  = 0.1
    LENGTH = 1.0
    T_MAX = 10.0
    NT = 1000

    MODE = "swingup"
    # MODE = "stabilize"

    settings = steepestDescentSettings()
    settings["max_iter"] = 20000
    settings["step_size"] = 1e-0
    settings["grad_tol"] = 1e-6
    settings["verbose"] = False


    np.random.seed(111)
    cartpole = CartPoleODE(CART_MASS, BALL_MASS, LENGTH)

    t_all = np.linspace(0, T_MAX, NT+1)
    if MODE == "swingup":
        x0 = np.array([0.0, 0.0, np.pi, 0.0])
    elif MODE == "stabilize":
        x0 = np.array([0.0, 0.0, 0.01, 0.0])

    N_WINDOW = 1
    # cumulative = [CumulativeStability(0.0, 1.0), CumulativeL2Penalty(np.array([0.0]), 1e-6)]
    cumulative = [CumulativeStability(0.0, 1.0)]
    # cumulative = [CumulativeL2Misfit(x0, 1.0)]
    cumulative = [] 
    # terminal = [TerminalStability(0.0, 1.0)]
    terminal = [TerminalL2Misfit(np.zeros(4), 1e-1)]
    # terminal = [] 
    ode_model = ODEControlModel(cartpole, t_all, forwardEuler, cumulative, terminal)
    
    u_all = np.zeros((NT+1, DIM_CONTROL))
    du_all = np.ones(u_all.shape)
    finiteDifferenceControl(ode_model, x0, u_all, du_all)

    u_guess = u_all
    for i in range(N_WINDOW):
        u_opt = steepestDescent(ode_model, x0, u_guess, settings)
        x_opt = ode_model.solveFwd(x0, u_opt)

        if i == 0:
            x_cumulative = x_opt 
            u_cumulative = u_opt
            t_cumulative = t_all
        else:
            x_cumulative = np.append(x_cumulative, x_opt, axis=0)
            u_cumulative = np.append(u_cumulative, u_opt, axis=0)
            t_cumulative = np.append(t_cumulative, t_all + t_cumulative[-1])

        # plt.figure()
        # plt.plot(t_all, u_opt[:])
        # 
        # renderCartPole(cartpole, x_opt, t_all)
        # plt.show()
        x0 = x_opt[-1]
        # u_guess = u_opt

    plt.figure()
    plt.plot(t_cumulative, u_cumulative[:])

    ani = renderCartPole(cartpole, x_cumulative, t_cumulative)
    # writervideo = animation.FFMpegWriter(fps=60)
    ani.save('%s.gif' %(MODE), fps=100)
    plt.show()






