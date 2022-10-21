import numpy as np
import math
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from odeControl import *

from lorenzODE import Lorenz63ODE

def finiteDifferenceIC(ode_model, x0, u_all):
    x_all = ode_model.solveFwd(x0, u_all)
    p_all = ode_model.solveAdj(x_all, u_all)
    c = ode_model.cost(x_all, u_all)
    adj_grad = ode_model.evalGradientInitialCondition(x_all, u_all, p_all)

    delta = 1e-3
    dim = x0.shape[0]

    fd_grad = np.zeros(x0.shape)

    for i in range(dim):
        dx0 = np.zeros(x0.shape)
        dx0[i] = 1.0
        x_new_all = ode_model.solveFwd(x0 + delta * dx0, u_all)
        c_new = ode_model.cost(x_new_all, u_all)
        fd_grad[i] = (c_new - c)/delta


    print("Adjoint gradient: ")
    print(adj_grad)

    print("Finite difference gradient: ")
    print(fd_grad)

    return adj_grad, fd_grad


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



if __name__ == "__main__":
    DIM = 3

    lorenz_system = Lorenz63ODE(10.0, 28.0, 8.0/3.0)
    T_MAX = 10
    NT = 20000

    np.random.seed(111)
    t_all = np.linspace(0, T_MAX, NT+1)
    u_all = np.zeros((NT+1, DIM))
    x_target = np.zeros(DIM)
    u0 = np.zeros(DIM)
    x0 = np.array([0.9, 0.1, 0.0])
    cumulative = [CumulativeL2Misfit(x_target, 0.5)]
    terminal = []
    ode_model = ODEControlModel(lorenz_system, t_all, forwardEuler, cumulative, terminal)
    x_all_model = ode_model.solveFwd(x0, u_all)
    p_all_model = ode_model.solveAdj(x_all_model, u_all)

    cost = ode_model.cost(x_all_model, u_all)
    adj, fd = finiteDifferenceIC(ode_model, x0, u_all)

    print("Total cost: %g" %(cost))

    plt.figure()
    diff_all = np.linalg.norm(x_all_model - x_target, axis=1)**2
    plt.plot(t_all, diff_all, label="Squared misfit")


    plt.figure(figsize=(12,8))
    plt.subplot(311)
    # plt.plot(t_all, x_all[:,0], '-', label="Solver")
    plt.plot(t_all, x_all_model[:,0], '--', label="With model")
    plt.plot(t_all, x_target[0]*np.ones(t_all.shape), '--', label="Target")
    plt.xlabel("t")
    plt.ylabel("s")

    plt.subplot(312)
    # plt.plot(t_all, x_all[:,1], '-', label="Solver")
    plt.plot(t_all, x_all_model[:,1], '--', label="With model")
    plt.plot(t_all, x_target[1]*np.ones(t_all.shape), '--', label="Target")
    plt.xlabel("t")
    plt.ylabel("i")

    plt.subplot(313)
    # plt.plot(t_all, x_all[:,2], '-', label="Solver")
    plt.plot(t_all, x_all_model[:,2], '--', label="With model")
    plt.plot(t_all, x_target[2]*np.ones(t_all.shape), '--', label="Target")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.tight_layout()
    plt.legend()

    plt.figure(figsize=(12,8))
    plt.subplot(311)
    plt.plot(t_all, p_all_model[:,0], '--', label="Model adjoint")
    plt.xlabel("t")
    plt.ylabel("s")

    plt.subplot(312)
    plt.plot(t_all, p_all_model[:,1], '--', label="Model adjoint")
    plt.xlabel("t")
    plt.ylabel("i")

    plt.subplot(313)
    plt.plot(t_all, p_all_model[:,2], '--', label="Model adjoint")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.tight_layout()
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_all_model[:,0], x_all_model[:,1], x_all_model[:,2], label="Forward Euler")
    plt.title("Forward solution")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p_all_model[:,0], p_all_model[:,1], p_all_model[:,2], label="Forward Euler")
    plt.title("Adjoint solution")

    plt.show()
