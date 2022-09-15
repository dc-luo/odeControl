import numpy as np
import math
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from adjoints import *


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
    BETA = 1.2
    GAMMA = 1
    T_MAX = 1.0
    NT = 10000

    np.random.seed(111)
    sir = SIRModel(BETA, GAMMA)
    t_all = np.linspace(0, T_MAX, NT+1)
    u_all = np.zeros((NT+1, DIM))
    x_target = np.zeros(DIM)
    u0 = np.zeros(DIM)


    x0 = np.array([0.9, 0.1, 0.0])
    x_all = forwardEuler(sir.forwardSystem, x0, u_all, t_all)

    cumulative = [CumulativeL2Misfit(x_target, 2.0), CumulativeL2Penalty(u0, 2.0)]
    terminal = [TerminalL2Misfit(x_target, 1.3)]
    ode_model = ODEControlModel(sir, t_all, forwardEuler, cumulative, terminal)
    x_all_model = ode_model.solveFwd(x0, u_all)
    p_all_model = ode_model.solveAdj(x_all, u_all)

    cost = ode_model.cost(x_all_model, u_all)
    # grad_control = ode_model.evalGradientControl(x_all_model, u_all, p_all_model, t_all)
    # grad_ic = ode_model.evalGradientInitialCondition(x_all_model, u_all, p_all_model, t_all)

    adj, fd = finiteDifferenceIC(ode_model, x0, u_all)

    du_all = np.random.randn(u_all.shape[0], u_all.shape[1])
    finiteDifferenceControl(ode_model, x0, u_all, du_all)

    print("Total cost: %g" %(cost))

    plt.figure()
    diff_all = np.linalg.norm(x_all - x_target, axis=1)**2
    plt.plot(t_all, diff_all, label="Squared misfit")


    plt.figure(figsize=(12,8))
    plt.subplot(311)
    plt.plot(t_all, x_all[:,0], '-', label="Solver")
    plt.plot(t_all, x_all_model[:,0], '--', label="With model")
    plt.plot(t_all, x_target[0]*np.ones(t_all.shape), '--', label="Target")
    plt.xlabel("t")
    plt.ylabel("s")

    plt.subplot(312)
    plt.plot(t_all, x_all[:,1], '-', label="Solver")
    plt.plot(t_all, x_all_model[:,1], '--', label="With model")
    plt.plot(t_all, x_target[1]*np.ones(t_all.shape), '--', label="Target")
    plt.xlabel("t")
    plt.ylabel("i")

    plt.subplot(313)
    plt.plot(t_all, x_all[:,2], '-', label="Solver")
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
    plt.show()
