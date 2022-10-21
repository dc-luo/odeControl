import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.interpolate
sys.path.append("../../")
from odeControl import *
from lorenzODE import Lorenz63ODE
from stableODE import StabilizedAdjointControlProblem, steepestDescentStabilized

class DataMisfit(CumulativeCost):
    def __init__(self, x_data, t_data, alpha=1.0):
        self.x_data = x_data
        self.t_data = t_data
        self.alpha = alpha
        self.xt = scipy.interpolate.interp1d(t_data, x_data, axis=0, kind='linear', fill_value="extrapolate")

    def eval(self, x, u, t):
        if self.W is None:
            return self.alpha*np.linalg.norm(x - self.xt(t))**2
        else:
            raise NotImplementedError("Currently not implemented with nonzero weight")

    def jacobian_x(self, x, u, t):
        return 2*self.alpha*(x - self.xt(t))

    def jacobian_u(self, x, u, t):
        return np.zeros(u.shape)

    def eval_all(self, x_all, u_all, t_all):
        # assert x_all.shape[0] == t_all.shape[0]
        # assert x_all.shape[1] == self.x_target.shape[0]
        diff_trajectory = self.alpha*np.linalg.norm(x_all - self.xt(t_all), axis=1)**2
        dts = t_all[1:] - t_all[:-1]

        # Integration by trapezoidal rule
        integral = np.inner(diff_trajectory[:-1], 0.5*dts)
        integral += np.inner(diff_trajectory[1:], 0.5*dts)

        return integral



def compareModels(ode_model, stabilized_model, u_all, x0, p0):
    x_all = ode_model.solveFwd(x0, u_all)
    p_all = ode_model.solveAdj(x_all, u_all)
    gm = ode_model.evalGradientInitialCondition(x_all, u_all, p_all)

    x_all_stab = stabilized_model.solveFwd(x0, u_all)

    learning_rate = 1e-3

    p_now = p_all[0]
    print("True adjoint initial condition: ", p_all[0])
    print("Guess adjoint initial condition: ", p_now)

    for i in range(100):
        p_all_stab = stabilized_model.solveAdjFwd(p_now, x_all_stab, u_all)
        q_all_stab = stabilized_model.solveAdjAdj(x_all_stab, u_all, p_all_stab)
        gs_stab = stabilized_model.solveAdjGradient(x_all_stab, u_all, p_all_stab, q_all_stab, x0, p_now)
        p_now -= learning_rate*gs_stab
        print("Current: ", p_now)

    print(gm)
    print("Graident of that: ", gs_stab)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_all[:, 0], x_all[:, 1], x_all[:,2], label="ODE")
    ax.plot(x_all_stab[:, 0], x_all_stab[:, 1], x_all_stab[:,2], label="Stabilized")
    plt.title("x")
    plt.legend()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p_all[:, 0], p_all[:, 1], p_all[:,2], label="ODE")
    # ax.plot(p_all_stab[:, 0], p_all_stab[:, 1], p_all_stab[:,2], label="Stabilized")
    plt.title("p")
    plt.legend()

    plt.figure()
    for i in range(3):
        plt.subplot(131+i)
        plt.plot(p_all_stab[:, i], label="Stabilized")
        plt.title("p"+str(i+1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(q_all_stab[:, 0], q_all_stab[:, 1], q_all_stab[:,2], label="Stabilized")
    plt.title("q")

    plt.show()


if __name__ == "__main__":
    DIM = 3
    T_MAX = 1
    NT = 1000

    RUN_OPTIMIZATION = False

    lorenz_system = Lorenz63ODE(10.0, 28.0, 8.0/3.0)
    np.random.seed(111)
    t_all = np.linspace(0, T_MAX, NT+1)
    u_all = np.zeros((NT+1, DIM))
    u0 = np.zeros(DIM)
    x0 = np.array([0.9, 0.1, 0.0])

    x_target = np.array([0.0, 0.0, 0.0])
    cumulative = [CumulativeL2Misfit(x_target, 0.5)]
    ode_data = ODEControlModel(lorenz_system, t_all, forwardEuler, cumulative, [])
    x_data = ode_data.solveFwd(x0, u_all)
    x_data += 0.01 * np.random.randn(x_data.shape[0], x_data.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_data[:, 0], x_data[:, 1], x_data[:, 2], label="Data")

    regularization = TikhnovRegularizationIC(0.0, x0)
    dataMisfit = [DataMisfit(x_data, t_all, 0.5)]
    ode_model = ODEControlModel(lorenz_system, t_all, forwardEuler, dataMisfit, [])

    beta = 0
    stable_adj = StabilizedAdjointControlProblem(lorenz_system, t_all, forwardEuler, beta, regularization, cumulative_costs=dataMisfit)
    compareModels(ode_model, stable_adj, u_all, x0, np.zeros(DIM))
    
    if RUN_OPTIMIZATION:
        x0_guess = np.random.randn(3) + x0
        x_guess = ode_model.solveFwd(x0_guess, u_all)
        ax.plot(x_guess[:, 0], x_guess[:, 1], x_guess[:, 2], label="Initial")
        ax.plot(x_guess[0:1, 0], x_guess[0:1, 1], x_guess[0:1, 2], 'ko')

        settings = steepestDescentSettings()
        settings["step_size"] = 0.1
        settings["max_iter"] = 100
        settings["grad_tol"] = 1e-6
        x0_opt = steepestDescentInitialCondition(ode_model, x0_guess, u_all, settings, regularization)
        x_opt = ode_model.solveFwd(x0_opt, u_all)
        ax.plot(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2], label="Optimal")
        ax.plot(x_opt[0:1, 0], x_opt[0:1, 1], x_opt[0:1, 2], 'ko')
        plt.legend()
        plt.show()

        settings["step_size"] = 0.1
        settings["max_iter"] = 400
        s0_guess = regularization.grad(x0_guess)
        
        x0_joint, p0_joint = steepestDescentStabilized(stable_adj, x0_guess, s0_guess, u_all, settings)
        x_joint = ode_model.solveFwd(x0_joint, u_all)
        ax.plot(x_joint[:, 0], x_joint[:, 1], x_joint[:, 2], label="Joint")
        ax.plot(x_joint[0:1, 0], x_joint[0:1, 1], x_joint[0:1, 2], 'ko')
