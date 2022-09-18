import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from odeControl import *

def plotSolution(x_all, u_all, t_all, label):
    plt.figure(figsize=(12,8))
    for i in range(3):
        plt.subplot(311+i)
        plt.plot(t_all, x_all[:,i])
        plt.title("%s state %d" %(label, i))
        plt.ylim([0.0, 1.0])
    plt.tight_layout()

    plt.figure(figsize=(12,8))
    for i in range(3):
        plt.subplot(311+i)
        plt.plot(t_all, u_all[:,i])
        plt.title("%s control %d" %(label, i))
    plt.tight_layout()


if __name__ == "__main__":
    DIM = 3
    BETA = 2.0
    GAMMA = 1
    T_MAX = 10.0
    NT = 1000

    np.random.seed(111)
    
    use_jax = False

    if use_jax:
        print("Using jax version of model")
        sir = SIRModelJax(BETA, GAMMA)
    else:
        print("Using original version of model")
        sir = SIRModel(BETA, GAMMA)

    x0 = np.array([0.8, 0.2, 0.0])
    t_all = np.linspace(0, T_MAX, NT+1)
    u_all = np.zeros((NT+1, DIM))
    x_target = np.array([0.5, 0.25, 0.25])
    u0 = np.zeros(DIM)
    cumulative = [CumulativeL2Misfit(x_target, 2.0), CumulativeL2Penalty(u0, 1e-3)]
    terminal = []
    ode_model = ODEControlModel(sir, t_all, forwardEuler, cumulative, terminal)
    
    # solve initial
    x_all = ode_model.solveFwd(x0, u_all)
    plotSolution(x_all, u_all, t_all, "initial")
        
    settings = steepestDescentSettings()
    settings["step_size"] = 0.1
    settings["max_iter"] = 1000
    settings["grad_tol"] = 1e-6
    u_opt = steepestDescent(ode_model, x0, u_all, settings)
    x_all = ode_model.solveFwd(x0, u_opt)


    plotSolution(x_all, u_all, t_all, "final")

    plt.show() 
