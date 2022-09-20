import numpy as np


def steepestDescentSettings():
    settings = dict()
    settings["max_iter"] = 100
    settings["step_size"] = 1.0
    settings["grad_tol"] = 1e-3
    settings["verbose"] = False
    return settings

def steepestDescent(ode_control_problem, x0, u_all_guess, settings):
    max_iter = settings["max_iter"]
    step_size = settings["step_size"]
    grad_tol = settings["grad_tol"]
    verbose = settings["verbose"]

    # Evaluate solves and gradients at initial guess
    u_all = u_all_guess
    x_all = ode_control_problem.solveFwd(x0, u_all)
    p_all = ode_control_problem.solveAdj(x_all, u_all)
    du_all = ode_control_problem.evalGradientControl(x_all, u_all, p_all)

    grad_norm0 = np.linalg.norm(du_all)
    cost_old = ode_control_problem.cost(x_all, u_all)

    num_iter = 0
    print("Iteration 0. Cost = %.3e \t GradNorm = %.3e" %(cost_old, grad_norm0))

    while num_iter < max_iter:
        u_all -= step_size * du_all

        # Evaluate new things
        x_all = ode_control_problem.solveFwd(x0, u_all)
        p_all = ode_control_problem.solveAdj(x_all, u_all)
        du_all = ode_control_problem.evalGradientControl(x_all, u_all, p_all)

        grad_norm = np.linalg.norm(du_all)
        cost_new = ode_control_problem.cost(x_all, u_all)
    
        fd_cost = (cost_new - cost_old)/step_size
        grad_cost = np.sum(du_all**2)
        if verbose:
            print("FD: %.3e, GRAD: %.3e" %(fd_cost, grad_cost))

        cost_old = cost_new

        num_iter += 1
        print("Iteration %d. Cost = %.3e \t GradNorm = %.3e" %(num_iter, cost_old, grad_norm))


        if grad_norm/grad_norm0 < grad_tol:
            print("Gradient norm less than tolerance")
            break

    return u_all
