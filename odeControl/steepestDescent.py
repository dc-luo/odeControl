import numpy as np


def steepestDescentSettings():
    settings = dict()
    settings["max_iter"] = 100
    settings["step_size"] = 1.0
    settings["grad_tol"] = 1e-3
    settings["max_ls"] = 5
    settings["c_armijo"] = 1e-4
    settings["verbose"] = False
    return settings


def steepestDescentInitialCondition(ode_control_problem, x0_guess, u_all, settings, reg=None):
    """
    Steepest descent for solving the inverse problem for initial condition 
    """

    max_iter = settings["max_iter"]
    step_size = settings["step_size"]
    grad_tol = settings["grad_tol"]
    verbose = settings["verbose"]
    max_ls = settings["max_ls"]
    c_armijo = settings["c_armijo"]

    # Evaluate solves and gradients at initial guess
    m_all = x0_guess # Parameter for inversion
    x_all = ode_control_problem.solveFwd(m_all, u_all)
    p_all = ode_control_problem.solveAdj(x_all, u_all)

    dm_all = ode_control_problem.evalGradientInitialCondition(x_all, u_all, p_all)
    if reg is not None:
        # Add regularization due to gradient 
        dm_all += reg.grad(m_all)

    grad_norm0 = np.linalg.norm(dm_all, 2)
    grad_norm = grad_norm0 
    cost_old = ode_control_problem.cost(x_all, u_all)
    if reg is not None:
        cost_old += reg.cost(m_all)

    num_iter = 0
    print("Iteration 0. Cost = %.3e \t GradNorm = %.3e" %(cost_old, grad_norm0))

    while num_iter < max_iter:
        # Armijo line search
        ls_size = 1.0 
        ls_count = 0 
        # check the initial step
        m_proposed = - ls_size * step_size * dm_all  + m_all 
        x_proposed = ode_control_problem.solveFwd(m_proposed, u_all)
        cost_proposed = ode_control_problem.cost(x_proposed, u_all)
        if reg is not None:
            cost_proposed += reg.cost(m_proposed)

        if cost_proposed - cost_old <= -c_armijo * grad_norm:
            sufficient_descent = True
        else:
            sufficient_descent = False
            ls_size = ls_size/2

        while not sufficient_descent and ls_count < max_ls: 
            m_proposed = - ls_size * step_size * dm_all  + m_all 
            x_proposed = ode_control_problem.solveFwd(m_proposed, u_all)
            cost_proposed = ode_control_problem.cost(x_proposed, u_all)
            if reg is not None:
                cost_proposed += reg.cost(m_proposed)

            if cost_proposed - cost_old <= -c_armijo * grad_norm:
                sufficient_descent = True
            else:
                ls_size = ls_size/2
                ls_count += 1

        fd_cost = (cost_proposed - cost_old)/step_size
        grad_cost = np.sum(dm_all**2)

        if verbose:
            print("FD: %.3e, GRAD: %.3e. Took %d backtracking iterations" %(fd_cost, grad_cost, ls_count))

        # Take the updated step 
        cost_old = cost_proposed
        m_all = m_proposed
        x_all = ode_control_problem.solveFwd(m_all, u_all)
        p_all = ode_control_problem.solveAdj(x_all, u_all)
        dm_all = ode_control_problem.evalGradientInitialCondition(x_all, u_all, p_all)
        if reg is not None:
            # Add regularization due to gradient 
            dm_all += reg.grad(m_all)

        grad_norm = np.linalg.norm(dm_all, 2)

        num_iter += 1
        print("Iteration %d. Cost = %.3e \t GradNorm = %.3e" %(num_iter, cost_old, grad_norm))
        if grad_norm/grad_norm0 < grad_tol:
            print("Gradient norm less than tolerance")
            break

    return m_all



def steepestDescent(ode_control_problem, x0, u_all_guess, settings):
    max_iter = settings["max_iter"]
    step_size = settings["step_size"]
    grad_tol = settings["grad_tol"]
    verbose = settings["verbose"]
    max_ls = settings["max_ls"]
    c_armijo = settings["c_armijo"]

    # Evaluate solves and gradients at initial guess
    u_all = u_all_guess
    x_all = ode_control_problem.solveFwd(x0, u_all)
    p_all = ode_control_problem.solveAdj(x_all, u_all)
    du_all = ode_control_problem.evalGradientControl(x_all, u_all, p_all)

    grad_norm0 = np.linalg.norm(du_all, 2)
    grad_norm = grad_norm0 
    cost_old = ode_control_problem.cost(x_all, u_all)


    num_iter = 0
    print("Iteration 0. Cost = %.3e \t GradNorm = %.3e" %(cost_old, grad_norm0))

    while num_iter < max_iter:
        # Armijo line search
        ls_size = 1.0 
        ls_count = 0 
        # check the initial step
        u_proposed = - ls_size * step_size * du_all  + u_all 
        x_proposed = ode_control_problem.solveFwd(x0, u_proposed)
        cost_proposed = ode_control_problem.cost(x_proposed, u_proposed)
        if cost_proposed - cost_old <= -c_armijo * grad_norm:
            sufficient_descent = True
        else:
            sufficient_descent = False
            ls_size = ls_size/2

        while not sufficient_descent and ls_count < max_ls: 
            u_proposed = - ls_size * step_size * du_all  + u_all 
            x_proposed = ode_control_problem.solveFwd(x0, u_proposed)
            cost_proposed = ode_control_problem.cost(x_proposed, u_proposed)
            if cost_proposed - cost_old <= -c_armijo * grad_norm:
                sufficient_descent = True
            else:
                ls_size = ls_size/2
                ls_count += 1

        fd_cost = (cost_proposed - cost_old)/step_size
        grad_cost = np.sum(du_all**2)

        if verbose:
            print("FD: %.3e, GRAD: %.3e. Took %d backtracking iterations" %(fd_cost, grad_cost, ls_count))

        # Take the updated step 
        cost_old = cost_proposed
        u_all = u_proposed
        x_all = ode_control_problem.solveFwd(x0, u_all)
        p_all = ode_control_problem.solveAdj(x_all, u_all)
        du_all = ode_control_problem.evalGradientControl(x_all, u_all, p_all)
        grad_norm = np.linalg.norm(du_all, 2)

        num_iter += 1
        print("Iteration %d. Cost = %.3e \t GradNorm = %.3e" %(num_iter, cost_old, grad_norm))

        if grad_norm/grad_norm0 < grad_tol:
            print("Gradient norm less than tolerance")
            break

    return u_all
