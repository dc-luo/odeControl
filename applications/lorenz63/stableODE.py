import numpy as np 





class StabilizedAdjointControlProblem:
    def __init__(self, ode, t_all, integrator, beta, regularization, cumulative_costs=[], terminal_costs=[]):
        """
        Assumes uniform time step for forward and adjoint solutions
        """
        self.ode = ode
        self.t_all = t_all
        self.integrator = integrator
        self.cumulative_costs = cumulative_costs
        self.terminal_costs = terminal_costs
        assert hasattr(ode, "dim")
        self.state_dim = ode.dim
        self.beta = beta 
        self.regularization = regularization 

    def cost(self, x_all, u_all, p_all, m_all, s_all):
        total_cost = 0
        for cumulative_cost in self.cumulative_costs:
            total_cost += cumulative_cost.eval_all(x_all, u_all, self.t_all)

        for terminal_cost in self.terminal_costs:
            total_cost += terminal_cost.eval(x_all[-1], u_all[-1])

        return total_cost + self.regularization.cost(m_all) 
        # return total_cost + self.regularization.cost(m_all) + self.beta * np.linalg.norm(s_all - m_all)**2


    def solveFwd(self, x0, u_all):
        return self.integrator(self.ode.forwardSystem, x0, u_all, self.t_all)

    def solveAdjFwd(self, p0, x_all, u_all):
        """
        Solve the adjoint problem 
        """
        xu_all = np.append(x_all, u_all, axis=1)
        p_all = self.integrator(self.adjointSystem, p0, xu_all, self.t_all)
        return p_all 

    def solveAdjAdj(self, x_all, u_all, p_all): 
        """
        Solve the adjoint of the adjoint problem
        """
        qT = -p_all[-1]
        xu_all_reversed = np.flipud(np.append(x_all, u_all, axis=1))
        t_all_reversed = np.flipud(self.t_all)
        q_all_reversed = self.integrator(self.linearizedSystem, qT, xu_all_reversed, t_all_reversed)
        return np.flipud(q_all_reversed)

    def solveAdjGradient(self, x_all, u_all, p_all, q_all, m_all, s_all):
        return -q_all[0] + self.beta * (s_all - self.regularization.grad(m_all))


    def solveAdj(self, x_all, u_all):
        """
        Assumes the forward problem has been solved
        """
        assert x_all.shape[0] == self.t_all.shape[0]
        assert u_all.shape[0] == self.t_all.shape[0]

        pT = np.zeros(x_all[0].shape)
        for terminal_cost in self.terminal_costs:
            pT -= terminal_cost.jacobian_x(x_all[-1], u_all[-1])

        xu_all_reversed = np.flipud(np.append(x_all, u_all, axis=1))
        t_all_reversed = np.flipud(self.t_all)
        p_all_reversed = self.integrator(self.adjointSystem, pT, xu_all_reversed, t_all_reversed)
        # p_all_reversed = self.integrator(self.adjointSystem, pT, xu_all_reversed, t_all_reversed)

        # Flip time around
        # tau = t_all_reversed[0] - t_all_reversed

        return np.flipud(p_all_reversed)


    def evalGradientControl(self, x_all, u_all, p_all):
        """
        Evaluates gradient w.r.t. the control variable, full time dependence
        """
        # print("Evaluating control gradient")
        du_all = np.zeros(u_all.shape)

        # For temporal
        for i in range(len(self.t_all)-1):
            dt_curr = np.abs(self.t_all[i+1] - self.t_all[i])
            odeJ = np.array(self.ode.jacobian_u(x_all[i], u_all[i], self.t_all[i]))
            du_all[i] -= odeJ.T @ p_all[i] * dt_curr
            for cumulative_cost in self.cumulative_costs:
                du_all[i] += cumulative_cost.jacobian_u(x_all[i], u_all[i], self.t_all[i]) * dt_curr

        # Terminal
        for terminal_cost in self.terminal_costs:
            du_all[-1] += terminal_cost.jacobian_u(x_all[-1], u_all[-1])

        # print("Done")
        return du_all

    def evalGradientInitialCondition(self, x_all, u_all, p_all):
        """
        Evaluates gradient w.r.t. initial condition
        """
        return -p_all[0]


    def evalGradientJointInitialCondition(self, x_all, u_all, p_all, q_all, m_all, s_all):
        """
        Evaluates gradient w.r.t. initial condition
        """
        return -p_all[0] + self.regularization.grad(m_all), q_all[0] - self.beta * (s_all - self.regularization.grad(m_all))


    def linearizedSystem(self, q, xu, t):
        """
        Forms the adjoint rhs for time stepping

        Parameters
            :code: `q` Linearized forward variable
            :code: `xu` concatenation of state and control variables
            :code: `t` time

        Returns the rhs vector of same dimension as :code: `p`
        """
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]

        linearized_rhs = np.array(self.ode.linearizedSystem(q, x, u, t))

        return linearized_rhs



    def adjointSystem(self, p, xu, t):
        """
        Forms the adjoint rhs for time stepping

        Parameters
            :code: `p` adjoint variable
            :code: `xu` concatenation of state and control variables
            :code: `t` time

        Returns the rhs vector of same dimension as :code: `p`
        """
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]

        adjoint_rhs = np.array(self.ode.adjointSystem(p, x, u, t))
        for cumulative_cost in self.cumulative_costs:
            adjoint_rhs -= cumulative_cost.jacobian_x(x, u, t)

        return adjoint_rhs




def steepestDescentStabilized(stable_adjoint_control_problem, x0_guess, p0_guess, u_all, settings):
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
    m_all = x0_guess # initial state
    s_all = p0_guess # initial adjoint 

    x_all = stable_adjoint_control_problem.solveFwd(m_all, u_all)
    p_all = stable_adjoint_control_problem.solveAdjFwd(s_all, x_all, u_all)
    q_all = stable_adjoint_control_problem.solveAdjAdj(x_all, u_all, p_all)

    dm_all, ds_all = stable_adjoint_control_problem.evalGradientJointInitialCondition(x_all, u_all, p_all, q_all, m_all, s_all)

    grad_norm0 = np.linalg.norm(dm_all, 2)
    grad_norm = grad_norm0 
    cost_old = stable_adjoint_control_problem.cost(x_all, u_all, p_all, m_all, s_all)

    num_iter = 0
    print("Iteration 0. Cost = %.3e \t GradNorm = %.3e" %(cost_old, grad_norm0))

    while num_iter < max_iter:
        # Armijo line search
        ls_size = 1.0 
        ls_count = 0 
        # check the initial step
        m_proposed = - ls_size * step_size * dm_all  + m_all 
        s_proposed = - ls_size * step_size * ds_all  + s_all
        x_proposed = stable_adjoint_control_problem.solveFwd(m_proposed, u_all)
        p_proposed = stable_adjoint_control_problem.solveAdjFwd(s_proposed, x_proposed, u_all)
        cost_proposed = stable_adjoint_control_problem.cost(x_proposed, u_all, p_proposed, m_proposed, s_proposed)

        if cost_proposed - cost_old <= -c_armijo * grad_norm:
            sufficient_descent = True
        else:
            sufficient_descent = False
            ls_size = ls_size/2

        while not sufficient_descent and ls_count < max_ls: 
            m_proposed = - ls_size * step_size * dm_all  + m_all 
            s_proposed = - ls_size * step_size * ds_all  + s_all 
            x_proposed = stable_adjoint_control_problem.solveFwd(m_proposed, u_all)
            p_proposed = stable_adjoint_control_problem.solveAdjFwd(s_proposed, x_proposed, u_all)
            cost_proposed = stable_adjoint_control_problem.cost(x_proposed, u_all, p_proposed, m_proposed, s_proposed)
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
        s_all = s_proposed 
        x_all = stable_adjoint_control_problem.solveFwd(m_all, u_all)
        p_all = stable_adjoint_control_problem.solveAdjFwd(s_all, x_all, u_all)
        q_all = stable_adjoint_control_problem.solveAdjAdj(x_all, u_all, p_all)
        dm_all, ds_all = stable_adjoint_control_problem.evalGradientJointInitialCondition(x_all, u_all, p_all, q_all, m_all, s_all)

        grad_norm = np.sqrt(np.linalg.norm(dm_all, 2)**2 + np.linalg.norm(ds_all, 2)**2)

        num_iter += 1
        print("Iteration %d. Cost = %.3e \t GradNorm = %.3e" %(num_iter, cost_old, grad_norm))
        if grad_norm/grad_norm0 < grad_tol:
            print("Gradient norm less than tolerance")
            break

    return m_all, s_all