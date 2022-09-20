import numpy as np
import matplotlib.pyplot as plt


class CumulativeCost:
    """
    Cost functional for \int_{t=0}^{T} C(x(t), u(t), t) dt
    """
    def eval(self, x, u, t):
        """
        Evaluates the instance C(x(t), u(t), t) at time t
        """
        raise NotImplementedError("child class needs to implement eval")

    def eval_all(self, x_all, u_all, t_all):
        """
        Evaluates the full integral of cost along trajectory x_all, u_all
        """
        raise NotImplementedError("child class needs to implement eval_all")

    def jacobian_x(self, x, u, t):
        """
        Evaluates the Jacobian dC/dx at (x(t), u(t), t)
        """
        raise NotImplementedError("child class needs to implement jacobian_x")


    def jacobian_u(self, x, u, t):
        """
        Evaluates the Jacobian dC/du at (x(t), u(t), t)
        """
        raise NotImplementedError("child class needs to implement jacobian_u")


class TerminalCost:
    """
    Cost functional terminal cost K(x(T), u(T), T)
    """
    def eval(self, x, u):
        """
        Evaluates the terminal cost for trajectory x, u
        """
        raise NotImplementedError("child class needs to implement eval")

    def jacobian_x(self, x, u):
        """
        Evaluates the jacobian dK/dx of the terminal cost for trajectory x, u
        """
        raise NotImplementedError("child class needs to implement jacobian_x")

    def jacobian_u(self, x, u):
        """
        Evaluates the jacobian dK/du of the terminal cost for trajectory x, u
        """
        raise NotImplementedError("child class needs to implement jacobian_u")


class CumulativeL2Penalty(CumulativeCost):
    def __init__(self, u0, alpha=1.0):
        self.u0 = u0
        self.alpha = alpha

    def eval(self, x, u, t):
        return self.alpha*np.linalg.norm(u - self.u0)**2

    def jacobian_x(self, x, u, t):
        return np.zeros(x.shape)

    def jacobian_u(self, x, u, t):
        return self.alpha*2*(u - self.u0)

    def eval_all(self, x_all, u_all, t_all):
        diff_trajectory = self.alpha*np.linalg.norm(u_all - self.u0, axis=1)**2
        dts = t_all[1:] - t_all[:-1]
        # Integration by trapezoidal rule
        integral = np.inner(diff_trajectory[:-1], 0.5*dts)
        integral += np.inner(diff_trajectory[1:], 0.5*dts)
        return integral




class CumulativeL2Misfit(CumulativeCost):
    def __init__(self, x_target, alpha=1.0, W=None):
        self.x_target = x_target
        self.alpha = alpha
        self.W = W # Weighting matrix

    def eval(self, x, u, t):
        if self.W is None:
            return self.alpha*np.linalg.norm(x - self.x_target)**2
        else:
            raise NotImplementedError("Currently not implemented with nonzero weight")

    def jacobian_x(self, x, u, t):
        if self.W is None:
            return 2*self.alpha*(x - self.x_target)
        else:
            raise NotImplementedError("Currently not implemented with nonzero weight")

    def jacobian_u(self, x, u, t):
        return np.zeros(u.shape)

    def eval_all(self, x_all, u_all, t_all):
        if self.W is None:
            assert x_all.shape[0] == t_all.shape[0]
            assert x_all.shape[1] == self.x_target.shape[0]
            diff_trajectory = self.alpha*np.linalg.norm(x_all - self.x_target, axis=1)**2

            dts = t_all[1:] - t_all[:-1]

            # Integration by trapezoidal rule
            integral = np.inner(diff_trajectory[:-1], 0.5*dts)
            integral += np.inner(diff_trajectory[1:], 0.5*dts)

            return integral
        else:
            raise NotImplementedError("Currently not implemented with nonzero weight")

class TerminalL2Misfit(TerminalCost):
    def __init__(self, x_target, alpha=1.0):
        self.x_target = x_target
        self.alpha = alpha

    def eval(self, x, u):
        return self.alpha*np.linalg.norm(x - self.x_target) ** 2

    def jacobian_x(self, x, u):
        return 2*self.alpha*(x - self.x_target)

    def jacobian_u(self, x, u):
        return np.zeros(u.shape)



class ODEControlModel:
    def __init__(self, ode, t_all, integrator, cumulative_costs=[], terminal_costs=[]):
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

    def cost(self, x_all, u_all):

        total_cost = 0

        for cumulative_cost in self.cumulative_costs:
            total_cost += cumulative_cost.eval_all(x_all, u_all, self.t_all)

        for terminal_cost in self.terminal_costs:
            total_cost += terminal_cost.eval(x_all[-1], u_all[-1])

        return total_cost


    def solveFwd(self, x0, u_all):
        return self.integrator(self.ode.forwardSystem, x0, u_all, self.t_all)

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

