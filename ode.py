import numpy as np

class ODESystem:
    """
    Class for abstract ODE system
    dx/dt = f(x, u, t)
    where x is the state variable and u is some control/parameter
    """
    def forwardSystem(self, x, u, t):
        raise NotImplementedError("Child class should implement forwardSystem")

    def adjointSystem(self, p, x, u, t):
        raise NotImplementedError("Child class should implement adjointSystem")

    def jacobian_x(self, x, u, t):
        raise NotImplementedError("Child class should implement jacobian_x")

    def jacobian_u(self, x, u, t):
        raise NotImplementedError("Child class should implement jacobian_u")


def forwardEuler(ode_system, x0, u_all, t_all):
    """
    Solve the ODE by forward euler
    Parameters:
        - :code: `ode_system` ode system containing the time derivative expression
        - :code: `x0` initial condition
        - :code: `t_all` temporal discretiztaion (t0, ..., T)
        - :code: `u_all` control values at time points
    """
    dim = x0.shape[0]
    nt = t_all.shape[0] - 1

    # set in initial condition
    x_all = np.zeros((nt+1, dim))
    x_all[0] = x0

    for i in range(nt):
        dt = np.abs(t_all[i+1] - t_all[i])
        x_all[i+1] = x_all[i] + dt * ode_system(x_all[i], u_all[i], t_all[i])

    return x_all



