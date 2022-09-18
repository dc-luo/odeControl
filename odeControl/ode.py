import time
import numpy as np
import jax
import jax.numpy as jnp

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

class ODESystemJax(ODESystem):
    """
    Class for abstract ODE system
    dx/dt = f(x, u, t)
    where x is the state variable and u is some control/parameter
    """

    def __init__(self, ode_form):
        self.ode_form = ode_form
        self.compiled_ode_form = jax.jit(self.ode_form)
        self.compiled_Jx = jax.jit(jax.jacobian(self.compiled_ode_form, argnums=0))
        self.compiled_Ju = jax.jit(jax.jacobian(self.compiled_ode_form, argnums=1))

    def forwardSystem(self, x, u, t):
        """
        Forward System needs to return a jnp array 
        """ 
        return self.compiled_ode_form(x,u,t)

    def adjointSystem(self, p, x, u, t):
        # t0 = time.time()
        rhs = self.jacobian_x(x, u, t).T @ p
        # t1 = time.time()
        # print("Time to compute adjoint system: ", t1-t0)
        return rhs

    def jacobian_x(self, x, u, t):
        return self.compiled_Jx(x,u,t)

    def jacobian_u(self, x, u, t):
        return self.compiled_Ju(x,u,t)


def forwardEuler(ode_system, x0, u_all, t_all):
    """
    Solve the ODE by forward euler
    Parameters:
        - :code: `ode_system` ode system containing the time derivative expression
        - :code: `x0` initial condition
        - :code: `t_all` temporal discretiztaion (t0, ..., T)
        - :code: `u_all` control values at time points
    """
    # print("Solve via forward euler")
    dim = x0.shape[0]
    nt = t_all.shape[0] - 1

    # set in initial condition
    x_all = np.zeros((nt+1, dim))
    x_all[0] = x0

    for i in range(nt):
        dt = np.abs(t_all[i+1] - t_all[i])
        x_all[i+1] = x_all[i] + dt * np.array(ode_system(x_all[i], u_all[i], t_all[i]))
    # print("Done")
    return x_all



