from dolfin import *
from Utilities import *

class KEpsilonGeneral:
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field):
        # K-epsilon model parameters
        self._K = K
        self._bck = bck
        self._bce = bce
        self._k_init = k_init
        self._e_init = e_init

        # General parameters
        self._nu = nu
        self._force = force
        self._dx = custom_dx
        self._ds = custom_ds
        self._y = distance_field  # distance function

        # Initialize all functions
        self._construct_functions()

    def _construct_functions(self):
        """Construct model functions."""
        self._k, self._phi, self._k1, self._k0 = initialize_functions(self._K, Constant(self._k_init))
        self._e, self._psi, self._e1, self._e0 = initialize_functions(self._K, Constant(self._e_init))

    def construct_forms(self):
        raise NotImplementedError("This method must be implemented in subclasses.")
    
    def solve_turbulence_model(self):
        """Solves the turbulence model equations for k and e."""
        # solve k 
        A_K = assemble(self._a_k); b_k = assemble(self._l_k)
        [bc.apply(A_K,b_k) for bc in self._bck]
        solve(A_K, self._k1.vector(), b_k)

        # solve e
        A_E = assemble(self._a_e); b_e = assemble(self._l_e)
        [bc.apply(A_E,b_e) for bc in self._bce]
        solve(A_E, self._e1.vector(), b_e)

        # bound from bellow
        self._k1 = bound_from_bellow(self._k1, 1e-16)
        self._e1 = bound_from_bellow(self._e1, 1e-16)

    def update_variables(self, relaxation = 1.0):
        """Update k and e variables with relaxation."""
        self._k0.assign(relaxation * self._k1 + (1.0 - relaxation) * self._k0)
        self._e0.assign(relaxation * self._e1 + (1.0 - relaxation) * self._e0)

    def _construct_turbulent_quantities(self, external_u1):
        """Construct turbulent quantities used in variational forms."""
        def Max(a, b): return (a+b+abs(a-b))/Constant(2)
        def Min(a, b): return (a+b-abs(a-b))/Constant(2)

        Re_t = (1. / self._nu) * (self._k0**2 / self._e0) 
        Re_k = (1. / self._nu) * (sqrt(self._k0) * self._y) 

        f_nu = Max(Min((1 - exp(- 0.0165 * Re_k))**2 * (1 + 20.5 / Re_t), Constant(1.0)), Constant(0.01116225))
        f_1  = 1 + (0.05 / f_nu)**3
        f_2  = Max(Min(1 - exp(- Re_t**2), Constant(1.0)), Constant(0.0))
        S_sq = 2 * inner(sym(nabla_grad(external_u1)), sym(nabla_grad(external_u1)))

        # Define turbulent quantities
        self._nu_t = 0.09 * f_nu * (self._k0**2 / self._e0)
        self._prod_k = self._nu_t * S_sq
        self._react_k = self._e0 / self._k0
        self._prod_e = 1.44 * self._react_k * f_1 * self._prod_k
        self._react_e = 1.92 * self._react_k * f_2

    # Properties which are refered to in main
    @property
    def nu_t(self):
        return self._nu_t
    
    @property
    def k0(self):
        return self._k0
    
    @property
    def k1(self):
        return self._k1
    
    @property
    def e0(self):
        return self._e0
    
    @property
    def e1(self):
        return self._e1


class KEpsilonSteadyState(KEpsilonGeneral):
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field):
        super().__init__(K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field)   
                
    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        # Transport equation for k
        FK  = dot(dot(external_u1, nabla_grad(self._k)), self._phi)*self._dx \
            + inner((self._nu + self._nu_t / 1.0) * grad(self._k), grad(self._phi))*self._dx \
            - dot(self._prod_k, self._phi)*self._dx \
            + dot(self._react_k * self._k, self._phi)*self._dx
        self._a_k = lhs(FK); self._l_k = rhs(FK)

        # Transport equation for e        
        FE  = dot(dot(external_u1, nabla_grad(self._e)), self._psi)*self._dx \
            + inner((self._nu + self._nu_t / 1.3) * grad(self._e), grad(self._psi))*self._dx \
            - dot(self._prod_e, self._psi)*self._dx \
            + dot(self._react_e * self._e, self._psi)*self._dx
        self._a_e = lhs(FE); self._l_e = rhs(FE)


class KEpsilonTransient(KEpsilonGeneral):
    def __init__(self, K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, dt, distance_field):
        self._dt = dt
        super().__init__(K, bck, bce, k_init, e_init, nu, force, custom_dx, custom_ds, distance_field)       
        
    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        # Transport equation for k
        FK  = dot((self._k - self._k0) / self._dt, self._phi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._k)), self._phi)*self._dx \
            + inner((self._nu + self._nu_t / 1.0) * grad(self._k), grad(self._phi))*self._dx \
            - dot(self._prod_k, self._phi)*self._dx \
            + dot(self._react_k * self._k, self._phi)*self._dx
        self._a_k = lhs(FK); self._l_k = rhs(FK)

        # Transport equation for e        
        FE  = dot((self._e - self._e0) / self._dt, self._psi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._e)), self._psi)*self._dx \
            + inner((self._nu + self._nu_t / 1.3) * grad(self._e), grad(self._psi))*self._dx \
            - dot(self._prod_e, self._psi)*self._dx \
            + dot(self._react_e * self._e, self._psi)*self._dx
        self._a_e = lhs(FE); self._l_e = rhs(FE)