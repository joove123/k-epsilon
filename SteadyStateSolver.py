from dolfin import *
import time

from Utilities import are_close_all
from Utilities import calculate_cfl_time_step, bound_from_bellow
from Utilities import save_pvd_file, save_h5_file, save_list
from Utilities import visualize_functions, visualize_convergence
from Utilities import initialize_functions, initialize_mixed_functions, initialize_turbulent_terms
from Utilities import load_mesh_from_file, calculate_Distance_field

# -------------------- Simulation parameters -------------------- #

# mesh and facet directory
MESH_DIRECTORY  = 'meshes/BackStep/Coarse/mesh.xdmf'
FACET_DIRECTORY = 'meshes/BackStep/Coarse/facet.xdmf'

# specify what markers belong to what boundary
INFLOW_MARKERS   = [4]
OUTFLOW_MARKERS  = [2]
WALL_MARKERS     = [1,3]
SYMMETRY_MARKERS = [5]

# specify physical constants
NU    = 0.000181818
FORCE = (0.0, 0.0)

# specify initial conditions
U_INIT = 25.0
P_INIT = 0.0
K_INIT = 1.404
E_INIT = 1.970

# specify symulation for numerical integration
QUADRATURE_DEGREE = 2
ITER_MAX          = 3000
PICARD_RELAXATION = 0.1
TOLERANCE         = 1e-6

# directory for saving 
PVD_DIRECTORY       = None
H5_DIRECTORY        = None
RESIDUALS_DIRECTORY = None

# Post-processing
IS_PLOTTING = True
IS_SAVING   = False

# --------------------------------------------------------------- #

# -------------------- Problem construction --------------------- #

# Load mesh and construct integration measure
[mesh, marked_facets] = Load_mesh_from_file(MESH_DIRECTORY, FACET_DIRECTORY)
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})

# Construct periodic boundary condition
height = mesh.coordinates()[:, 1].max() - mesh.coordinates()[:, 1].min() 
width  = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()

# Construct function spaces
Element1 = VectorElement("CG", mesh.ufl_cell(), 2)
Element2 = FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem   = MixedElement([Element1, Element2])
W = FunctionSpace(mesh, W_elem)                                     
K = FunctionSpace(mesh, "CG", 1) 

# Construct distance function
y = Calculate_Distance_field(K, marked_facets, WALL_MARKERS, 0.0125)

# Construct boundary conditions
bcw=[]; bck=[]; bce=[]

for marker in INFLOW_MARKERS:
    bcw.append(DirichletBC(W.sub(0), Constant((U_INIT, 0.0)), marked_facets, marker))
    bck.append(DirichletBC(K, Constant(K_INIT), marked_facets, marker))
    bce.append(DirichletBC(K, Constant(E_INIT), marked_facets, marker))

for marker in OUTFLOW_MARKERS:
    bcw.append(DirichletBC(W.sub(1), Constant(0.0), marked_facets, marker))

for marker in WALL_MARKERS:
    bcw.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), marked_facets, marker))
    bck.append(DirichletBC(K, Constant(0.0), marked_facets, marker))

for marker in SYMMETRY_MARKERS:
    bcw.append(DirichletBC(W.sub(0).sub(1), Constant(0.0), marked_facets, marker))

# Construct functions
u,v,u1,u0,p,q,p1,p0,w1,w0 = \
              initialize_mixed_functions(W, Constant((U_INIT, 0.0, P_INIT)))
k,phi,k1,k0 = initialize_functions(K, Constant(K_INIT))
e,psi,e1,e0 = initialize_functions(K, Constant(E_INIT))

# Initialize nu and stepsize as fenics constants
nu    = Constant(NU)
force = Constant(FORCE)
n     = FacetNormal(mesh)

# initialize turbulent terms
nu_t, prod_k, react_k, prod_e, react_e = \
    initialize_turbulent_terms(nu, k0, e0, u1, y)

# --------------------------------------------------------------- #

# ------------------- Variational formulation ------------------- #

# RANS: steady-state
FW  = dot(dot(u0, nabla_grad(u)), v)*dx\
    + (nu + nu_t) * inner(nabla_grad(u), nabla_grad(v))*dx \
    - div(v)*p*dx \
    - div(u)*q*dx \
    + dot(p*n, v)*ds \
    - dot((nu + nu_t)*nabla_grad(u)*n, v)*ds \
    - dot(force, v)*dx

a_w = lhs(FW); l_w = rhs(FW)

# Transport equation for k: steady-state
FK  = dot(dot(u1, nabla_grad(k)), phi)*dx \
    + (nu + nu_t/1.) * inner(grad(k), grad(phi))*dx \
    - dot(prod_k, phi)*dx \
    + dot(react_k * k, phi)*dx 

# Transport equation for epsilon: steady-state
FE  = dot(dot(u1, nabla_grad(e)), psi)*dx \
    + (nu + nu_t/1.3) * inner(grad(e), grad(psi))*dx \
    - dot(prod_e, psi)*dx \
    + dot(react_e * e, psi)*dx 

a_k = lhs(FK); l_k = rhs(FK)
a_e = lhs(FE); l_e = rhs(FE)

# --------------------------------------------------------------- #

# -------------------------- Main loop -------------------------- #

residuals = {'u':[], 'p':[], 'k':[], 'e':[]}

loop_timer = time.time()
for iter in range(ITER_MAX):

    # Loop that solves RANS steps 1,2,3 and KEPS
    for a,l,bcs,f in zip((a_w,a_k,a_e),
                         (l_w,l_k,l_e),
                         (bcw,bck,bce),
                         (w1, k1, e1)):
        # Solve linear system
        A = assemble(a); b = assemble(l)
        [bc.apply(A,b) for bc in bcs]
        solve(A, f.vector(), b)

    # Bound k and epsilon from bellow
    k1 = bound_from_bellow(k1, 1e-16)
    e1 = bound_from_bellow(e1, 1e-16)

    # Convergence check
    break_flag, errors = are_close_all([u1,p1,k1,e1],
                                        [u0,p0,k0,e0],
                                        TOLERANCE)
    
    # Print summary
    print('iter: %g (%.2f s)   -   L2 errors:' %(iter+1, time.time() - loop_timer), \
        '  |u1-u0|= %.2e,'     % errors[0], \
        '  |p1-p0|= %.2e,'     % errors[1], \
        '  |k1-k0|= %.2e,'     % errors[2], \
        '  |e1-e0|= %.2e'      % errors[3], \
        '  (required: %.2e)'   % TOLERANCE)
    
    # Update all functions
    w0.assign(PICARD_RELAXATION * w1 + (1 - PICARD_RELAXATION) * w0)
    k0.assign(PICARD_RELAXATION * k1 + (1 - PICARD_RELAXATION) * k0)
    e0.assign(PICARD_RELAXATION * e1 + (1 - PICARD_RELAXATION) * e0)
    
    # update residual plot
    for i, (key, _) in enumerate(residuals.items()):
        residuals[key].append(errors[i])

    # break if converged
    if break_flag == True:
        print('# -------------------------------------------- #')
        print('#             Simulation converged             #')
        print('# -------------------------------------------- #')
        break

u1,p1 = w1.split(deepcopy=True)
solutions = {'u':u1, 'p':p1, 'k':k1, 'e':e1}

# --------------------------------------------------------------- #

# --------------- Visualization, Saving & Summary --------------- #

if IS_PLOTTING==True:
    # Visualize 
    visualize_functions(solutions)
    visualize_convergence(residuals)

if IS_SAVING==True:
    # Save .pvd and .h5 files
    for (key, f) in solutions.items():
        save_pvd_file(f, PVD_DIRECTORY + key + '.pvd')
        save_h5_file( f, H5_DIRECTORY  + key + '.h5')

    # Save .txt residual files
    for (key, f) in residuals.items():
        save_list(f, RESIDUALS_DIRECTORY + key + '.txt')
 
# --------------------------------------------------------------- #
