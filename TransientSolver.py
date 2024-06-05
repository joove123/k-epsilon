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
MESH_DIRECTORY  = 'github-kepsilon/meshes/Channel/Coarse/mesh.xdmf'
FACET_DIRECTORY = 'github-kepsilon/meshes/Channel/Coarse/facet.xdmf'

# specify what markers belong to what boundary
INFLOW_MARKERS   = [4]
OUTFLOW_MARKERS  = [2]
WALL_MARKERS     = [1,3]
SYMMETRY_MARKERS = None

# specify physical constants
NU    = 0.00181818
FORCE = (0.0, 0.0)

# specify initial conditions
U_INIT = 20.0
P_INIT = 2.0
K_INIT = 1.5
E_INIT = 2.23

# specify symulation for numerical integration
QUADRATURE_DEGREE = 2
STEP_SIZE         = 0.005
ITER_MAX          = 3000
CFL_RELAXATION    = 0.25
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
[mesh, marked_facets] = load_mesh_from_file(MESH_DIRECTORY, FACET_DIRECTORY)
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})

# Construct periodic boundary condition
height = mesh.coordinates()[:, 1].max() - mesh.coordinates()[:, 1].min() 
width  = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()

class Periodic(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)
    def map(self, x, y):
        y[0] = x[0] - width
        y[1] = x[1]
periodic = Periodic(1E-5)

# Construct function spaces
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = periodic)       
Q = FunctionSpace(mesh, "CG", 1)                                        
K = FunctionSpace(mesh, "CG", 1, constrained_domain = periodic)  

# Construct distance function
y = Expression('H/2 - abs(H/2 - x[1])', H = height, degree = 2)

# Construct boundary conditions
bcu=[]; bcp=[]; bck=[]; bce=[]

for marker in INFLOW_MARKERS:
    bcp.append(DirichletBC(Q, Constant(P_INIT), marked_facets, marker))

for marker in OUTFLOW_MARKERS:
    bcp.append(DirichletBC(Q, Constant(0.0),    marked_facets, marker))

for marker in WALL_MARKERS:
    bcu.append(DirichletBC(V, Constant((0.0, 0.0)), marked_facets, marker))
    bck.append(DirichletBC(K, Constant(0.0), marked_facets, marker))

# Construct functions
u,v,  u1,u0 = initialize_functions(V, Constant((U_INIT, 0.0)))
p,q,  p1,p0 = initialize_functions(Q, Constant(P_INIT))
k,phi,k1,k0 = initialize_functions(K, Constant(K_INIT))
e,psi,e1,e0 = initialize_functions(K, Constant(E_INIT))

# Initialize nu and stepsize as fenics constants
nu    = Constant(NU)
force = Constant(FORCE)
dt    = Constant(STEP_SIZE)

# initialize turbulent terms
nu_t, prod_k, react_k, prod_e, react_e = \
    initialize_turbulent_terms(nu, k0, e0, u1, y)

# --------------------------------------------------------------- #

# ------------------- Variational formulation ------------------- #

# RANS: tentative velocity
F1  = dot((u - u0) / dt, v)*dx \
    + dot(dot(u0, nabla_grad(u)), v)*dx \
    + inner((nu + nu_t) * grad(u), grad(v))*dx \
    - dot(force, v)*dx

# RANS: Pressure correction
F2  = dot(grad(p), grad(q))*dx + dot(div(u1) / dt, q)*dx

# RANS: Velocity correction
F3  = dot(u, v)*dx - dot(u1, v)*dx + dt * dot(grad(p1), v)*dx

a_1 = lhs(F1); l_1 = rhs(F1)
a_2 = lhs(F2); l_2 = rhs(F2)
a_3 = lhs(F3); l_3 = rhs(F3)

# Transport equation for k
FK  = dot((k - k0) / dt, phi)*dx \
    + dot(dot(u1, nabla_grad(k)), phi)*dx \
    + inner((nu + nu_t / 1.0) * grad(k), grad(phi))*dx \
    - dot(prod_k, phi)*dx \
    + dot(react_k * k, phi)*dx 

# Transport equation for epsilon
FE  = dot((e - e0) / dt, psi)*dx \
    + dot(dot(u1, nabla_grad(e)), psi)*dx \
    + inner((nu + nu_t / 1.3) * grad(e), grad(psi))*dx \
    - dot(prod_e, psi)*dx \
    + dot(react_e * e, psi)*dx 

a_k = lhs(FK); l_k = rhs(FK)
a_e = lhs(FE); l_e = rhs(FE)

# --------------------------------------------------------------- #

# -------------------------- Main loop -------------------------- #

residuals = {'u':[], 'p':[], 'k':[], 'e':[]}

loop_timer = time.time()
for iter in range(ITER_MAX):

    # Compute step size based on cfl 
    if iter > 0:
        h_x = MaxCellEdgeLength(mesh)
        h_y = MinCellEdgeLength(mesh)
        step_size = calculate_cfl_time_step(u0, h_x, h_y, CFL_RELAXATION, mesh)
        dt.assign(Constant(step_size))

    # Loop that solves RANS steps 1,2,3 and KEPS
    for a,l,bcs,f in zip((a_1,a_2,a_3,a_k,a_e),
                         (l_1,l_2,l_3,l_k,l_e),
                         (bcu,bcp,bcu,bck,bce),
                         (u1, p1, u1, k1, e1)):
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
    u0.assign(u1)
    p0.assign(p1)
    k0.assign(k1)
    e0.assign(e1)
    
    # update residual plot
    for i, (key, _) in enumerate(residuals.items()):
        residuals[key].append(errors[i])

    # break if converged
    if break_flag == True:
        print('# -------------------------------------------- #')
        print('#             Simulation converged             #')
        print('# -------------------------------------------- #')
        break

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
