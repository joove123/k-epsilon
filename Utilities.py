from dolfin import *
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
import os

# ------------- Utilities for checking convergence  ------------- #

def _compute_l2_error(f1, f0):
    '''Compute l2 error of two functions: f1, f0''', 
    error = f1 - f0
    error = assemble(error**2*dx)
    return error

def _are_close(f1, f0, tol):
    '''Check if two functions: f1, f0 are sufficiently (tol) close'''
    error = _compute_l2_error(f1, f0)
    if error <= tol:
        return True, error
    else:
        return False, error

def are_close_all(fs1, fs0, tol):
    '''Check if all functions in fs1, fs0 are sufficiently (tol) close'''
    break_flag = False
    count  = 0
    errors = []

    for f1, f0 in zip(fs1, fs0):
        flag, error = _are_close(f1, f0, tol)
        errors.append(error)
        if flag == False:
            count += 1

    if count == 0:
        break_flag = True
    return  break_flag, errors

# --------------------------------------------------------------- #

# ---------------------- Saving utilities ----------------------- #

def save_pvd_file(f, directory):
    '''Saves function f as .pvd file (to inspect in ParaView)'''
    File(directory) << f

def save_h5_file(f, directory):
    '''Saves function f as .h5 file (to load back to FEniCS)'''
    fFile = HDF5File(MPI.COMM_WORLD, directory, "w")
    fFile.write(f,"/f")
    fFile.close()

def load_H5_files(Space, directory):
    '''Loads function from .h5 file to Space'''
    f = Function(Space)
    fFile = HDF5File(MPI.COMM_WORLD, directory, "r")
    fFile.read(f,"/f")
    fFile.close()
    return f

def save_list(dataset, directory):
    '''Saves python list as .txt file'''
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    with open(directory, 'w+') as file:
        for value in dataset:
            file.write(str(value) + '\n')

def load_list(directory):
    '''Loads .txt file into python list'''
    dataset = []
    with open(directory, 'r') as file:
        for line in file:
            dataset.append(float(line.strip()))
    return dataset

# --------------------------------------------------------------- #

# ---------------------- Solver utilities ----------------------- #

def calculate_cfl_time_step(u, delta_x, delta_y, relax, mesh):
    '''calculate time dt step base on cfl condition'''
    u_x = u[0]
    u_y = u[1]
    
    a = (abs(u_x) / delta_x + abs(u_y) / delta_y)
    local_cfl  = project(1. / a, FunctionSpace(mesh, "DG", 0))
    global_cfl = np.min(local_cfl.vector()[:])
    return relax * global_cfl

def bound_from_bellow(f, lb):
    '''bounds function f from bellow by lb'''
    new_f = Function(f.function_space())
    dimension = len(f.vector().get_local())      
    new_f.vector()[:] = np.max([f.vector()[:], lb * np.ones(dimension)], axis=0)
    return new_f

# --------------------------------------------------------------- #

# ------------------- Visualization utilities ------------------- #

def visualize_functions(solution_dictionary):
    '''Plots solutions'''
    num_plots = len(solution_dictionary)
    num_cols  = int(np.ceil(np.sqrt(num_plots)))  
    num_rows  = int(np.ceil(num_plots / num_cols))
    fig, _ = plt.subplots(num_rows, num_cols)

    for i, (key, f) in enumerate(solution_dictionary.items()):
        plt.subplot(num_rows, num_cols, i+1)
        if f.ufl_shape != ():
            c=plot(sqrt(dot(f, f)), title=key + ' magnitude')
        else:
            c=plot(f, title=key)
        plt.colorbar(c)
        plt.xlabel('x-direction')
        plt.ylabel('y-direction')

    fig.suptitle('Function plots')
    plt.tight_layout()
    plt.show()

def visualize_convergence(convergence_dictionary):
    '''Plots residuals'''
    num_plots = len(convergence_dictionary)
    num_cols = int(np.ceil(np.sqrt(num_plots)))  
    num_rows = int(np.ceil(num_plots / num_cols))  
    fig, axs = plt.subplots(num_rows, num_cols)
 
    for i, (key, values) in enumerate(convergence_dictionary.items()):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].plot(range(1, len(values)+1), values)
        axs[row, col].set_title(key)
        axs[row, col].set_yscale('log')
        axs[row, col].set_xlabel('iterations')
        axs[row, col].set_ylabel('error (log scale)')

    fig.suptitle('Convergence plots')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------- #

# -------------------- Function constructor --------------------- #

def _apply_initial_condition(Space, initial_condition):
    '''Applies initial condition (Function/Constant/float)'''
    if isinstance(initial_condition, Function):
        applied_condition = initial_condition  

    elif isinstance(initial_condition, Constant):
        applied_condition = project(initial_condition, Space)

    else:
        applied_condition = project(Constant(initial_condition), Space)
    return applied_condition

def initialize_functions(Space, initial_condition=None):
    '''Initialize all functions appearing in weak form'''
    trial_f = TrialFunction(Space)
    test_f  = TestFunction(Space)
    current_f  = Function(Space)
    previous_f = Function(Space)
    
    if initial_condition:
        previous_f = _apply_initial_condition(Space, initial_condition)

    return trial_f, test_f, current_f, previous_f

def initialize_mixed_functions(Space, initial_condition=None):
    '''Initialize all functions appearing in weak form (mixed)'''    
    (trial_f1, trial_f2) = TrialFunctions(Space)
    (test_f1,   test_f2) = TestFunctions(Space)
    current_mixed  = Function(Space)
    previous_mixed = Function(Space)

    if initial_condition:
        previous_mixed = _apply_initial_condition(Space, initial_condition)

    current_f1, current_f2   = split(current_mixed)
    previous_f1, previous_f2 = split(previous_mixed)

    return trial_f1, test_f1, current_f1, previous_f1, \
           trial_f2, test_f2, current_f2, previous_f2, \
           current_mixed, previous_mixed 

# --------------------------------------------------------------- #

# ----------------- Turbulent terms constructor ----------------- #

def initialize_turbulent_terms(nu, k0, e0, u1, y):
    '''Initializes turbulent terms'''
    def Max(a, b): return (a+b+abs(a-b))/Constant(2)
    def Min(a, b): return (a+b-abs(a-b))/Constant(2)

    Re_t = (1. / nu) * (k0**2 / e0) 
    Re_k = (1. / nu) * (sqrt(k0) * y) 

    f_nu = Max(Min((1 - exp(- 0.0165 * Re_k))**2 * (1 + 20.5 / Re_t), \
                   Constant(1.0)), Constant(0.01116225))
    f_1  = 1 + (0.05 / f_nu)**3
    f_2  = Max(Min(1 - exp(- Re_t**2), Constant(1.0)), Constant(0.0))
    S_sq = 2 * inner(sym(nabla_grad(u1)), sym(nabla_grad(u1)))

    nu_t    = 0.09 * f_nu * (k0**2 / e0)
    prod_k  = nu_t * S_sq
    react_k = e0 / k0
    prod_e  = 1.44 * react_k * f_1 * prod_k
    react_e = 1.92 * react_k * f_2

    return nu_t, prod_k, react_k, prod_e, react_e

# --------------------------------------------------------------- #

# ---------------- Mesh and distance constructor ---------------- #

def load_mesh_from_file(mesh_directory, facet_directory):
    '''Loads .xdmf mesh and faces mesh'''
    mesh = Mesh()
    with XDMFFile(mesh_directory) as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(facet_directory) as infile:
        infile.read(mvc)
    marked_facets = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return mesh, marked_facets

def calculate_Distance_field(Space, mf, wall_index, relax):
    '''computes ditance to boundaries specified by wall_index on mf'''
    bcy = []
    for index in wall_index:
        bc = DirichletBC(Space, Constant(0), mf, index)
        bcy.append(bc)

    y =  Function(Space)
    dy = TrialFunction(Space)
    z =  TestFunction(Space)
    relaxation = Constant(relax)
    g = Constant(1.0)

    #Linear approximation
    F0 = inner(grad(dy), grad(z))*dx - g*z*dx
    a0, L0 = lhs(F0), rhs(F0)
    solve(a0==L0, y, bcy)

    # Non-linear solver
    F0  = sqrt(inner(grad(y), grad(y)))*z*dx - g*z*dx \
        + relaxation*inner(grad(y),grad(z))*dx
    problem = NonlinearVariationalProblem(F0, y,J=derivative(F0, y), bcs=bcy)
    solver = NonlinearVariationalSolver(problem)    
    solver.solve()
    return y

# --------------------------------------------------------------- #
