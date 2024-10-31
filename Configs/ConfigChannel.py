
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/Channel/Coarse/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/Channel/Coarse/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': None
}

# Initial conditions
initial_conditions = {
    'U': (20.0, 0.0),
    'P': 2.0,
    'K': 1.5,
    'E': 2.23
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': None,
        'P': 2.0,
        'K': None,
        'E': None
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': None
    },
    'SYMMETRY':{
        'U': None,
        'P': None,
        'K': None,
        'E': None
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.00181818,
    'FORCE': (0.0, 0.0),
    'STEP_SIZE': 0.005
}

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6,
    'CFL_RELAXATION': 0.25
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/Channel/PVD files/',
    'H5_FILES':  'Results/Channel/H5 files/',
    'RESIDUALS': 'Results/Channel/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}