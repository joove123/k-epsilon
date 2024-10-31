
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/BackStep/Fine/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/BackStep/Fine/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': [5]
}

# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0),
    'P': 0.0,
    'K': 1.73,
    'E': 1.46
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (25.0, 0.0),
        'P': None,
        'K': 1.73,
        'E': 1.46
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
        'U': 0.0,
        'P': None,
        'K': 1.73,
        'E': 1.46
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.000181818,
    'FORCE': (0.0, 0.0)
}

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6,
    'PICARD_RELAXATION': 0.1
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/BackStep/PVD files/',
    'H5_FILES':  'Results/BackStep/H5 files/',
    'RESIDUALS': 'Results/BackStep/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}