"""
MNPBEM Structure Configuration

This file defines the geometric structure and materials of nanoparticles.
All structure-related parameters should be defined here.
"""

from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME (IDENTIFIER)
# ============================================================================
# Give your structure a descriptive name for easy identification
args['structure_name'] = 'au_ag_agcl_dimer'

# ============================================================================
# STRUCTURE TYPE
# ============================================================================
# Choose the basic structure type:
# 
# Single particles:
#   - 'sphere'                    : Single sphere
#   - 'cube'                      : Single cube
#   - 'rod'                       : Rod/cylinder
#   - 'ellipsoid'                 : Ellipsoid
#   - 'triangle'                  : Triangular nanoparticle
#
# Core-shell structures:
#   - 'core_shell_sphere'         : Core-shell sphere (2 layers)
#   - 'core_shell_cube'           : Core-shell cube (2 layers)
#   - 'multi_shell_sphere'        : Multi-shell sphere (3+ layers) **NEW**
#   - 'multi_shell_cube'          : Multi-shell cube (3+ layers) **NEW**
#
# Dimers:
#   - 'dimer_sphere'              : Two spheres
#   - 'dimer_cube'                : Two cubes
#   - 'dimer_core_shell_sphere'   : Two core-shell spheres
#   - 'dimer_core_shell_cube'     : Two core-shell cubes
#   - 'dimer_multi_shell_sphere'  : Two multi-shell spheres **NEW**
#   - 'dimer_multi_shell_cube'    : Two multi-shell cubes **NEW**

args['structure'] = 'dimer_multi_shell_sphere'

# ============================================================================
# MESH DENSITY
# ============================================================================
# Controls the number of boundary elements
# Higher = more accurate but slower and more memory
# Recommended values:
#   - 144 for spheres (standard)
#   - 12-16 for cubes
#   - Increase for complex geometries or high accuracy requirements

args['mesh_density'] = 144

# ============================================================================
# GEOMETRY PARAMETERS (STRUCTURE-SPECIFIC)
# ============================================================================
# Different structure types require different parameters
# Uncomment and modify the section for your chosen structure

# --- For 'sphere' ---
# args['diameter'] = 50  # nm

# --- For 'cube' ---
# args['size'] = 40  # nm (edge length)
# args['rounding'] = 0.25  # Edge rounding parameter (0-1, smaller = sharper edges)

# --- For 'rod' (cylinder along z-axis) ---
# args['diameter'] = 20  # nm
# args['height'] = 80  # nm

# --- For 'ellipsoid' ---
# args['axes'] = [20, 30, 40]  # [x, y, z] semi-axes in nm

# --- For 'triangle' (2D extruded) ---
# args['side_length'] = 50  # nm
# args['thickness'] = 10  # nm

# --- For 'dimer_sphere' ---
# args['diameter'] = 50  # nm
# args['gap'] = 5  # Surface-to-surface gap in nm

# --- For 'dimer_cube' ---
# args['size'] = 40  # nm
# args['gap'] = 10  # nm
# args['rounding'] = 0.25

# --- For 'core_shell_sphere' (2 layers) ---
# args['core_diameter'] = 40  # nm
# args['shell_thickness'] = 10  # nm (total diameter = 40 + 2*10 = 60)

# --- For 'core_shell_cube' (2 layers) ---
# args['core_size'] = 30  # nm
# args['shell_thickness'] = 5  # nm
# args['rounding'] = 0.25

# ============================================================================
# MULTI-SHELL STRUCTURE (3+ LAYERS) **NEW**
# ============================================================================
# For structures like Au@Ag@AgCl, define layers from core to outermost shell
# Each layer is a dictionary with material and size information

# **ACTIVE CONFIGURATION: Au@Ag@AgCl Multi-shell Dimer**
args['layers'] = [
    {
        'name': 'core',           # Layer name (for reference)
        'material': 'gold',       # Material name or custom dict
        'radius': 47,             # Core radius in nm (for first layer only)
    },
    {
        'name': 'shell_1',        # First shell
        'material': 'silver',     # Material name or custom dict
        'thickness': 3,           # Shell thickness in nm (total radius: 47+3=50)
    },
    {
        'name': 'shell_2',        # Second shell (outermost)
        'material': 'agcl',       # Custom material (defined below)
        'thickness': 1,           # Shell thickness in nm (total radius: 50+1=51)
    }
]

# Notes on layers:
# - First layer MUST have 'radius' (this is the core)
# - Subsequent layers MUST have 'thickness'
# - Layers are built from inside out
# - Total radius = core_radius + sum(all shell thicknesses)
# - For cube structures, 'radius' → 'size' and it refers to edge length

# Example: 4-layer structure (Au@Ag@AgCl@SiO2)
# args['layers'] = [
#     {'name': 'core',    'material': 'gold',   'radius': 40},
#     {'name': 'shell_1', 'material': 'silver', 'thickness': 5},
#     {'name': 'shell_2', 'material': 'agcl',   'thickness': 2},
#     {'name': 'shell_3', 'material': 'sio2',   'thickness': 3},
# ]

# ============================================================================
# DIMER CONFIGURATION (FOR DIMER STRUCTURES)
# ============================================================================
# Define how two particles are arranged and transformed

args['dimer'] = {
    # Gap between particles (surface-to-surface distance in nm)
    'gap': 5,
    
    # Particle 1 configuration (reference particle)
    'particle1': {
        'position': [0, 0, 0],          # [x, y, z] position in nm
        'rotation': [0, 0, 0],          # [rx, ry, rz] rotation in degrees
        'rotation_order': 'xyz',        # Order of rotation axes ('xyz', 'zyx', etc.)
    },
    
    # Particle 2 configuration (transformed particle)
    'particle2': {
        'position': None,               # None = auto-calculate based on gap
                                        # Or specify [x, y, z] manually
        
        'rotation': [0, 0, 0],          # [rx, ry, rz] rotation in degrees
        'rotation_order': 'xyz',        # Order of rotation axes
        
        'offset': [0, 0, 0],            # Additional offset [x, y, z] in nm
                                        # Applied after automatic positioning
    }
}

# Detailed explanation of dimer positioning:
#
# 1. Automatic positioning (position=None):
#    - Particle 1 is placed at particle1['position']
#    - Particle 2 is automatically placed along x-axis with specified gap
#    - Additional offset can be applied via particle2['offset']
#
# 2. Manual positioning (position=[x,y,z]):
#    - Both particles are placed exactly at specified positions
#    - 'gap' parameter is ignored
#    - Use this for custom arrangements
#
# 3. Rotation:
#    - Applied around particle's own center
#    - Rotation order matters: 'xyz' means rotate around x, then y, then z
#    - Angles in degrees
#
# 4. Complete transformation order:
#    - Create particle → Rotate → Translate to position → Apply offset

# Example: Tilted dimer
# args['dimer'] = {
#     'gap': 10,
#     'particle1': {
#         'position': [0, 0, 0],
#         'rotation': [0, 0, 0],
#         'rotation_order': 'xyz',
#     },
#     'particle2': {
#         'position': None,              # Auto-calculate
#         'rotation': [0, 15, 0],        # Tilt 15° around y-axis
#         'rotation_order': 'xyz',
#         'offset': [0, 5, 0],           # Shift 5nm in y-direction
#     }
# }

# Example: Custom arrangement (L-shaped dimer)
# args['dimer'] = {
#     'gap': None,  # Ignored when using manual positions
#     'particle1': {
#         'position': [0, 0, 0],
#         'rotation': [0, 0, 0],
#         'rotation_order': 'xyz',
#     },
#     'particle2': {
#         'position': [0, 60, 0],        # Place along y-axis instead of x
#         'rotation': [0, 0, 90],        # Rotate 90° around z-axis
#         'rotation_order': 'xyz',
#         'offset': [0, 0, 0],
#     }
# }

# ============================================================================
# MATERIALS
# ============================================================================
# Define custom materials used in the structure
# Built-in materials (no definition needed): 
#   'air', 'water', 'vacuum', 'glass', 'silicon', 'sapphire', 'sio2',
#   'gold', 'silver', 'aluminum'

# Custom materials dictionary
args['custom_materials'] = {
    # Material name: material definition
    
    # Example 1: Constant dielectric function
    # 'my_dielectric': {
    #     'type': 'constant',
    #     'epsilon': 2.5
    # },
    
    # Example 2: From data file (wavelength-dependent with interpolation)
    'agcl': {
        'type': 'table',
        'file': Path.home() / 'research' / 'materials' / 'agcl.dat'
    },
    
    # Example 3: Relative path
    # 'custom_metal': {
    #     'type': 'table',
    #     'file': './materials/custom_metal.dat'
    # },
    
    # Example 4: Custom function (advanced)
    # 'drude_metal': {
    #     'type': 'function',
    #     'formula': '1 - 9.0^2/(w*(w+1i*0.1))',  # Drude model
    #     'unit': 'eV'
    # },
}

# File format for 'table' type materials:
#
# Option A - Two columns (real refractive index only, no absorption):
#   # wavelength(nm)  n
#   400              1.5
#   410              1.6
#   420              1.7
#   ...
#
# Option B - Three columns (complex refractive index with absorption):
#   # wavelength(nm)  n      k
#   400              1.5    0.1
#   410              1.6    0.12
#   420              1.7    0.14
#   ...
#
# Notes:
# - Comments start with '#'
# - The system automatically interpolates to simulation wavelengths
# - Linear interpolation is used
# - Wavelengths should be in ascending order

# ============================================================================
# MEDIUM (ENVIRONMENT)
# ============================================================================
# The medium surrounding the nanoparticles
# Options:
#   - Built-in: 'air', 'water', 'vacuum'
#   - Custom constant: {'type': 'constant', 'epsilon': 1.77}
#   - Custom from file: {'type': 'table', 'file': 'medium.dat'}

args['medium'] = 'air'

# Example: Nanoparticles in water
# args['medium'] = 'water'

# Example: Custom medium (e.g., oil with n=1.33)
# args['medium'] = {'type': 'constant', 'epsilon': 1.33**2}  # epsilon = n^2

# ============================================================================
# SUBSTRATE (OPTIONAL)
# ============================================================================
# Use substrate below the nanoparticles?
args['use_substrate'] = False

# Substrate configuration (only used if use_substrate=True)
args['substrate'] = {
    'material': 'glass',           # Substrate material
    'position': 0,                 # z-position of interface (nm)
}

# Example: Gold nanoparticle on glass substrate
# args['use_substrate'] = True
# args['medium'] = 'air'
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,
# }

# Example: Custom substrate material
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': {'type': 'constant', 'epsilon': 2.1},
#     'position': -10,
# }

# ============================================================================
# ADDITIONAL STRUCTURE EXAMPLES
# ============================================================================

# Example 1: Simple gold sphere
# args['structure'] = 'sphere'
# args['diameter'] = 50
# args['mesh_density'] = 144
# args['medium'] = 'water'

# Example 2: Silver nanocube
# args['structure'] = 'cube'
# args['size'] = 40
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['medium'] = 'air'

# Example 3: Core-shell nanoparticle (Au@Ag)
# args['structure'] = 'core_shell_sphere'
# args['core_diameter'] = 40
# args['shell_thickness'] = 10
# args['mesh_density'] = 144
# args['medium'] = 'air'
# # Materials will be: [medium, shell, core] = ['air', 'silver', 'gold']

# Example 4: Simple dimer
# args['structure'] = 'dimer_sphere'
# args['diameter'] = 50
# args['gap'] = 5
# args['mesh_density'] = 144
# args['medium'] = 'air'

# Example 5: Rotated dimer
# args['structure'] = 'dimer_cube'
# args['size'] = 30
# args['gap'] = 10
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['dimer'] = {
#     'gap': 10,
#     'particle1': {'position': [0, 0, 0], 'rotation': [0, 0, 0], 'rotation_order': 'xyz'},
#     'particle2': {'position': None, 'rotation': [0, 15, 0], 'rotation_order': 'xyz', 'offset': [0, 0, 0]}
# }