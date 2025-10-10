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
args['structure_name'] = 'simple_au_sphere'

# ============================================================================
# REFRACTIVE INDEX FILE PATHS
# ============================================================================
# Override default refractive index data file paths
# Default built-in materials use MNPBEM's internal .dat files
# Use this to specify custom paths for refractive index data files

args['refractive_index_paths'] = {
    # Built-in material overrides (optional)
    # 'gold': '/custom/path/to/gold.dat',
    # 'silver': '/custom/path/to/silver.dat',
    # 'aluminum': '/custom/path/to/aluminum.dat',
}

# Examples:
# args['refractive_index_paths'] = {
#     'gold': Path.home() / 'research' / 'materials' / 'gold_custom.dat',
#     'silver': '/usr/local/materials/silver_jc.dat',
# }

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
#   - 'multi_shell_sphere'        : Multi-shell sphere (3+ layers)
#   - 'multi_shell_cube'          : Multi-shell cube (3+ layers)
#
# Dimers:
#   - 'dimer_sphere'              : Two spheres
#   - 'dimer_cube'                : Two cubes
#   - 'dimer_core_shell_sphere'   : Two core-shell spheres
#   - 'dimer_core_shell_cube'     : Two core-shell cubes
#   - 'dimer_multi_shell_sphere'  : Two multi-shell spheres
#   - 'dimer_multi_shell_cube'    : Two multi-shell cubes

args['structure'] = 'sphere'

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
args['diameter'] = 50  # nm

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
# MULTI-SHELL STRUCTURE (3+ LAYERS)
# ============================================================================
# For structures like Au@Ag@AgCl, define layers from core to outermost shell
# Each layer is a dictionary with material and size information

# Example: Au@Ag@AgCl Multi-shell Dimer
# args['layers'] = [
#     {
#         'name': 'core',
#         'material': 'gold',
#         'diameter': 30  # nm (for sphere) or 'size' for cube
#     },
#     {
#         'name': 'shell1',
#         'material': 'silver',
#         'thickness': 5  # nm
#     },
#     {
#         'name': 'shell2',
#         'material': 'agcl',
#         'thickness': 3  # nm
#     }
# ]

# ============================================================================
# DIMER CONFIGURATION (FOR DIMER STRUCTURES)
# ============================================================================
# Advanced dimer positioning and orientation

# args['dimer'] = {
#     'gap': 10,  # Surface-to-surface gap in nm
#     'particle1': {
#         'position': [0, 0, 0],
#         'rotation': [0, 0, 0],  # Rotation angles in degrees [x, y, z]
#         'rotation_order': 'xyz'
#     },
#     'particle2': {
#         'position': None,  # Auto-calculated from gap if None
#         'rotation': [0, 15, 0],
#         'rotation_order': 'xyz',
#         'offset': [0, 0, 0]  # Additional offset after gap calculation
#     }
# }

# ============================================================================
# CUSTOM MATERIALS
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
    # 'agcl': {
    #     'type': 'table',
    #     'file': Path.home() / 'research' / 'materials' / 'agcl.dat'
    # },
    
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
# - Cubic spline interpolation is used (same as MATLAB epstable)  # ← 수정
# - Wavelengths should be in ascending order

# ============================================================================
# MEDIUM (ENVIRONMENT)
# ============================================================================
# The medium surrounding the nanoparticles
# Options:
#   - Built-in: 'air', 'water', 'vacuum'
#   - Custom constant: {'type': 'constant', 'epsilon': 1.77}
#   - Custom from file: {'type': 'table', 'file': 'medium.dat'}

args['medium'] = 'water'

# Example: Nanoparticles in water
# args['medium'] = 'water'

# Example: Custom medium (e.g., oil with n=1.33)
# args['medium'] = {'type': 'constant', 'epsilon': 1.33**2}  # epsilon = n^2

# ============================================================================
# PARTICLE MATERIALS
# ============================================================================
# Materials for the nanoparticle(s)
# Order depends on structure type

# For single particle: [particle_material]
args['materials'] = ['gold']

# For core-shell: [shell_material, core_material]
# args['materials'] = ['silver', 'gold']  # Silver shell, gold core

# For multi-shell: defined in args['layers'] above

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

# Example 1: Simple gold sphere in water
# args['structure'] = 'sphere'
# args['diameter'] = 50
# args['mesh_density'] = 144
# args['medium'] = 'water'
# args['materials'] = ['gold']

# Example 2: Silver nanocube
# args['structure'] = 'cube'
# args['size'] = 40
# args['rounding'] = 0.25
# args['mesh_density'] = 12
# args['medium'] = 'air'
# args['materials'] = ['silver']

# Example 3: Core-shell nanoparticle (Au@Ag)
# args['structure'] = 'core_shell_sphere'
# args['core_diameter'] = 40
# args['shell_thickness'] = 10
# args['mesh_density'] = 144
# args['medium'] = 'air'
# args['materials'] = ['silver', 'gold']  # [shell, core]

# Example 4: Simple dimer
# args['structure'] = 'dimer_sphere'
# args['diameter'] = 50
# args['gap'] = 5
# args['mesh_density'] = 144
# args['medium'] = 'air'
# args['materials'] = ['gold']

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