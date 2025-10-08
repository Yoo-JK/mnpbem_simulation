"""
MNPBEM Simulation Configuration File

This file contains all parameters needed for the MNPBEM simulation.
Modify the values in the args dictionary to customize your simulation.
"""

args = {}

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# Simulation method: 'stat' (quasistatic) or 'ret' (retarded/full Maxwell)
# - 'stat': Fast, suitable for small particles (<50nm)
# - 'ret': Accurate, suitable for larger particles (>50nm)
args['simulation_type'] = 'stat'

# Interpolation: 'flat' or 'curv'
# - 'curv': Curved boundary (more accurate, recommended)
args['interp'] = 'curv'

# Wait bar: 0 (off) or 1 (on)
args['waitbar'] = 0


# ============================================================================
# GEOMETRY / STRUCTURE
# ============================================================================
# Structure type: 
# - 'sphere': Single sphere
# - 'cube': Single cube
# - 'rod': Rod/cylinder
# - 'ellipsoid': Ellipsoid
# - 'triangle': Triangle (2D extruded)
# - 'dimer_sphere': Two spheres
# - 'dimer_cube': Two cubes
# - 'core_shell_sphere': Core-shell sphere
# - 'core_shell_cube': Core-shell cube
# - 'dimer_core_shell_cube': Two core-shell cubes
args['structure'] = 'dimer_core_shell_cube'

# Mesh density (higher = more accurate but slower)
# Recommended: 12-16 for most cases
args['mesh_density'] = 12

# ============================================================================
# GEOMETRY PARAMETERS (structure-specific)
# ============================================================================

# --- For 'sphere' ---
# args['diameter'] = 10  # nm

# --- For 'cube' ---
# args['size'] = 20  # nm
# args['rounding'] = 0.25  # Edge rounding (0-1, smaller = more rounded)

# --- For 'rod' ---
# args['diameter'] = 10  # nm
# args['height'] = 50  # nm

# --- For 'ellipsoid' ---
# args['axes'] = [10, 15, 20]  # [x, y, z] axis lengths

# --- For 'triangle' ---
# args['side_length'] = 30  # nm
# args['thickness'] = 5  # nm

# --- For 'dimer_sphere' ---
# args['diameter'] = 10  # nm
# args['gap'] = 5  # Gap distance between spheres

# --- For 'dimer_cube' ---
# args['size'] = 20  # nm
# args['gap'] = 10  # nm
# args['rounding'] = 0.25

# --- For 'core_shell_sphere' ---
# args['core_diameter'] = 10  # nm
# args['shell_thickness'] = 5  # nm

# --- For 'core_shell_cube' ---
# args['core_size'] = 15  # nm
# args['shell_thickness'] = 5  # nm
# args['rounding'] = 0.25

# --- For 'dimer_core_shell_cube' (ACTIVE) ---
args['core_size'] = 20  # nm
args['shell_thickness'] = 5  # nm
args['gap'] = 10  # Gap between the two cubes
args['rounding'] = 0.25  # Edge rounding parameter


# ============================================================================
# MATERIALS (NEW SIMPLIFIED SYSTEM)
# ============================================================================

# --- Medium (surrounding environment) ---
# The medium in which the particle is embedded
args['medium'] = 'air'
# Options: 'air', 'water', 'vacuum'
# Custom: {'type': 'constant', 'epsilon': 1.77}

# --- Particle Materials ---
# List of materials for the particle structure (from outside to inside)
# For single particle: ['material']
# For core-shell: ['shell_material', 'core_material']
# For dimer core-shell: ['shell_material', 'core_material']
args['materials'] = [
    'silver',  # Shell material
    'gold'     # Core material
]

# Available built-in materials:
# - 'air', 'water', 'vacuum'
# - 'glass', 'silicon', 'sapphire', 'sio2'
# - 'gold', 'silver', 'aluminum'

# --- Custom Materials ---
# Three ways to define custom materials:
#
# 1. Constant dielectric function:
#    {'type': 'constant', 'epsilon': 2.5}
#
# 2. From data file (wavelength-dependent):
#    {'type': 'table', 'file': 'custom_material.dat'}
#
# 3. Custom function (advanced):
#    {'type': 'function', 'formula': '1 - 3.3^2/(w*(w+1i*0.165))', 'unit': 'eV'}
#
# Example with custom materials:
# args['medium'] = {'type': 'constant', 'epsilon': 1.77}
# args['materials'] = [
#     {'type': 'table', 'file': 'my_metal.dat'},
#     'gold'
# ]


# ============================================================================
# SUBSTRATE CONFIGURATION (OPTIONAL)
# ============================================================================
# Use substrate? (True/False)
args['use_substrate'] = False

# If use_substrate = True, define substrate properties:
# args['substrate'] = {
#     'material': 'glass',        # Substrate material
#     'position': 0,              # z-position of interface (nm)
#     'particle_height': 5        # Height of particle above substrate (nm)
# }

# Supported substrate materials: 'glass', 'silicon', 'sapphire', 'sio2'
# Or custom: {'type': 'constant', 'epsilon': 2.25}

# Example configurations:
#
# --- Gold nanosphere on glass substrate ---
# args['use_substrate'] = True
# args['medium'] = 'air'
# args['materials'] = ['gold']
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,
#     'particle_height': 5
# }
#
# --- Silver cube on silicon ---
# args['use_substrate'] = True
# args['medium'] = 'air'
# args['materials'] = ['silver']
# args['substrate'] = {
#     'material': 'silicon',
#     'position': 0,
#     'particle_height': 2
# }


# ============================================================================
# EXCITATION
# ============================================================================
# Excitation type: 'planewave', 'dipole', 'eels'
args['excitation_type'] = 'planewave'

# --- For 'planewave' ---
# Polarization directions: list of [x, y, z] vectors
args['polarizations'] = [
    [1, 0, 0],  # x-polarization (along dimer axis)
    [0, 1, 0],  # y-polarization (perpendicular)
    [0, 0, 1]   # z-polarization (perpendicular)
]

# Propagation directions: list of [x, y, z] vectors (one per polarization)
args['propagation_dirs'] = [
    [0, 0, 1],  # z-direction for x-pol
    [0, 0, 1],  # z-direction for y-pol
    [1, 0, 0]   # x-direction for z-pol
]

# --- For 'dipole' ---
# args['dipole_position'] = [0, 0, 15]  # [x, y, z] position
# args['dipole_moment'] = [1, 0, 0]  # [x, y, z] dipole moment direction

# --- For 'eels' ---
# args['impact_parameter'] = [10, 0]  # [x, y] impact parameter
# args['beam_energy'] = 200e3  # Electron beam energy (eV)
# args['beam_width'] = 0.2  # Beam width


# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Wavelength range for spectrum calculation
# Format: [min, max, num_points] in nanometers
args['wavelength_range'] = [400, 800, 80]  # 400-800nm, 80 points

# Alternative: specific wavelengths
# args['wavelengths'] = [400, 450, 500, 550, 600, 650, 700, 750, 800]


# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
# Output directory
args['output_dir'] = './results'

# Output file prefix
args['output_prefix'] = 'simulation'

# Save MATLAB script? (True/False)
args['save_matlab_script'] = True

# Output formats: list from ['txt', 'csv', 'json', 'mat']
args['output_formats'] = ['txt', 'csv', 'json']

# Save plots? (True/False)
args['save_plots'] = True

# Plot formats: list from ['png', 'pdf', 'svg']
args['plot_formats'] = ['png', 'pdf']

# Plot DPI (for raster formats)
args['plot_dpi'] = 300


# ============================================================================
# POSTPROCESSING OPTIONS
# ============================================================================
# Calculate absorption? (True/False)
# Note: absorption = extinction - scattering
args['calculate_absorption'] = True

# Calculate field enhancement? (True/False)
# Warning: This can be time-consuming
args['calculate_fields'] = False

# Field calculation region (if calculate_fields = True)
# args['field_region'] = {
#     'x_range': [-50, 50, 101],  # [min, max, num_points]
#     'y_range': [-50, 50, 101],
#     'z_range': [0, 0, 1]  # Single plane at z=0
# }


# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
# Refine parameter (for integration accuracy)
# Higher = more accurate but slower
# Default: 1, Recommended: 1-3
args['refine'] = 2

# Mirror symmetry (True/False)
# Use only if structure has mirror symmetry for speed up
args['use_mirror_symmetry'] = False

# Iterative solver (for very large structures)
# Only use if memory issues occur
args['use_iterative_solver'] = False

# Nonlocal effects (advanced)
args['use_nonlocality'] = False


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================
# 
# 1. Gold nanosphere in water:
#    args['structure'] = 'sphere'
#    args['diameter'] = 50
#    args['medium'] = 'water'
#    args['materials'] = ['gold']
#    args['simulation_type'] = 'ret'
#
# 2. Silver nanocube on glass substrate:
#    args['structure'] = 'cube'
#    args['size'] = 60
#    args['medium'] = 'air'
#    args['materials'] = ['silver']
#    args['use_substrate'] = True
#    args['substrate'] = {'material': 'glass', 'position': 0, 'particle_height': 5}
#
# 3. Gold-silver core-shell nanoparticle:
#    args['structure'] = 'core_shell_sphere'
#    args['core_diameter'] = 30
#    args['shell_thickness'] = 10
#    args['medium'] = 'air'
#    args['materials'] = ['silver', 'gold']  # [shell, core]
#
# 4. Dimer with custom material:
#    args['structure'] = 'dimer_sphere'
#    args['diameter'] = 40
#    args['gap'] = 5
#    args['medium'] = {'type': 'constant', 'epsilon': 1.77}
#    args['materials'] = [{'type': 'table', 'file': 'my_metal.dat'}]
#


# ============================================================================
# EXCITATION
# ============================================================================
# Excitation type: 'planewave', 'dipole', 'eels'
args['excitation_type'] = 'planewave'

# --- For 'planewave' ---
# Polarization directions: list of [x, y, z] vectors
# Common options:
# - [[1,0,0]]: x-polarization only
# - [[0,1,0]]: y-polarization only
# - [[1,0,0], [0,1,0], [0,0,1]]: all three directions
args['polarizations'] = [
    [1, 0, 0],  # x-polarization (along dimer axis)
    [0, 1, 0],  # y-polarization (perpendicular)
    [0, 0, 1]   # z-polarization (perpendicular)
]

# Propagation directions: list of [x, y, z] vectors (one per polarization)
args['propagation_dirs'] = [
    [0, 0, 1],  # z-direction for x-pol
    [0, 0, 1],  # z-direction for y-pol
    [1, 0, 0]   # x-direction for z-pol
]

# --- For 'dipole' ---
# args['dipole_position'] = [0, 0, 15]  # [x, y, z] position
# args['dipole_moment'] = [1, 0, 0]  # [x, y, z] dipole moment direction

# --- For 'eels' ---
# args['impact_parameter'] = [10, 0]  # [x, y] impact parameter
# args['beam_energy'] = 200e3  # Electron beam energy (eV)
# args['beam_width'] = 0.2  # Beam width


# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Wavelength range for spectrum calculation
# Format: [min, max, num_points] in nanometers
args['wavelength_range'] = [400, 800, 80]  # 400-800nm, 80 points

# Alternative: specific wavelengths
# args['wavelengths'] = [400, 450, 500, 550, 600, 650, 700, 750, 800]


# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
# Output directory
args['output_dir'] = './results'

# Output file prefix
args['output_prefix'] = 'simulation'

# Save MATLAB script? (True/False)
args['save_matlab_script'] = True

# Output formats: list from ['txt', 'csv', 'json', 'mat']
args['output_formats'] = ['txt', 'csv', 'json']

# Save plots? (True/False)
args['save_plots'] = True

# Plot formats: list from ['png', 'pdf', 'svg']
args['plot_formats'] = ['png', 'pdf']

# Plot DPI (for raster formats)
args['plot_dpi'] = 300


# ============================================================================
# POSTPROCESSING OPTIONS
# ============================================================================
# Calculate absorption? (True/False)
# Note: absorption = extinction - scattering
args['calculate_absorption'] = True

# Calculate field enhancement? (True/False)
# Warning: This can be time-consuming
args['calculate_fields'] = False

# Field calculation region (if calculate_fields = True)
# args['field_region'] = {
#     'x_range': [-50, 50, 101],  # [min, max, num_points]
#     'y_range': [-50, 50, 101],
#     'z_range': [0, 0, 1]  # Single plane at z=0
# }


# ============================================================================
# SUBSTRATE / LAYER STRUCTURE
# ============================================================================
# Use substrate? (True/False)
args['use_substrate'] = False

# --- Substrate configuration (if use_substrate = True) ---
# args['substrate'] = {
#     'material': 'glass',        # Substrate material: 'glass', 'silicon', 'sapphire'
#     'position': 0,              # z-position of interface (nm)
#     'particle_height': 5        # Height of particle above substrate (nm)
# }

# Example: Gold nanosphere on glass substrate
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,
#     'particle_height': 5
# }

# --- Multi-layer structure (advanced) ---
# args['use_multilayer'] = False
# args['multilayer'] = {
#     'materials': ['air', 'thin_layer', 'substrate'],  # Top to bottom
#     'positions': [10, 0],       # Interface positions (nm)
#     'particle_layer': 1         # Which layer contains the particle (0-indexed)
# }


# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
# Refine parameter (for integration accuracy)
# Higher = more accurate but slower
# Default: 1, Recommended: 1-3
args['refine'] = 2

# Mirror symmetry (True/False)
# Use only if structure has mirror symmetry for speed up
args['use_mirror_symmetry'] = False

# Iterative solver (for very large structures)
# Only use if memory issues occur
args['use_iterative_solver'] = False

# Nonlocal effects (advanced)
args['use_nonlocality'] = False


# ============================================================================
# ADDITIONAL NOTES
# ============================================================================
# 
# Example configurations for common scenarios:
#
# 1. Gold nanosphere in water:
#    args['structure'] = 'sphere'
#    args['diameter'] = 50
#    args['materials'] = ['water', 'gold']
#    args['simulation_type'] = 'ret'
#
# 2. Silver nanocube on glass substrate:
#    args['structure'] = 'cube'
#    args['size'] = 60
#    args['materials'] = ['glass', 'silver']
#    Use layer structure (requires additional setup)
#
# 3. Core-shell nanoparticle:
#    args['structure'] = 'core_shell_sphere'
#    args['core_diameter'] = 30
#    args['shell_thickness'] = 10
#    args['materials'] = ['air', 'gold', 'silver']  # medium, shell, core
#
# 4. Bowtie antenna (dimer):
#    args['structure'] = 'dimer_sphere'
#    args['diameter'] = 40
#    args['gap'] = 5
#    args['polarizations'] = [[1, 0, 0]]  # Along dimer axis only
#