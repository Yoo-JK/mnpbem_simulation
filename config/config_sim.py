"""
Simulation Configuration for Voxel Structure

This config defines the electromagnetic simulation parameters
for extinction/scattering spectra calculation.
"""
import os
from pathlib import Path

args = {}

# ============================================================================
# SIMULATION NAME
# ============================================================================
args['simulation_name'] = 'voxel_extinction_spectrum'

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/test')

# ============================================================================
# MNPBEM PATH (MATLAB Toolbox)
# ============================================================================
# UPDATE THIS to your actual MNPBEM installation path
args['mnpbem_path'] = '/home/yoojk20/workspace/MNPBEM'

# Example paths:
# args['mnpbem_path'] = '/home/user/MNPBEM17'
# args['mnpbem_path'] = '/opt/MNPBEM17'

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# 'stat' = Quasistatic approximation (fast, for small particles << wavelength)
# 'ret'  = Full retarded solution (accurate for all sizes)
args['simulation_type'] = 'ret'

# ============================================================================
# EXCITATION TYPE
# ============================================================================
args['excitation_type'] = 'planewave'

# Plane wave parameters
# Polarization: [x, y, z] components
# For x-polarized light: [1, 0, 0]
# For y-polarized light: [0, 1, 0]
# For z-polarized light: [0, 0, 1]
# For circular polarization: [1, 1j, 0] (need to modify code for complex)
args['polarizations'] = [
    [1, 0, 0],  # x-polarized
]

# Propagation direction: [x, y, z] vector
# For light propagating along +z: [0, 0, 1]
# For light propagating along -z: [0, 0, -1]
args['propagation_dirs'] = [
    [0, 0, 1],  # propagating along +z axis
]

# To scan multiple angles, add more entries:
# args['polarizations'] = [[1, 0, 0]] * 10
# import numpy as np
# angles = np.linspace(0, 90, 10)
# args['propagation_dirs'] = [
#     [0, np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))] 
#     for a in angles
# ]

# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Format: [start_nm, end_nm, num_points]
# Visible range (400-800 nm) with 80 points
args['wavelength_range'] = [400, 800, 80]

# Other common ranges:
# UV-Vis-NIR: [300, 1500, 240]
# Visible only: [400, 700, 60]
# NIR: [700, 1500, 160]

# ============================================================================
# ACCURACY SETTINGS
# ============================================================================

# Mesh refinement level (1-3)
# Higher = more accurate but slower
# 1: Coarse (fast)
# 2: Medium (recommended)
# 3: Fine (accurate but slow)
args['refine'] = 2

# Relative cutoff for field calculations
# Higher = more accurate but slower
# Typical values: 2-4
args['relcutoff'] = 2

# ============================================================================
# FIELD CALCULATION (OPTIONAL)
# ============================================================================

# Set to True to calculate electromagnetic field enhancement
args['calculate_fields'] = False

# If calculate_fields = True, define the field calculation region:
# args['calculate_fields'] = True
# args['field_region'] = {
#     'x_range': [-100, 100, 101],  # [min, max, num_points] in nm
#     'y_range': [-100, 100, 101],
#     'z_range': [0, 0, 1]  # Single z-plane at z=0
# }

# For a single plane (faster):
# args['field_region'] = {
#     'x_range': [-100, 100, 201],
#     'y_range': [-100, 100, 201],
#     'z_range': [0, 0, 1]  # z=0 plane only
# }

# For 3D volume (slow):
# args['field_region'] = {
#     'x_range': [-100, 100, 51],
#     'y_range': [-100, 100, 51],
#     'z_range': [-100, 100, 51]
# }

# ============================================================================
# VISUALIZATION
# ============================================================================

# Save figures
args['save_figures'] = True

# Figure format: 'png', 'pdf', or 'both'
args['figure_format'] = 'both'

# Figure resolution (DPI)
args['dpi'] = 300

# ============================================================================
# ADVANCED OPTIONS (Usually don't need to change)
# ============================================================================

# Interpolation method
# args['interp'] = 'curv'  # Default: 'curv'

# Show waitbar during MATLAB execution
# args['waitbar'] = 0  # 0=off, 1=on

# ============================================================================
# ALTERNATIVE EXCITATION TYPES
# ============================================================================

# --- Dipole Excitation ---
# Uncomment to use dipole source instead of plane wave
# args['excitation_type'] = 'dipole'
# args['dipole_position'] = [0, 0, 10]  # Position in nm
# args['dipole_moment'] = [0, 0, 1]     # Moment direction [x, y, z]

# --- EELS (Electron Energy Loss Spectroscopy) ---
# Uncomment to simulate EELS
# args['excitation_type'] = 'eels'
# args['beam_energy'] = 200e3  # Electron beam energy in eV (e.g., 200 keV)
# args['beam_width'] = 0.2     # Beam width in nm
# args['impact_parameter'] = [10, 0]  # Impact parameter [b, phi] in nm
