"""
MNPBEM Simulation Configuration

This file defines simulation parameters: excitation, wavelengths, 
numerical settings, and output options.
"""

args = {}

# ============================================================================
# SIMULATION NAME (IDENTIFIER)
# ============================================================================
# Give your simulation a descriptive name
args['simulation_name'] = 'vis_nir_spectrum'

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# Simulation method:
#   - 'stat' : Quasistatic approximation (fast, suitable for small particles <50nm)
#   - 'ret'  : Retarded/full Maxwell equations (accurate, for larger particles >50nm)

args['simulation_type'] = 'stat'

# Interpolation method:
#   - 'curv' : Curved boundary elements (more accurate, recommended)
#   - 'flat' : Flat boundary elements (faster, less accurate)

args['interp'] = 'curv'

# Wait bar (progress indicator):
#   - 0 : Off (recommended for batch jobs)
#   - 1 : On (shows progress in MATLAB)

args['waitbar'] = 0

# ============================================================================
# EXCITATION TYPE
# ============================================================================
# Type of excitation:
#   - 'planewave' : Plane wave illumination (most common)
#   - 'dipole'    : Point dipole excitation (for LDOS, decay rates)
#   - 'eels'      : Electron energy loss spectroscopy

args['excitation_type'] = 'planewave'

# --- PLANEWAVE EXCITATION ---
# Define polarization and propagation directions

# Polarization directions (list of [x, y, z] unit vectors)
# Common configurations:
#   [[1,0,0]]                    : x-polarization only
#   [[0,1,0]]                    : y-polarization only
#   [[1,0,0], [0,1,0]]          : x and y polarizations
#   [[1,0,0], [0,1,0], [0,0,1]] : all three directions

args['polarizations'] = [
    [1, 0, 0],  # x-polarization (e.g., along dimer axis)
    [0, 1, 0],  # y-polarization (perpendicular)
    [0, 0, 1],  # z-polarization (perpendicular)
]

# Propagation directions (list of [x, y, z] unit vectors, one per polarization)
# Must have same length as polarizations list
# Common: [0, 0, 1] for normal incidence from above

args['propagation_dirs'] = [
    [0, 0, 1],  # z-direction for x-pol
    [0, 0, 1],  # z-direction for y-pol
    [1, 0, 0],  # x-direction for z-pol (different angle)
]

# Example: Normal incidence, x-pol only
# args['polarizations'] = [[1, 0, 0]]
# args['propagation_dirs'] = [[0, 0, 1]]

# Example: Angular dependence (45° incidence)
# args['polarizations'] = [[1, 0, 0]]
# args['propagation_dirs'] = [[0, 0.707, 0.707]]  # 45° from normal

# --- DIPOLE EXCITATION ---
# (Only used if excitation_type = 'dipole')
# args['dipole_position'] = [0, 0, 15]  # [x, y, z] position in nm
# args['dipole_moment'] = [1, 0, 0]     # [x, y, z] dipole moment direction

# Example: Dipole 10nm above particle
# args['dipole_position'] = [0, 0, 10]
# args['dipole_moment'] = [0, 0, 1]  # z-oriented dipole

# --- EELS EXCITATION ---
# (Only used if excitation_type = 'eels')
# args['impact_parameter'] = [10, 0]  # [x, y] impact parameter in nm
# args['beam_energy'] = 200e3         # Electron beam energy in eV (e.g., 200 keV)
# args['beam_width'] = 0.2            # Beam width in nm

# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Define the wavelength range for spectrum calculation

# Format: [min, max, num_points] in nanometers
args['wavelength_range'] = [400, 800, 80]  # 400-800nm, 80 points

# Examples:
# args['wavelength_range'] = [300, 1000, 140]  # UV to near-IR
# args['wavelength_range'] = [500, 700, 40]    # Visible only
# args['wavelength_range'] = [700, 1500, 80]   # Near-IR

# Alternative: Specify exact wavelengths (uncomment to use)
# args['wavelengths'] = [400, 450, 500, 550, 600, 650, 700, 750, 800]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output directory (relative or absolute path)
args['output_dir'] = './results'

# Output file prefix (will be prepended to all output files)
args['output_prefix'] = 'simulation'

# Save MATLAB script? (for debugging or manual execution)
args['save_matlab_script'] = True

# Output formats for numerical data
# Options: 'txt', 'csv', 'json', 'mat'
args['output_formats'] = ['txt', 'csv', 'json']

# Generate plots?
args['save_plots'] = True

# Plot formats
# Options: 'png', 'pdf', 'svg', 'eps'
args['plot_formats'] = ['png', 'pdf']

# Plot resolution (DPI) for raster formats (png, jpg)
args['plot_dpi'] = 300

# ============================================================================
# POSTPROCESSING OPTIONS
# ============================================================================

# Calculate absorption cross-section?
# Note: absorption = extinction - scattering
args['calculate_absorption'] = True

# Calculate electromagnetic field enhancement?
# Warning: This is time-consuming, only enable if needed
args['calculate_fields'] = False

# Field calculation region (only used if calculate_fields=True)
args['field_region'] = {
    'x_range': [-60, 60, 121],  # [min, max, num_points] in nm
    'y_range': [-60, 60, 121],
    'z_range': [0, 0, 1],        # Single plane at z=0
}

# Example: Calculate fields in xz-plane
# args['field_region'] = {
#     'x_range': [-100, 100, 201],
#     'y_range': [0, 0, 1],      # Single slice at y=0
#     'z_range': [-50, 50, 101],
# }

# Peak detection in spectrum
args['peak_detection'] = {
    'enabled': True,
    'prominence': 0.1,  # Minimum peak prominence (0-1)
}

# ============================================================================
# ADVANCED NUMERICAL OPTIONS
# ============================================================================

# Refine parameter (integration accuracy)
# Controls the number of integration points for boundary elements
# Higher = more accurate but slower
# Recommended: 1-3 for most cases, increase for very small gaps or high accuracy
args['refine'] = 2

# Relative cutoff (integration distance)
# Determines which boundary elements need refined integration
# Default: 2
# Increase if accuracy issues occur
args['relcutoff'] = 2

# Mirror symmetry (only if structure has perfect mirror symmetry)
# Can significantly speed up calculations
# Options: False, 'x', 'y', 'xy'
# Warning: Use only if structure AND excitation preserve symmetry!
args['use_mirror_symmetry'] = False

# Example: Use x-symmetry for symmetric dimer with x-polarization
# args['use_mirror_symmetry'] = 'x'

# Iterative solver (for very large structures with >10,000 elements)
# Uses less memory but may be slower
# Enable if you encounter out-of-memory errors
args['use_iterative_solver'] = False

# Nonlocal effects (advanced, for very small particles <5nm)
# Includes quantum effects at metal surfaces
# Requires additional setup
args['use_nonlocality'] = False

# ============================================================================
# MATLAB SETTINGS (ADVANCED)
# ============================================================================

# MATLAB executable path
# Options:
#   - 'matlab' : Use system default
#   - '/path/to/matlab' : Use specific installation

args['matlab_executable'] = 'matlab'

# MATLAB command-line options
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'

# ============================================================================
# ADDITIONAL SIMULATION EXAMPLES
# ============================================================================

# Example 1: Single wavelength field calculation
# args['wavelength_range'] = [550, 550, 1]  # Single wavelength
# args['calculate_fields'] = True
# args['field_region'] = {
#     'x_range': [-100, 100, 201],
#     'y_range': [-100, 100, 201],
#     'z_range': [0, 0, 1]
# }

# Example 2: Broadband spectrum (UV to NIR)
# args['wavelength_range'] = [300, 1500, 240]
# args['simulation_type'] = 'ret'  # Use retarded for broad range

# Example 3: Angle-resolved measurements
# args['polarizations'] = [[1, 0, 0]] * 37  # Same polarization
# # Generate angles from 0° to 90°
# import numpy as np
# angles = np.linspace(0, 90, 37)
# args['propagation_dirs'] = [
#     [0, np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))] 
#     for a in angles
# ]

# Example 4: Dipole emission study
# args['excitation_type'] = 'dipole'
# args['dipole_position'] = [0, 0, 10]
# args['dipole_moment'] = [0, 0, 1]
# args['wavelength_range'] = [400, 800, 80]

# Example 5: EELS line scan
# args['excitation_type'] = 'eels'
# args['beam_energy'] = 200e3
# args['beam_width'] = 0.2
# # For line scan, vary impact_parameter in a loop (not shown here)
# args['impact_parameter'] = [10, 0]

# Example 6: High accuracy calculation
# args['refine'] = 3
# args['relcutoff'] = 3
# args['mesh_density'] = 288  # Double standard density
# args['simulation_type'] = 'ret'