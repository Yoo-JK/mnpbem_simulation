"""
Gold Sphere Simulation Test Configuration
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# MNPBEM TOOLBOX PATH
# ============================================================================
args['mnpbem_path'] = '/home/yoojk20/workspace/MNPBEM'

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
args['simulation_name'] = 'au_sphere_spectrum'
args['simulation_type'] = 'stat'
args['interp'] = 'curv'
args['waitbar'] = 0

# ============================================================================
# EXCITATION
# ============================================================================
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0], [0, 1, 0]]  # x and y polarization
args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]  # Both propagating in +z

# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
args['wavelength_range'] = [400, 800, 100]  # 400-800 nm, 100 points

# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================
args['refine'] = 3
args['relcutoff'] = 3

# ============================================================================
# CALCULATION OPTIONS
# ============================================================================
args['calculate_cross_sections'] = True
args['calculate_fields'] = True

# ============================================================================
# FIELD CALCULATION
# ============================================================================
args['field_region'] = {
    'x_range': [-80, 80, 161],  # -80 to 80 nm, 161 points
    'y_range': [0, 0, 1],       # xz-plane at y=0
    'z_range': [-80, 80, 161]   # -80 to 80 nm, 161 points
}

args['field_mindist'] = 0.5     # Minimum distance from particle surface (nm)
args['field_nmax'] = 2000       # Portion size for large grids
args['field_wavelength_idx'] = 'middle'  # Calculate at middle wavelength

# ============================================================================
# FIELD ANALYSIS OPTIONS (NEW)
# ============================================================================
args['export_field_arrays'] = False  # Don't export arrays to JSON (saves space)
args['field_hotspot_count'] = 10     # Number of hotspots to identify
args['field_hotspot_min_distance'] = 3  # Minimum distance between hotspots (grid points)

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/sphere_test')
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False

# ============================================================================
# MATLAB SETTINGS
# ============================================================================
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'