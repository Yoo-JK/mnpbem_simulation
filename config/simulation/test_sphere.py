"""
Test simulation configuration for surface charge visualization
"""
import os
from pathlib import Path

args = {}

# MNPBEM path
args['mnpbem_path'] = os.path.join(Path.home(), 'scratch/bins/MNPBEM')

# Simulation name
args['simulation_name'] = 'test_sphere_surface_charge'

# Simulation type
args['simulation_type'] = 'stat'  # Quasistatic for small sphere
args['interp'] = 'curv'
args['waitbar'] = 0

# Parallel computing (fast test)
args['use_parallel'] = False  # Single core for quick test
args['wavelength_chunk_size'] = None  # No chunking for small test

# Excitation
args['excitation_type'] = 'planewave'
args['polarizations'] = [
    [1, 0, 0],  # x-polarization
    [0, 1, 0],  # y-polarization
]
args['propagation_dirs'] = [
    [0, 0, 1],  # +z direction
    [0, 0, 1],  # +z direction
]

# Wavelength range (small for quick test)
args['wavelength_range'] = [400, 800, 50]  # 50 points

# Accuracy
args['refine'] = 2  # Lower for quick test
args['relcutoff'] = 3

# Field calculation (enables surface charge automatically)
args['calculate_fields'] = True
args['field_wavelength_idx'] = 'peak'  # Peak absorption for each polarization

# Field region (small 2D slice for quick test)
args['field_region'] = {
    'x_range': [-60, 60, 61],  # xz-plane
    'y_range': [0, 0, 1],       # at y=0
    'z_range': [-60, 60, 61]
}
args['field_mindist'] = 0.5
args['field_nmax'] = 2000

# Output
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/test_surface_charge')
args['save_plots'] = True
args['plot_format'] = ['png']  # PNG only for quick test
args['plot_dpi'] = 150  # Lower DPI for quick test
args['output_formats'] = ['txt', 'json']

# MATLAB settings
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'
