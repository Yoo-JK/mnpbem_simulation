import os
from pathlib import Path

args = {}
args['mnpbem_path'] = os.path.join(Path.home(), 'workspace/MNPBEM')

args['simulation_name'] = 'johnson/Au97Cu03'
args['simulation_type'] = 'ret'
args['interp'] = 'curv'

args['waitbar'] = 1

args['excitation_type'] = 'planewave'
args['polarizations'] = [
    [0, 0, 1],
    [0, 1, 0],
]
args['propagation_dirs'] = [
    [1, 0, 0],
    [1, 0, 0],
]

args['wavelength_range'] = [400, 800, 100]  # 400-800 nm, 100 points
args['refine'] = 3
args['relcutoff'] = 3
args['calculate_cross_sections'] = True

args['calculate_fields'] = True
args['field_region'] = {
    'x_range': [-80, 80, 161],  # [min, max, num_points] in nm
    'y_range': [0, 0, 1],       # xz-plane at y=0
    'z_range': [-80, 80, 161]   # [min, max, num_points] in nm
}
args['field_mindist'] = 0.5     # Minimum distance from particle surface (nm)
args['field_nmax'] = 2000       # Work off calculation in portions (for large grids)
args['field_wavelength_idx'] = 'middle'  # Which wavelength to calculate fields: 'middle', 'peak', or integer index
args['export_field_arrays'] = False  # Exports downsampled field arrays to JSON

args['field_hotspot_count'] = 10  # Number of hotspots to identify
args['field_hotspot_min_distance'] = 3  # Minimum distance between hotspots (grid points)
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/rod')

args['output_formats'] = ['txt']
args['save_plots'] = True
args['plot_format'] = ['png']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'energy'
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'
