import os
from pathlib import Path

args = {}

# MNPBEM 경로 (필수)
args['mnpbem_path'] = '/home/yoojk20/workspace/MNPBEM'

args['simulation_name'] = 'au_sphere_spectrum'
args['simulation_type'] = 'stat'
args['interp'] = 'curv'
args['waitbar'] = 0
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0]]
args['propagation_dirs'] = [[0, 0, 1]]
args['wavelength_range'] = [400, 800, 100]
args['refine'] = 2
args['relcutoff'] = 2
args['calculate_cross_sections'] = True
args['calculate_fields'] = False
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/sphere_test')
args['save_format'] = ['txt', 'mat', 'csv']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'