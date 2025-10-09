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

# 변경: save_format → output_formats
args['output_formats'] = ['txt', 'csv', 'json']  # ← 이름만 바꾸면 됩니다!

args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']  # ← 이미 PNG 포함!
args['plot_dpi'] = 300
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False
args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'