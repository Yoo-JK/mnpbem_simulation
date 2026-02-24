import os
from pathlib import Path

args = {}

args['mnpbem_path'] = os.path.join(Path.home(), 'workspace/MNPBEM')
args['simulation_name'] = 'au_dimer_nonlocal_mirror'

args['simulation_type'] = 'ret'
args['interp'] = 'curv'
args['waitbar'] = 0

args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0]]
args['propagation_dirs'] = [[0, 0, 1]]

args['wavelength_range'] = [500, 1000, 20]

args['refine'] = 3
args['relcutoff'] = 3

args['calculate_cross_sections'] = True
args['calculate_fields'] = False

args['use_nonlocality'] = True
args['use_mirror_symmetry'] = 'yz'

args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/test_nonlocal_mirror')
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png']
args['plot_dpi'] = 150

args['use_parallel'] = True
args['num_workers'] = 2

