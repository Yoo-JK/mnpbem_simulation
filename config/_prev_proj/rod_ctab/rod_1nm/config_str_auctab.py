import os
from pathlib import Path

args = {}
args['structure_name'] = 'AuNR'
args['structure'] = 'core_shell_rod'
args['core_diameter'] = 20  # nm
args['shell_thickness'] = 3.5  # nm (total diameter = 25nm)
args['height'] = 67  # nm (along z-axis)
args['nphi'] = 3
args['ntheta'] = 3           
args['nz'] = 3
args['materials'] = ['gold_olmon', 'ctab']  # [core, shell]
args['medium'] = {'type': 'constant', 'epsilon': 1}

args['refractive_index_paths'] = {
    'gold_olmon': os.path.join(Path.home(), 'dataset/mnpbem/refrac/gold_olmon.dat'),
    'ctab': {'type': 'constant', 'epsilon': 1.44**2}
}

args['use_substrate'] = True
args['substrate'] = {
    'material': {'type': 'constant', 'epsilon': 1.52**2},
    'position': -14.5,  # z-coordinate of interface (nm)
}

