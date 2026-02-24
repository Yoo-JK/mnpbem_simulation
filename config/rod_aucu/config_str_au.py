import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au_vac/johnson_christy'

args['structure'] = 'rod'
args['diameter'] = 9.0  # nm
args['height'] = 33  # nm (along z-axis)
args['nphi'] = 3
args['ntheta'] = 3
args['nz'] = 3
args['materials'] = ['gold']
args['medium'] = 'air'
# Options: 'air', 'water', 'vacuum', 'glass'
# OR custom constant: args['medium'] = {'type': 'constant', 'epsilon': 1.77}

args['refractive_index_paths'] = {}
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }

args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',  # or 'silicon', custom dict
    'position': -4.501,  # z-coordinate of interface (nm)
}

