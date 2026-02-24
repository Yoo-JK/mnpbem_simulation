import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au90Cu10_vac/johnson_christy'
args['structure'] = 'core_shell_rod'
args['core_diameter'] = 9  # nm
args['shell_thickness'] = 2  # nm (total diameter = 25nm)
args['height'] = 37  # nm (along z-axis)
args['nphi'] = 3
args['ntheta'] = 3
args['nz'] = 3

args['materials'] = ['gold', 'Au90Cu10']  # [core, shell]
args['medium'] = 'air'
args['refractive_index_paths'] = {
        'au90cu10': os.path.join(Path.home(), 'dataset/mnpbem/refrac/Au90Cu10.dat')
        }
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }

args['use_substrate'] = True
args['substrate'] = {
    'material': 'glass',  # or 'silicon', custom dict
    'position': -6.501,  # z-coordinate of interface (nm)
}

