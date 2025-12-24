import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au90Cu10_vac/johnson_christy'
args['structure'] = 'core_shell_rod'
args['core_diameter'] = 20  # nm
args['shell_thickness'] = 3.5  # nm (total diameter = 25nm)
args['height'] = 67  # nm (along z-axis)
args['rod_mesh'] = [15, 20, 20]   # [nphi, ntheta, nz]: circumference / caps / length
args['materials'] = ['gold', 'ctab']  # [core, shell]
args['medium'] = 'air'
args['refractive_index_paths'] = {
        'ctab': {'type': 'constant', 'epsilon': 1.42**2}
        }
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }

args['use_substrate'] = False
args['substrate'] = {
    'material': 'glass',  # or 'silicon', custom dict
    'position': -13.501,  # z-coordinate of interface (nm)
}

