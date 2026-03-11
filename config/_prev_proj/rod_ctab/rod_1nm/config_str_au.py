import os
from pathlib import Path

args = {}
args['structure_name'] = 'Au_vac/olmon'

args['structure'] = 'rod'
args['diameter'] = 20.0  # nm
args['height'] = 60  # nm (along z-axis)
args['nphi'] = 3
args['ntheta'] = 3  
args['nz'] = 3
args['materials'] = ['gold_olmon']
args['medium'] = {'type': 'constant', 'epsilon': 1}

args['refractive_index_paths'] = {
    'gold_olmon': os.path.join(Path.home(), 'dataset/mnpbem/refrac/gold_olmon.dat'),
    'ctab': {'type': 'constant', 'epsilon': 1.44**2}
}

args['use_substrate'] = True
args['substrate'] = {
    'material': {'type': 'constant', 'epsilon': 1.52**2},
    'position': -11.0,  # z-coordinate of interface (nm)
}

