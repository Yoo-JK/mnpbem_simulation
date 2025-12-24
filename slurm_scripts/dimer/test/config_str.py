import os
from pathlib import Path

args = {}

args['structure_name'] = 'au_dimer_20nm_gap1nm'

args['structure'] = 'dimer_sphere'
args['diameter'] = 20
args['gap'] = 0.5
args['mesh_density'] = 144
args['materials'] = ['gold']

args['medium'] = 'air'
args['use_substrate'] = False

