"""
Test configuration for gold nanosphere surface charge visualization
"""
import os
from pathlib import Path

args = {}

# Structure
args['structure_name'] = 'test_gold_sphere'
args['structure'] = 'sphere'
args['diameter'] = 50  # nm
args['mesh_density'] = 144  # Standard for spheres

# Material
args['materials'] = ['gold']
args['medium'] = 'water'
