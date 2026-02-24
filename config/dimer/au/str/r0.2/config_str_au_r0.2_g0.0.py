import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = 'au_dimer'


args['structure'] = 'connected_dimer_cube'

# --- Core Parameters ---
args['core_size'] = 47  # Size of each cube (nm)
args['gap'] = -0.1  # Surface-to-surface distance (nm)
args['rounding'] = 0.2  # Edge rounding parameter

# --- Materials ---
args['materials'] = ['gold']  # Single material for fused structure

# --- Mesh Density ---
args['mesh_density'] = 2  # element size in nm

# --- Transformations (Particle 2) ---
args['offset'] = [0, 0, 0]  # [x, y, z] shift for particle 2
args['tilt_angle'] = 0  # degrees
args['tilt_axis'] = [1, 0, 0]
args['rotation_angle'] = 0  # degrees (z-axis)

# --- Medium (Surrounding Environment) ---
args['medium'] = 'water'

args['refractive_index_paths'] = {}

# --- Substrate (Optional) ---
args['use_substrate'] = False
# Uncomment to add substrate (half-space) below nanoparticle:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',  # or 'silicon', custom dict
#     'position': 0,  # z-coordinate of interface (nm)
# }
