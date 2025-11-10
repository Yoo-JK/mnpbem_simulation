import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = 'my_structure'

# ============================================================================
# STRUCTURE TYPE - Choose ONE
# ============================================================================

# --- OPTION 1: Built-in MNPBEM Structures ---
# Uncomment to use predefined shapes

args['structure'] = 'sphere'
args['mesh_density'] = 144
args['diameter'] = 50  # nm

# Other built-in options:
# args['structure'] = 'cube'
# args['size'] = 40
# args['rounding'] = 0.25
# args['mesh_density'] = 12

# args['structure'] = 'core_shell_sphere'
# args['core_diameter'] = 40
# args['shell_thickness'] = 10
# args['mesh_density'] = 144

# --- OPTION 2: DDA Shape File ---
# Uncomment to use DDA .shape file (with material indices)

# args['structure'] = 'from_shape'
# args['shape_file'] = os.path.join(Path.home(), 'dataset/mnpbem/str.shape')
# args['voxel_size'] = 2.0  # nm
# args['voxel_method'] = 'surface'  # 'surface' or 'cube'

# ============================================================================
# MATERIALS
# ============================================================================

# Medium (surrounding environment)
args['medium'] = 'air'

# Particle materials
# The order here maps to mat_type indices in .shape file:
#   materials[0] -> mat_type 1 
#   materials[1] -> mat_type 2

args['materials'] = ['gold']

# Example for DDA with multiple materials:
# args['materials'] = ['gold', 'silver', 'sio2']
# → mat_idx 1 = gold, mat_idx 2 = silver, mat_idx 3 = sio2

# ============================================================================
# OPTIONAL: Custom Refractive Index
# ============================================================================

args['refractive_index_paths'] = {}

# Use if you want to use custom refractive index data files
# Example:
# args['refractive_index_paths'] = {
#     'gold': './materials/gold_palik.dat', 'sliver': '.materials/silver_jc.dat'
# }

# ============================================================================
# OPTIONAL: Substrate
# ============================================================================

# Uncomment to add a substrate (half-space) below the nanoparticle
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,
# }
# 'position' indicates the z-coordinate of interface (nm)
