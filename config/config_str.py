import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = 'voxel_au_ag_particle'

# ============================================================================
# DDA SHAPE FILE STRUCTURE
# ============================================================================
args['structure'] = 'from_shape'

# Shape file path - UPDATE THIS to your actual file path
args['shape_file'] = os.path.join(Path.home(), 'dataset/mnpbem/model_Au47.0_Ag4.0_AgCl0.0_gap3.0.shape')

# Voxel size in nanometers (physical size of each voxel)
# UPDATE THIS to match your DDA simulation voxel size
args['voxel_size'] = 1.0  # nm

# Voxel conversion method:
# - 'surface' (RECOMMENDED): Fast, extracts only outer surface
# - 'cube': Slow but accurate, each voxel becomes a small cube
args['voxel_method'] = 'surface'

# ============================================================================
# MATERIALS
# ============================================================================

# Medium (surrounding environment)
args['medium'] = 'air'  # Options: 'air', 'water', 'vacuum', 'glass'

# Particle materials
# The order here maps to mat_type indices in .shape file:
#   materials[0] → mat_type 1 (gold in your file)
#   materials[1] → mat_type 2 (silver in your file)
args['materials'] = ['gold', 'silver']

# ============================================================================
# OPTIONAL: Custom Refractive Index
# ============================================================================
# Uncomment if you want to use custom refractive index data files
# args['refractive_index_paths'] = {
#     'gold': './materials/gold_palik.dat',
#     'silver': './materials/silver_jc.dat',
# }

# ============================================================================
# OPTIONAL: Substrate
# ============================================================================
# Uncomment to add a substrate (half-space) below the nanoparticle
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',
#     'position': 0,  # z-coordinate of interface (nm)
# }
