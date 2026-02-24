"""
MNPBEM Structure Configuration - Complete Recipe Book

This file contains examples for ALL available structure types.
Uncomment the section you need and customize the parameters.

For detailed documentation, see: config/structure/guide_structure.txt
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = '53x106_03'

# --- Rod (Cylinder) ---
# args['structure'] = 'rod'
# args['diameter'] = 22  # nm
# args['height'] = 47  # nm (along z-axis)
# args['mesh_density'] = 2  # element size in nm (smaller = finer mesh)
# args['materials'] = ['gold']
# Legacy mode (optional): use nphi, ntheta, nz instead of mesh_density
# args['nphi'] = 15    # circumference divisions
# args['ntheta'] = 20  # cap divisions
# args['nz'] = 20      # length divisions

# --- Core-Shell Rod (Nanorod) ---
args['structure'] = 'core_shell_rod'
args['core_diameter'] = 53  # nm
args['shell_thickness'] = 3  # nm (total diameter = 25nm)
args['height'] = 112  # nm (total length along z-axis)
args['mesh_density'] = 2  # element size in nm (smaller = finer mesh)
args['materials'] = ['gold', 'polymer']  # [core, shell]
# Legacy mode (optional): use nphi, ntheta, nz instead of mesh_density
# args['nphi'] = 15    # circumference divisions
# args['ntheta'] = 20  # cap divisions
# args['nz'] = 20      # length divisions

# ============================================================================
# COMMON SETTINGS (All Structures)
# ============================================================================

# --- Medium (Surrounding Environment) ---
args['medium'] = 'water'
# Options: 'air', 'water', 'vacuum', 'glass'
# OR custom constant: args['medium'] = {'type': 'constant', 'epsilon': 1.77}

# --- Custom Refractive Index Paths (Optional) ---
args['refractive_index_paths'] = {
        'polymer': os.path.join(Path.home(), 'dataset/mnpbem/refrac/ho_polymer.txt')
}
# Override built-in material data with custom files
# Example:
# args['refractive_index_paths'] = {
#     'gold': os.path.join(Path.home(), 'materials/gold_palik.dat'),
#     'silver': os.path.join(Path.home(), 'materials/silver_jc.dat')
# }
# File format: [wavelength(nm), n, k] per line

# --- Substrate (Optional) ---
args['use_substrate'] = False
# Uncomment to add substrate (half-space) below nanoparticle:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',  # or 'silicon', custom dict
#     'position': 0,  # z-coordinate of interface (nm)
# }

