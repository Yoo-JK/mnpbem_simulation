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
args['structure_name'] = 'auag_dimer'

args['structure'] = 'connected_dimer_cube'
args['core_size'] = 47
args['shell_layers'] = [1]  # shell_size = 40nm, enables core-shell mode
args['gap'] = -0.1  # shell overlap = 2nm, core_gap = 8nm (separate)
args['materials'] = ['gold', 'silver']  # [core, shell]
args['roundings'] = [0.2, 0.2]
args['mesh_density'] = 2

# ============================================================================
# COMMON SETTINGS (All Structures)
# ============================================================================

# --- Medium (Surrounding Environment) ---
args['medium'] = 'water'
# Options: 'air', 'water', 'vacuum', 'glass'
# OR custom constant: args['medium'] = {'type': 'constant', 'epsilon': 1.77}

# --- Custom Refractive Index Paths (Optional) ---
args['refractive_index_paths'] = {}
# Override built-in material data with custom files
# Example:
args['refractive_index_paths'] = {
    'agcl': {'type': 'constant', 'epsilon': 2.02}
}
# File format: [wavelength(nm), n, k] per line

# --- Substrate (Optional) ---
args['use_substrate'] = False
# Uncomment to add substrate (half-space) below nanoparticle:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',  # or 'silicon', custom dict
#     'position': 0,  # z-coordinate of interface (nm)
# }

