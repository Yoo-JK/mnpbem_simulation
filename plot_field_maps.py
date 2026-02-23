#!/usr/bin/env python3
"""
Standalone Field Map Visualization for MNPBEM Simulation Results.

Generates 9 individual field map plots (log scale, no percentile clipping):
  - Enhancement |E/E0|:  External, Internal, Merged
  - Intensity |E/E0|^2:  External, Internal, Merged
  - Raw |E|^2:           External, Internal, Merged

Usage:
    python plot_field_maps.py simulation_results.mat --field_entry 15
    python plot_field_maps.py simulation_results.mat --field_entry 15 --config_structure /path/to/config_structure.py
    python plot_field_maps.py simulation_results.mat --field_entry 15 --outdir ./plots
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle


# ============================================================================
# Data Loading
# ============================================================================

def load_field_entry(mat_path, entry_idx):
    """
    Load a specific field entry from the .mat file.

    Parameters
    ----------
    mat_path : str
        Path to simulation_results.mat
    entry_idx : int
        Field entry index (0-based internally, but user specifies 1-based or
        the actual index stored in wavelength_idx)

    Returns
    -------
    dict with keys: wavelength, wavelength_idx, polarization_idx,
        x_grid, y_grid, z_grid, and all available field arrays
    """
    mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if 'results' in mat_data:
        results = mat_data['results']
        fields_struct = results.fields
    elif 'field_data' in mat_data:
        fields_struct = mat_data['field_data']
    else:
        raise KeyError("MAT file has neither 'results.fields' nor 'field_data'")

    # Handle single entry
    if not isinstance(fields_struct, np.ndarray):
        fields_struct = np.array([fields_struct])

    n_entries = len(fields_struct)
    print(f"Total field entries in file: {n_entries}")

    if entry_idx < 0 or entry_idx >= n_entries:
        # Print available entries for user reference
        print("\nAvailable field entries:")
        for i, f in enumerate(fields_struct):
            wl = float(f.wavelength) if hasattr(f, 'wavelength') else '?'
            wl_idx = int(f.wavelength_idx) if hasattr(f, 'wavelength_idx') else '?'
            pol_idx = int(f.polarization_idx) if hasattr(f, 'polarization_idx') else '?'
            print(f"  [{i}] wavelength = {wl} nm, wl_idx = {wl_idx}, pol_idx = {pol_idx}")
        raise IndexError(f"field_entry {entry_idx} out of range [0, {n_entries-1}]")

    field = fields_struct[entry_idx]

    # Extract all available fields
    result = {}
    for attr in ['wavelength', 'wavelength_idx', 'polarization_idx']:
        if hasattr(field, attr):
            result[attr] = float(getattr(field, attr)) if attr == 'wavelength' else int(getattr(field, attr))

    for attr in ['x_grid', 'y_grid', 'z_grid', 'polarization',
                 'enhancement', 'enhancement_ext', 'enhancement_int',
                 'intensity', 'intensity_ext', 'intensity_int',
                 'e_sq', 'e_sq_ext', 'e_sq_int', 'e0_sq',
                 'e_total', 'e_total_ext', 'e_total_int']:
        if hasattr(field, attr):
            val = np.array(getattr(field, attr))
            if val.size > 0:
                result[attr] = val

    wl = result.get('wavelength', '?')
    wl_idx = result.get('wavelength_idx', '?')
    pol_idx = result.get('polarization_idx', '?')
    print(f"Loaded entry [{entry_idx}]: wavelength = {wl} nm, wl_idx = {wl_idx}, pol_idx = {pol_idx}")

    # Report available fields
    field_keys = [k for k in result if k not in
                  ['wavelength', 'wavelength_idx', 'polarization_idx', 'polarization',
                   'x_grid', 'y_grid', 'z_grid']]
    print(f"Available field data: {field_keys}")

    return result


# ============================================================================
# Plane Detection (from visualizer.py logic)
# ============================================================================

def determine_plane(x_grid, y_grid, z_grid):
    """
    Determine which 2D plane the field data lies on.

    Returns
    -------
    plane_type : str
        'xy', 'xz', 'yz', or '3d'
    extent : list
        [left, right, bottom, top] for imshow
    x_label, y_label : str
    """
    x_grid = np.atleast_2d(np.asarray(x_grid, dtype=float))
    y_grid = np.atleast_2d(np.asarray(y_grid, dtype=float))
    z_grid = np.atleast_2d(np.asarray(z_grid, dtype=float))

    x_constant = len(np.unique(x_grid)) == 1
    y_constant = len(np.unique(y_grid)) == 1
    z_constant = len(np.unique(z_grid)) == 1

    def safe_extent(a_min, a_max, b_min, b_max):
        if a_min == a_max:
            a_min -= 0.5; a_max += 0.5
        if b_min == b_max:
            b_min -= 0.5; b_max += 0.5
        return [a_min, a_max, b_min, b_max]

    if y_constant:
        extent = safe_extent(x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max())
        return 'xz', extent, 'x (nm)', 'z (nm)'
    elif z_constant:
        extent = safe_extent(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
        return 'xy', extent, 'x (nm)', 'y (nm)'
    elif x_constant:
        extent = safe_extent(y_grid.min(), y_grid.max(), z_grid.min(), z_grid.max())
        return 'yz', extent, 'y (nm)', 'z (nm)'
    else:
        extent = safe_extent(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
        return '3d', extent, 'x (nm)', 'y (nm)'


# ============================================================================
# Geometry Cross-Section (embedded from geometry_cross_section.py)
# ============================================================================

def load_geometry_config(config_path):
    """Load config_structure.py by exec() and return the args dict."""
    namespace = {'os': os, 'Path': __import__('pathlib').Path}
    with open(config_path, 'r') as f:
        exec(f.read(), namespace)
    if 'args' not in namespace:
        raise KeyError(f"config_structure.py does not define 'args' dict: {config_path}")
    return namespace['args']


def get_cross_section(config, z_plane, plane_type='xy'):
    """
    Calculate geometry cross-section at given z-plane.
    Simplified embedded version of GeometryCrossSection.
    """
    structure = config.get('structure', '')
    sections = []

    if structure in ('sphere',):
        sections = _cs_sphere(config, z_plane)
    elif structure in ('core_shell', 'core_shell_sphere'):
        sections = _cs_core_shell(config, z_plane)
    elif structure in ('cube', 'core_shell_cube', 'advanced_monomer_cube'):
        sections = _cs_cube(config, z_plane)
    elif structure in ('dimer', 'dimer_sphere'):
        sections = _cs_dimer_sphere(config, z_plane)
    elif structure in ('dimer_cube', 'dimer_core_shell_cube', 'advanced_dimer_cube', 'connected_dimer_cube'):
        sections = _cs_dimer_cube(config, z_plane)
    elif structure in ('rod', 'core_shell_rod'):
        sections = _cs_rod(config, z_plane)
    elif structure in ('sphere_cluster', 'sphere_cluster_aggregate'):
        sections = _cs_sphere_cluster(config, z_plane)
    else:
        print(f"  Warning: Unknown structure '{structure}', skipping boundary")

    return sections


def _cs_sphere(config, z_plane):
    radius = config.get('diameter', config.get('radius', 10.0) * 2) / 2
    center = config.get('center', [0, 0, 0])
    z_diff = z_plane - center[2]
    if abs(z_diff) > radius:
        return []
    r_cross = np.sqrt(radius**2 - z_diff**2)
    return [{'type': 'circle', 'center': [center[0], center[1]], 'radius': r_cross}]


def _cs_core_shell(config, z_plane):
    radii = config.get('radii', [])
    if not radii:
        core_d = config.get('core_diameter', 40.0)
        shell_t = config.get('shell_thickness', 10.0)
        radii = [core_d / 2, core_d / 2 + shell_t]
    center = config.get('center', [0, 0, 0])
    z_diff = z_plane - center[2]
    sections = []
    for r in sorted(radii):
        if abs(z_diff) <= r:
            r_cross = np.sqrt(r**2 - z_diff**2)
            sections.append({'type': 'circle', 'center': [center[0], center[1]], 'radius': r_cross})
    return sections


def _cs_cube(config, z_plane):
    # Handle various config naming conventions
    side = config.get('side_length', config.get('size', config.get('core_size', 20.0)))
    shells = config.get('shell_layers', [])
    total_side = side + 2 * sum(shells)
    center = config.get('center', [0, 0, 0])
    half = total_side / 2
    if z_plane < center[2] - half or z_plane > center[2] + half:
        return []
    sections = []
    # Outermost first (shell)
    sections.append({
        'type': 'rectangle',
        'bounds': [center[0] - half, center[0] + half, center[1] - half, center[1] + half]
    })
    # Core if there are shells
    if shells:
        half_core = side / 2
        sections.append({
            'type': 'rectangle',
            'bounds': [center[0] - half_core, center[0] + half_core,
                       center[1] - half_core, center[1] + half_core]
        })
    return sections


def _cs_dimer_sphere(config, z_plane):
    radius = config.get('diameter', config.get('radius', 10.0) * 2) / 2
    gap = config.get('gap', 2.0)
    center = config.get('center', [0, 0, 0])
    axis = config.get('dimer_axis', 'x')
    offset = radius + gap / 2
    if axis == 'x':
        pos1 = [center[0] - offset, center[1], center[2]]
        pos2 = [center[0] + offset, center[1], center[2]]
    elif axis == 'y':
        pos1 = [center[0], center[1] - offset, center[2]]
        pos2 = [center[0], center[1] + offset, center[2]]
    else:
        pos1 = [center[0], center[1], center[2] - offset]
        pos2 = [center[0], center[1], center[2] + offset]
    sections = []
    for pos in [pos1, pos2]:
        z_diff = z_plane - pos[2]
        if abs(z_diff) <= radius:
            r_cross = np.sqrt(radius**2 - z_diff**2)
            sections.append({'type': 'circle', 'center': [pos[0], pos[1]], 'radius': r_cross})
    return sections


def _cs_dimer_cube(config, z_plane):
    core_size = config.get('side_length', config.get('size', config.get('core_size', 20.0)))
    shells = config.get('shell_layers', [])
    total_side = core_size + 2 * sum(shells)
    gap = config.get('gap', 2.0)
    center = config.get('center', [0, 0, 0])
    axis = config.get('dimer_axis', 'x')
    offset = total_side / 2 + gap / 2
    half = total_side / 2
    if axis == 'x':
        positions = [[center[0] - offset, center[1], center[2]],
                     [center[0] + offset, center[1], center[2]]]
    elif axis == 'y':
        positions = [[center[0], center[1] - offset, center[2]],
                     [center[0], center[1] + offset, center[2]]]
    else:
        positions = [[center[0], center[1], center[2] - offset],
                     [center[0], center[1], center[2] + offset]]
    sections = []
    for pos in positions:
        z_min = pos[2] - half
        z_max = pos[2] + half
        if z_min <= z_plane <= z_max:
            sections.append({
                'type': 'rectangle',
                'bounds': [pos[0] - half, pos[0] + half, pos[1] - half, pos[1] + half]
            })
            # Core boundary if shells exist
            if shells:
                half_core = core_size / 2
                sections.append({
                    'type': 'rectangle',
                    'bounds': [pos[0] - half_core, pos[0] + half_core,
                               pos[1] - half_core, pos[1] + half_core]
                })
    return sections


def _cs_rod(config, z_plane):
    radius = config.get('diameter', config.get('radius', 5.0) * 2) / 2
    length = config.get('length', config.get('height', 40.0))
    center = config.get('center', [0, 0, 0])
    axis = config.get('axis', 'z')
    if axis == 'z':
        z_min = center[2] - length / 2
        z_max = center[2] + length / 2
        if z_plane < z_min or z_plane > z_max:
            return []
        return [{'type': 'circle', 'center': [center[0], center[1]], 'radius': radius}]
    elif axis == 'x':
        z_diff = z_plane - center[2]
        if abs(z_diff) > radius:
            return []
        y_half = np.sqrt(radius**2 - z_diff**2)
        return [{'type': 'rectangle',
                 'bounds': [center[0] - length/2, center[0] + length/2,
                            center[1] - y_half, center[1] + y_half]}]
    return []


def _cs_sphere_cluster(config, z_plane):
    num = config.get('n_spheres', 1)
    diameter = config.get('diameter', 50.0)
    gap = config.get('gap', -0.1)
    radius = diameter / 2
    spacing = diameter + gap
    positions = _cluster_positions(num, spacing)
    sections = []
    for pos in positions:
        z_diff = z_plane - pos[2]
        if abs(z_diff) <= radius:
            r_cross = np.sqrt(radius**2 - z_diff**2)
            sections.append({'type': 'circle', 'center': [pos[0], pos[1]], 'radius': r_cross})
    return sections


def _cluster_positions(n, spacing):
    dy60 = spacing * 0.866025404
    hex_pos = []
    for i in range(6):
        angle = i * 60 * np.pi / 180
        hex_pos.append((spacing * np.cos(angle), spacing * np.sin(angle), 0))
    layouts = {
        1: [(0, 0, 0)],
        2: [(-spacing/2, 0, 0), (spacing/2, 0, 0)],
        3: [(-spacing/2, 0, 0), (spacing/2, 0, 0), (0, dy60, 0)],
        4: [(0, 0, 0)] + hex_pos[:3],
        5: [(0, 0, 0)] + hex_pos[:4],
        6: [(0, 0, 0)] + hex_pos[:5],
        7: [(0, 0, 0)] + hex_pos[:6],
    }
    return layouts.get(n, [(0, 0, 0)])


# ============================================================================
# Drawing Helpers
# ============================================================================

def draw_material_boundary(ax, sections):
    """Draw material boundary patches on axes."""
    for section in sections:
        if section['type'] == 'circle':
            c = Circle(section['center'], section['radius'],
                       fill=False, edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.add_patch(c)
        elif section['type'] == 'rectangle':
            b = section['bounds']
            r = Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                           fill=False, edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.add_patch(r)


# ============================================================================
# Field Map Plotting
# ============================================================================

def plot_single_fieldmap(data_2d, extent, x_label, y_label, title, cbar_label,
                         cmap, out_path, sections=None, dpi=300):
    """
    Plot a single field map with log scale.

    Parameters
    ----------
    data_2d : 2D array
    extent : [left, right, bottom, top]
    title : str
    cbar_label : str
    cmap : str
    out_path : str
    sections : list of geometry cross-sections (optional)
    """
    # Handle complex
    if np.iscomplexobj(data_2d):
        data_2d = np.abs(data_2d)

    # Mask NaN / inf
    data_masked = np.ma.masked_invalid(data_2d)
    valid = data_masked.compressed()

    if len(valid) == 0:
        print(f"  SKIP {out_path}: no valid data")
        return

    positive = valid[valid > 0]
    if len(positive) == 0:
        print(f"  SKIP {out_path}: no positive data for log scale")
        return

    vmin = float(positive.min())
    vmax = float(positive.max())

    if vmin >= vmax:
        vmin = vmax / 10

    print(f"  {os.path.basename(out_path)}: range [{vmin:.4e}, {vmax:.4e}]")

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(data_masked, extent=extent, origin='lower',
                   cmap=cmap, aspect='auto',
                   norm=LogNorm(vmin=vmin, vmax=vmax))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)

    if sections:
        draw_material_boundary(ax, sections)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

# Define the 9 plot configurations
PLOT_CONFIGS = [
    # (data_key,           quantity_label,               cbar_label,       cmap,      filename_tag)
    ('enhancement_ext',    'Enhancement |E/E0| (Ext)',   '|E/E\u2080|',   'hot',     'enhancement_ext'),
    ('enhancement_int',    'Enhancement |E/E0| (Int)',   '|E/E\u2080|',   'hot',     'enhancement_int'),
    ('enhancement',        'Enhancement |E/E0| (Merged)','|E/E\u2080|',   'hot',     'enhancement_merged'),
    ('intensity_ext',      'Intensity |E/E0|\u00b2 (Ext)',  '|E/E\u2080|\u00b2', 'inferno', 'intensity_ext'),
    ('intensity_int',      'Intensity |E/E0|\u00b2 (Int)',  '|E/E\u2080|\u00b2', 'inferno', 'intensity_int'),
    ('intensity',          'Intensity |E/E0|\u00b2 (Merged)','|E/E\u2080|\u00b2','inferno', 'intensity_merged'),
    ('e_sq_ext',           'Raw |E|\u00b2 (Ext)',        '|E|\u00b2 (a.u.)', 'magma', 'e_sq_ext'),
    ('e_sq_int',           'Raw |E|\u00b2 (Int)',        '|E|\u00b2 (a.u.)', 'magma', 'e_sq_int'),
    ('e_sq',               'Raw |E|\u00b2 (Merged)',     '|E|\u00b2 (a.u.)', 'magma', 'e_sq_merged'),
]


def main():
    parser = argparse.ArgumentParser(
        description='Plot field maps from MNPBEM simulation results (log scale, full range)')
    parser.add_argument('mat_file', help='Path to simulation_results.mat')
    parser.add_argument('--field_entry', type=int, required=True,
                        help='Field entry index (0-based)')
    parser.add_argument('--config_structure', type=str, default=None,
                        help='Path to config_structure.py for material boundary overlay')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory (default: current dir)')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'], help='Output format (default: png)')
    args = parser.parse_args()

    # Load field entry
    field = load_field_entry(args.mat_file, args.field_entry)

    # Grid and plane
    x_grid = field['x_grid']
    y_grid = field['y_grid']
    z_grid = field['z_grid']
    plane_type, extent, x_label, y_label = determine_plane(x_grid, y_grid, z_grid)

    wl = field.get('wavelength', 0)
    wl_idx = field.get('wavelength_idx', args.field_entry)
    pol_idx = field.get('polarization_idx', 0)
    print(f"\nPlane: {plane_type}, wavelength = {wl:.1f} nm")

    # Transpose check
    n_unique_x = len(np.unique(x_grid))
    n_unique_y = len(np.unique(y_grid))

    # Load geometry (optional)
    sections = None
    if args.config_structure:
        try:
            geo_config = load_geometry_config(args.config_structure)
            # Determine z_plane for cross-section
            if plane_type == 'xy':
                z_plane = float(np.unique(z_grid).mean())
            elif plane_type == 'xz':
                z_plane = 0.0  # y-constant plane, use z=0 for cross-section
            else:
                z_plane = 0.0
            sections = get_cross_section(geo_config, z_plane, plane_type)
            if sections:
                print(f"Geometry boundary: {len(sections)} section(s) from {os.path.basename(args.config_structure)}")
            else:
                print(f"Geometry: no cross-section at z = {z_plane:.1f} nm")
        except Exception as e:
            print(f"Warning: Could not load geometry config: {e}")
            sections = None

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Generate 9 plots
    print(f"\nGenerating field maps (log scale, full range)...")
    n_plotted = 0
    n_skipped = 0

    for data_key, qty_label, cbar_label, cmap, fname_tag in PLOT_CONFIGS:
        if data_key not in field:
            print(f"  SKIP {fname_tag}: '{data_key}' not in field data")
            n_skipped += 1
            continue

        data_2d = np.array(field[data_key])

        # Handle scalar / 1D
        if data_2d.ndim == 0:
            data_2d = np.array([[data_2d.item()]])
        elif data_2d.ndim == 1:
            data_2d = data_2d.reshape(1, -1)

        # Transpose if needed
        if data_2d.shape == (n_unique_x, n_unique_y) and n_unique_x != n_unique_y:
            data_2d = data_2d.T

        title = f'{qty_label}\n\u03bb = {wl:.1f} nm (entry {args.field_entry}), pol {pol_idx}'
        out_path = os.path.join(args.outdir, f'fieldmap_{fname_tag}_wl{wl_idx}.{args.format}')

        plot_single_fieldmap(data_2d, extent, x_label, y_label, title, cbar_label,
                             cmap, out_path, sections=sections, dpi=args.dpi)
        n_plotted += 1

    print(f"\nDone: {n_plotted} plots saved, {n_skipped} skipped")


if __name__ == '__main__':
    main()
