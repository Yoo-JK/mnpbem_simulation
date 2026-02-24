#!/usr/bin/env python3
"""
Standalone Field Map Visualization for MNPBEM Simulation Results.

Generates 9 individual field map plots (log scale, hybrid edge artifact filter):
  - Enhancement |E/E0|:  External, Internal, Merged
  - Intensity |E/E0|^2:  External, Internal, Merged
  - Raw |E|^2:           External, Internal, Merged

Internal field data is filtered using hybrid edge + spatial isolation filter
to remove BEM boundary artifacts while preserving physical features (gap hotspots).

Usage:
    python plot_field_maps.py --input field_data.mat --field_entry 15 --config_structure /path/to/config_structure.py
    python plot_field_maps.py --input field_data.mat --field_entry 15 --config_structure /path/to/config_structure.py --vmax_percentile 99
    python plot_field_maps.py --input field_data.mat --field_entry 15  # no filtering without config
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

# Import shared edge filter module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from post_utils.edge_filter import get_sphere_boundaries_from_config, find_edge_artifacts
    HAS_EDGE_FILTER = True
except ImportError:
    print("Warning: edge_filter module not found. Edge artifact filtering disabled.")
    print("  Expected location: post_utils/edge_filter.py (relative to this script)")
    HAS_EDGE_FILTER = False


# ============================================================================
# Data Loading
# ============================================================================

_FIELD_ATTRS = ['x_grid', 'y_grid', 'z_grid', 'polarization',
                'enhancement', 'enhancement_ext', 'enhancement_int',
                'intensity', 'intensity_ext', 'intensity_int',
                'e_sq', 'e_sq_ext', 'e_sq_int', 'e0_sq',
                'e_total', 'e_total_ext', 'e_total_int']


def load_field_entry(mat_path, entry_idx):
    """
    Load a specific field entry from the .mat file.
    Supports both MATLAB v5/v7 (scipy) and v7.3 HDF5 (h5py) formats.
    """
    try:
        mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        print("Detected MATLAB v7.3 (HDF5) format, using h5py...")
        return _load_field_entry_h5(mat_path, entry_idx)

    # --- scipy path (MATLAB v5/v7) ---
    if 'results' in mat_data:
        results = mat_data['results']
        fields_struct = results.fields
    elif 'field_data' in mat_data:
        fields_struct = mat_data['field_data']
    else:
        raise KeyError("MAT file has neither 'results.fields' nor 'field_data'")

    if not isinstance(fields_struct, np.ndarray):
        fields_struct = np.array([fields_struct])

    n_entries = len(fields_struct)
    print(f"Total field entries in file: {n_entries}")

    if entry_idx < 0 or entry_idx >= n_entries:
        print("\nAvailable field entries:")
        for i, f in enumerate(fields_struct):
            wl = float(f.wavelength) if hasattr(f, 'wavelength') else '?'
            wl_idx = int(f.wavelength_idx) if hasattr(f, 'wavelength_idx') else '?'
            pol_idx = int(f.polarization_idx) if hasattr(f, 'polarization_idx') else '?'
            print(f"  [{i}] wavelength = {wl} nm, wl_idx = {wl_idx}, pol_idx = {pol_idx}")
        raise IndexError(f"field_entry {entry_idx} out of range [0, {n_entries-1}]")

    field = fields_struct[entry_idx]

    result = {}
    for attr in ['wavelength', 'wavelength_idx', 'polarization_idx']:
        if hasattr(field, attr):
            result[attr] = float(getattr(field, attr)) if attr == 'wavelength' else int(getattr(field, attr))

    for attr in _FIELD_ATTRS:
        if hasattr(field, attr):
            val = np.array(getattr(field, attr))
            if val.size > 0:
                result[attr] = val

    return _report_loaded(result, entry_idx)


def _load_field_entry_h5(mat_path, entry_idx):
    """Load field entry from MATLAB v7.3 (HDF5) file using h5py."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "This .mat file is MATLAB v7.3 (HDF5) format.\n"
            "Install h5py to read it:  pip install h5py"
        )

    with h5py.File(mat_path, 'r') as f:
        if 'results' in f and 'fields' in f['results']:
            fields_grp = f['results']['fields']
        elif 'field_data' in f:
            fields_grp = f['field_data']
        else:
            raise KeyError("MAT file has neither 'results/fields' nor 'field_data'")

        field_names = list(fields_grp.keys())
        if not field_names:
            raise ValueError("No fields found in struct")

        first_ds = fields_grp[field_names[0]]
        is_ref = (first_ds.dtype == h5py.ref_dtype or first_ds.dtype == object)

        if is_ref:
            n_entries = first_ds.shape[0]
        else:
            n_entries = 1

        print(f"Total field entries in file: {n_entries}")

        if entry_idx < 0 or entry_idx >= n_entries:
            if is_ref and 'wavelength' in fields_grp:
                print("\nAvailable field entries:")
                wl_ds = fields_grp['wavelength']
                for i in range(n_entries):
                    try:
                        wl = float(np.array(f[wl_ds[i, 0]]).squeeze())
                    except Exception:
                        wl = '?'
                    print(f"  [{i}] wavelength = {wl} nm")
            raise IndexError(f"field_entry {entry_idx} out of range [0, {n_entries-1}]")

        result = {}
        all_attrs = ['wavelength', 'wavelength_idx', 'polarization_idx'] + _FIELD_ATTRS

        for attr in all_attrs:
            if attr not in fields_grp:
                continue
            ds = fields_grp[attr]
            try:
                if is_ref:
                    ref = ds[entry_idx, 0]
                    val = np.array(f[ref]).squeeze()
                else:
                    val = np.array(ds).squeeze()
            except Exception:
                continue

            if isinstance(val, np.ndarray) and val.ndim == 2:
                val = val.T

            if attr == 'wavelength':
                result[attr] = float(val)
            elif attr in ('wavelength_idx', 'polarization_idx'):
                result[attr] = int(val)
            else:
                val = np.array(val)
                if val.size > 0:
                    result[attr] = val

    return _report_loaded(result, entry_idx)


def _report_loaded(result, entry_idx):
    """Print summary of loaded field data and return result."""
    wl = result.get('wavelength', '?')
    wl_idx = result.get('wavelength_idx', '?')
    pol_idx = result.get('polarization_idx', '?')
    print(f"Loaded entry [{entry_idx}]: wavelength = {wl} nm, wl_idx = {wl_idx}, pol_idx = {pol_idx}")

    field_keys = [k for k in result if k not in
                  ['wavelength', 'wavelength_idx', 'polarization_idx', 'polarization',
                   'x_grid', 'y_grid', 'z_grid']]
    print(f"Available field data: {field_keys}")
    return result


# ============================================================================
# Plane Detection
# ============================================================================

def determine_plane(x_grid, y_grid, z_grid):
    """Determine which 2D plane the field data lies on."""
    x_grid = np.atleast_2d(np.asarray(x_grid, dtype=float))
    y_grid = np.atleast_2d(np.asarray(y_grid, dtype=float))
    z_grid = np.atleast_2d(np.asarray(z_grid, dtype=float))

    y_constant = len(np.unique(y_grid)) == 1
    z_constant = len(np.unique(z_grid)) == 1
    x_constant = len(np.unique(x_grid)) == 1

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
# Geometry Cross-Section (embedded)
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
    """Calculate geometry cross-section at given z-plane."""
    structure = config.get('structure', '')
    sections = []

    if structure in ('sphere',):
        sections = _cs_sphere(config, z_plane)
    elif structure in ('core_shell', 'core_shell_sphere'):
        sections = _cs_core_shell(config, z_plane)
    elif structure in ('dimer', 'dimer_sphere'):
        sections = _cs_dimer_sphere(config, z_plane)
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
                         cmap, out_path, sections=None, dpi=300,
                         vmax_percentile=None):
    """
    Plot a single field map with log scale.

    Parameters
    ----------
    vmax_percentile : float, optional
        If set, use this percentile of positive data as vmax for colorbar.
        E.g., 99 means 99th percentile as vmax. Does NOT remove data.
    """
    if np.iscomplexobj(data_2d):
        data_2d = np.abs(data_2d)

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

    # Optional percentile-based vmax clamping (colorbar only, no data removal)
    if vmax_percentile is not None and 0 < vmax_percentile < 100:
        vmax = float(np.percentile(positive, vmax_percentile))
        if vmax <= vmin:
            vmax = float(positive.max())

    if vmin >= vmax:
        vmin = vmax / 10

    print(f"  {os.path.basename(out_path)}: range [{vmin:.4e}, {vmax:.4e}]")

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='black')

    im = ax.imshow(data_masked, extent=extent, origin='lower',
                   cmap=cmap_obj, aspect='auto',
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

# 9 plot configurations: (data_key, label, cbar_label, cmap, filename_tag, int_mask_key)
# int_mask_key: for _ext plots, mask out pixels where the corresponding _int has valid data
PLOT_CONFIGS = [
    ('enhancement_ext',    'Enhancement |E/E0| (Ext)',   '|E/E0|',   'hot',     'enhancement_ext',  'enhancement_int'),
    ('enhancement_int',    'Enhancement |E/E0| (Int)',   '|E/E0|',   'hot',     'enhancement_int',  None),
    ('enhancement',        'Enhancement |E/E0| (Merged)','|E/E0|',   'hot',     'enhancement_merged',None),
    ('intensity_ext',      'Intensity |E/E0|^2 (Ext)',  '|E/E0|^2', 'inferno', 'intensity_ext',    'intensity_int'),
    ('intensity_int',      'Intensity |E/E0|^2 (Int)',  '|E/E0|^2', 'inferno', 'intensity_int',    None),
    ('intensity',          'Intensity |E/E0|^2 (Merged)','|E/E0|^2','inferno', 'intensity_merged',  None),
    ('e_sq_ext',           'Raw |E|^2 (Ext)',        '|E|^2 (a.u.)', 'magma', 'e_sq_ext',         'e_sq_int'),
    ('e_sq_int',           'Raw |E|^2 (Int)',        '|E|^2 (a.u.)', 'magma', 'e_sq_int',         None),
    ('e_sq',               'Raw |E|^2 (Merged)',     '|E|^2 (a.u.)', 'magma', 'e_sq_merged',      None),
]

# Mapping: internal field key -> merged field key (artifact mask shared)
INT_TO_MERGED = {
    'enhancement_int': 'enhancement',
    'intensity_int': 'intensity',
    'e_sq_int': 'e_sq',
}


def main():
    parser = argparse.ArgumentParser(
        description='Plot field maps from MNPBEM simulation results (log scale, hybrid edge filter)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to field_data.mat or simulation_results.mat')
    parser.add_argument('--field_entry', type=int, required=True,
                        help='Field entry index (0-based)')
    parser.add_argument('--config_structure', type=str, default=None,
                        help='Path to config_structure.py (required for edge artifact filter + boundary overlay)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory (default: current dir)')
    parser.add_argument('--prefix', type=str, default='fieldmap',
                        help='Output filename prefix (default: fieldmap)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Output filename suffix before extension (default: none)')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'], help='Output format (default: png)')
    parser.add_argument('--vmax_percentile', type=float, default=None,
                        help='Percentile for colorbar vmax (e.g., 99 = 99th percentile). '
                             'Does NOT remove pixels, only adjusts color range. (default: full range)')
    parser.add_argument('--edge_threshold', type=float, default=1.0,
                        help='Edge zone threshold in nm for artifact filter (default: 1.0)')
    parser.add_argument('--isolation_ratio', type=float, default=1.3,
                        help='Isolation ratio threshold for artifact filter (default: 1.3)')
    args = parser.parse_args()

    # Load field entry
    field = load_field_entry(args.input, args.field_entry)

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

    # Load geometry config (for boundary overlay + edge filter)
    geo_config = None
    sections = None
    spheres = None

    if args.config_structure:
        try:
            geo_config = load_geometry_config(args.config_structure)

            # Cross-section for boundary overlay
            if plane_type == 'xy':
                z_plane = float(np.unique(z_grid).mean())
            elif plane_type == 'xz':
                z_plane = 0.0
            else:
                z_plane = 0.0

            sections = get_cross_section(geo_config, z_plane, plane_type)
            if sections:
                print(f"Geometry boundary: {len(sections)} section(s)")

            # Sphere boundaries for edge filter
            if HAS_EDGE_FILTER:
                spheres = get_sphere_boundaries_from_config(geo_config)
                if spheres:
                    print(f"Edge filter: {len(spheres)} sphere(s) for artifact detection")
                else:
                    print("Edge filter: structure not supported, skipping artifact filter")

        except Exception as e:
            print(f"Warning: Could not load geometry config: {e}")

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Compute artifact masks for internal fields ----
    # Use enhancement_int to detect artifacts, apply same mask to all int + merged fields
    artifact_masks = {}  # artifact_masks[data_key] = boolean mask (True = artifact pixel)

    if spheres is not None and HAS_EDGE_FILTER:
        # Prepare 2D grids for edge filter
        x_2d = np.atleast_2d(np.asarray(x_grid, dtype=float))
        y_2d = np.atleast_2d(np.asarray(y_grid, dtype=float))
        z_2d = np.atleast_2d(np.asarray(z_grid, dtype=float))

        # Detect artifacts from enhancement_int (primary), fall back to others
        detection_key = None
        for candidate in ['enhancement_int', 'intensity_int', 'e_sq_int']:
            if candidate in field:
                detection_key = candidate
                break

        if detection_key is not None:
            det_data = np.array(field[detection_key], dtype=float)
            if det_data.ndim == 1:
                det_data = det_data.reshape(1, -1)
            if det_data.shape == (n_unique_x, n_unique_y) and n_unique_x != n_unique_y:
                det_data = det_data.T

            # Grid shape must match data shape
            if x_2d.shape != det_data.shape:
                x_2d = x_2d.reshape(det_data.shape) if x_2d.size == det_data.size else x_2d
                y_2d = y_2d.reshape(det_data.shape) if y_2d.size == det_data.size else y_2d
                z_2d = z_2d.reshape(det_data.shape) if z_2d.size == det_data.size else z_2d

            int_mask = ~np.isnan(det_data) & np.isfinite(det_data)

            artifact_mask, n_artifacts = find_edge_artifacts(
                det_data, x_2d, y_2d, z_2d, spheres,
                mask=int_mask,
                edge_threshold=args.edge_threshold,
                isolation_ratio=args.isolation_ratio,
                verbose=True
            )

            if n_artifacts > 0:
                # Apply same artifact mask to all internal and merged fields
                for int_key, merged_key in INT_TO_MERGED.items():
                    artifact_masks[int_key] = artifact_mask
                    artifact_masks[merged_key] = artifact_mask

                print(f"  Artifact filter: {n_artifacts} pixels will be removed from "
                      f"internal + merged plots (detected from {detection_key})")
            else:
                print(f"  Artifact filter: no artifacts detected")

    # ---- Generate 9 plots ----
    filter_msg = f"hybrid edge filter (edge<={args.edge_threshold}nm, ratio>{args.isolation_ratio}x)" \
                 if spheres and HAS_EDGE_FILTER else "no filter"
    vmax_msg = f", vmax={args.vmax_percentile}th pct" if args.vmax_percentile else ""
    print(f"\nGenerating field maps (log scale, {filter_msg}{vmax_msg})...")

    n_plotted = 0
    n_skipped = 0

    for data_key, qty_label, cbar_label, cmap, fname_tag, int_mask_key in PLOT_CONFIGS:
        if data_key not in field:
            print(f"  SKIP {fname_tag}: '{data_key}' not in field data")
            n_skipped += 1
            continue

        data_2d = np.array(field[data_key])

        if data_2d.ndim == 0:
            data_2d = np.array([[data_2d.item()]])
        elif data_2d.ndim == 1:
            data_2d = data_2d.reshape(1, -1)

        if data_2d.shape == (n_unique_x, n_unique_y) and n_unique_x != n_unique_y:
            data_2d = data_2d.T

        data_2d = data_2d.astype(float, copy=True)

        # For _ext plots: mask out internal region pixels
        if int_mask_key is not None and int_mask_key in field:
            int_data = np.array(field[int_mask_key])
            if int_data.ndim == 1:
                int_data = int_data.reshape(1, -1)
            if int_data.shape == (n_unique_x, n_unique_y) and n_unique_x != n_unique_y:
                int_data = int_data.T
            int_valid_mask = ~np.isnan(int_data) if not np.iscomplexobj(int_data) else ~np.isnan(np.abs(int_data))
            n_masked = int_valid_mask.sum()
            if n_masked > 0:
                data_2d[int_valid_mask] = np.nan
                print(f"  {fname_tag}: masked {n_masked} internal pixels in ext data")

        # For _int and merged plots: apply artifact mask
        if data_key in artifact_masks:
            mask = artifact_masks[data_key]
            if mask.shape == data_2d.shape:
                n_clipped = mask.sum()
                data_2d[mask] = np.nan
                print(f"  {fname_tag}: removed {n_clipped} edge artifact pixels")

        title = f'{qty_label}\n\u03bb = {wl:.1f} nm (entry {args.field_entry}), pol {pol_idx}'
        suffix_part = f'_{args.suffix}' if args.suffix else ''
        out_path = os.path.join(args.outdir, f'{args.prefix}_{fname_tag}_wl{wl_idx}{suffix_part}.{args.format}')

        plot_single_fieldmap(data_2d, extent, x_label, y_label, title, cbar_label,
                             cmap, out_path, sections=sections, dpi=args.dpi,
                             vmax_percentile=args.vmax_percentile)
        n_plotted += 1

    print(f"\nDone: {n_plotted} plots saved, {n_skipped} skipped")


if __name__ == '__main__':
    main()
