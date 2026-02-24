"""
Field Analysis Utilities

Analyzes electromagnetic field distributions.
"""

import numpy as np
from scipy.ndimage import maximum_filter


class FieldAnalyzer:
    """Analyzes electromagnetic field data."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.near_field_distances = [2.0, 15.0]
    
    def analyze_field(self, field_data):
        """
        Comprehensive field analysis.
        
        Parameters
        ----------
        field_data : dict or list of dict
            Single field data dict or list of field data for multiple polarizations
        
        Returns
        -------
        dict or list of dict
            Analysis results (single dict if input is dict, list if input is list)
        """
        # FIX: Handle both single and multiple polarizations properly
        if isinstance(field_data, list):
            # Multiple polarizations: analyze each separately
            if self.verbose:
                print(f"  Analyzing {len(field_data)} polarizations...")
            
            return [self._analyze_single_field(pol_data) for pol_data in field_data]
        else:
            # Single polarization
            return self._analyze_single_field(field_data)
    
    def _analyze_single_field(self, pol_data):
        """
        Analyze a single polarization field data.
        
        Parameters
        ----------
        pol_data : dict
            Field data for a single polarization
        
        Returns
        -------
        dict
            Analysis results
        """
        # Extract data
        enhancement = pol_data.get('enhancement')
        intensity = pol_data.get('intensity')
        x_grid = pol_data.get('x_grid')
        y_grid = pol_data.get('y_grid')
        z_grid = pol_data.get('z_grid')
        
        if enhancement is None:
            return None

        # Handle complex data
        if np.iscomplexobj(enhancement):
            if self.verbose:
                print("  Converting complex enhancement to magnitude...")
            enhancement = np.abs(enhancement)

        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        # Extract polarization once to avoid multiple .get() calls
        polarization = pol_data.get('polarization')
        if hasattr(polarization, 'tolist'):
            polarization = polarization.tolist()

        analysis = {
            'wavelength': pol_data.get('wavelength'),
            'wavelength_idx': pol_data.get('wavelength_idx'),
            'polarization': polarization,
            'polarization_idx': pol_data.get('polarization_idx'),
        }

        # Enhancement statistics
        analysis['enhancement_stats'] = self._calculate_statistics(enhancement)

        # Intensity statistics (only if intensity data exists)
        if intensity is not None:
            analysis['intensity_stats'] = self._calculate_statistics(intensity)
        else:
            analysis['intensity_stats'] = None
        
        # Find hotspots
        hotspots = self._find_hotspots(enhancement, x_grid, y_grid, z_grid)
        analysis['hotspots'] = hotspots
        
        # Volume above threshold
        analysis['high_field_regions'] = self._analyze_high_field_regions(
            enhancement, x_grid, y_grid, z_grid
        )
        
        if self.verbose:
            self._print_analysis(analysis)
        
        return analysis
    
    def _calculate_statistics(self, data):
        """Calculate statistical measures of field data."""
        if not isinstance(data, np.ndarray):
            data = np.array([data])

        if data.ndim == 0:
            data = np.array([data.item()])
        
        data_flat = data.flatten()
        data_flat = data_flat[np.isfinite(data_flat)]  # Remove inf/nan
        
        if len(data_flat) == 0:
            return {
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'percentile_90': 0.0,
                'percentile_95': 0.0,
                'percentile_99': 0.0
            }
        
        stats = {
            'max': float(np.max(data_flat)),
            'min': float(np.min(data_flat)),
            'mean': float(np.mean(data_flat)),
            'median': float(np.median(data_flat)),
            'std': float(np.std(data_flat)),
            'percentile_90': float(np.percentile(data_flat, 90)),
            'percentile_95': float(np.percentile(data_flat, 95)),
            'percentile_99': float(np.percentile(data_flat, 99))
        }
        
        return stats
    
    def _find_hotspots(self, enhancement, x_grid, y_grid, z_grid, 
                       num_hotspots=10, min_distance=3):
        """
        Find local maximum enhancement hotspots.
        """
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([enhancement])
        
        if enhancement.ndim == 0:
            enhancement = np.array([enhancement.item()])
        
        if enhancement.size == 1:
            hotspot = {
                'rank': 1,
                'position': [float(x_grid), float(y_grid), float(z_grid)],
                'enhancement': float(enhancement.item()),
                'intensity_enhancement': float(enhancement.item()**2)
            }
            return [hotspot]

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
            if self.verbose:
                print("  Converting complex enhancement to magnitude")

        neighborhood_size = min_distance * 2 + 1
        local_max = maximum_filter(enhancement, size=neighborhood_size)
        
        # Points that are equal to local max are local maxima
        is_local_max = (enhancement == local_max)
        
        # Also require enhancement > 1 (at least some enhancement)
        is_local_max = is_local_max & (enhancement > 1.0)
        
        # Get indices of local maxima
        max_indices = np.where(is_local_max)
        max_values = enhancement[max_indices]
        
        # Sort by enhancement value (descending)
        sorted_idx = np.argsort(max_values)[::-1]
        
        # Take top N hotspots
        hotspots = []
        for i in range(min(num_hotspots, len(sorted_idx))):
            idx = sorted_idx[i]

            # Get position based on array dimensionality
            ndim = enhancement.ndim
            n_indices = len(max_indices)

            if ndim == 1 or n_indices == 1:
                # 1D case
                flat_idx = max_indices[0][idx]
                x_pos = float(x_grid.flat[flat_idx]) if hasattr(x_grid, 'flat') else float(x_grid)
                y_pos = float(y_grid.flat[flat_idx]) if hasattr(y_grid, 'flat') else float(y_grid)
                z_pos = float(z_grid.flat[flat_idx]) if hasattr(z_grid, 'flat') else float(z_grid)
            elif ndim == 2 or n_indices == 2:
                # 2D case
                idx_0, idx_1 = max_indices[0][idx], max_indices[1][idx]
                x_pos = float(x_grid[idx_0, idx_1]) if x_grid.ndim >= 2 else float(x_grid.flat[idx_0])
                y_pos = float(y_grid[idx_0, idx_1]) if y_grid.ndim >= 2 else float(y_grid.flat[idx_0])
                z_pos = float(z_grid[idx_0, idx_1]) if z_grid.ndim >= 2 else float(z_grid.flat[idx_0])
            else:
                # 3D case
                i_idx, j_idx, k_idx = max_indices[0][idx], max_indices[1][idx], max_indices[2][idx]
                x_pos = float(x_grid[i_idx, j_idx, k_idx])
                y_pos = float(y_grid[i_idx, j_idx, k_idx])
                z_pos = float(z_grid[i_idx, j_idx, k_idx])

            hotspot = {
                'rank': i + 1,
                'position': [x_pos, y_pos, z_pos],
                'enhancement': float(max_values[idx]),
                'intensity_enhancement': float(max_values[idx]**2)
            }
            hotspots.append(hotspot)

        return hotspots
    
    def _analyze_high_field_regions(self, enhancement, x_grid, y_grid, z_grid):
        """
        Analyze regions with high field enhancement.
        
        Returns volume/area above various enhancement thresholds.
        """
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([enhancement])
        
        if enhancement.ndim == 0:
            enhancement = np.array([enhancement.item()])
        
        # Ensure all grids are numpy arrays
        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([x_grid])
        if not isinstance(y_grid, np.ndarray):
            y_grid = np.array([y_grid])
        if not isinstance(z_grid, np.ndarray):
            z_grid = np.array([z_grid])

        # Handle 0D arrays
        if x_grid.ndim == 0:
            x_grid = np.array([x_grid.item()])
        if y_grid.ndim == 0:
            y_grid = np.array([y_grid.item()])
        if z_grid.ndim == 0:
            z_grid = np.array([z_grid.item()])

        if enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)
            x_grid = x_grid.reshape(1, -1) if x_grid.ndim == 1 else x_grid
            y_grid = y_grid.reshape(1, -1) if y_grid.ndim == 1 else y_grid
            z_grid = z_grid.reshape(1, -1) if z_grid.ndim == 1 else z_grid

        # Calculate grid spacing (assuming uniform)
        if enhancement.ndim == 2:
            dx = np.abs(x_grid[0, 1] - x_grid[0, 0]) if x_grid.shape[1] > 1 else 1.0
            dy = np.abs(y_grid[1, 0] - y_grid[0, 0]) if y_grid.shape[0] > 1 else 1.0
            dz = 0
            element_area = dx * dy if dx > 0 and dy > 0 else 1.0
            is_3d = False
        else:  # 3D
            dx = np.abs(x_grid[0, 0, 1] - x_grid[0, 0, 0]) if x_grid.shape[2] > 1 else 1.0
            dy = np.abs(y_grid[0, 1, 0] - y_grid[0, 0, 0]) if y_grid.shape[1] > 1 else 1.0
            dz = np.abs(z_grid[1, 0, 0] - z_grid[0, 0, 0]) if z_grid.shape[0] > 1 else 1.0
            element_volume = dx * dy * dz if dx > 0 and dy > 0 and dz > 0 else 1.0
            is_3d = True
        
        thresholds = [2, 5, 10, 20, 50, 100]
        regions = {}
        
        for threshold in thresholds:
            mask = enhancement > threshold
            count = np.sum(mask)
            
            if is_3d:
                volume = count * element_volume
                regions[f'enhancement_above_{threshold}'] = {
                    'num_points': int(count),
                    'volume_nm3': float(volume)
                }
            else:
                area = count * element_area
                regions[f'enhancement_above_{threshold}'] = {
                    'num_points': int(count),
                    'area_nm2': float(area)
                }
        
        return regions
    
    def _print_analysis(self, analysis):
        """Print field analysis summary."""
        pol_idx = analysis.get('polarization_idx', '?')
        wl_idx = analysis.get('wavelength_idx', '?')
        print(f"\n  Field Analysis (λ = {analysis['wavelength']:.1f} nm, wl_idx={wl_idx}, pol_idx={pol_idx}):")
        print("  " + "-"*50)
        
        # Enhancement statistics
        stats = analysis['enhancement_stats']
        print(f"  Enhancement Statistics:")
        print(f"    Max:       {stats['max']:.2f}")
        print(f"    Mean:      {stats['mean']:.2f}")
        print(f"    Median:    {stats['median']:.2f}")
        print(f"    95th %ile: {stats['percentile_95']:.2f}")
        
        # Top hotspots
        if analysis['hotspots']:
            print(f"\n  Top {len(analysis['hotspots'])} Hotspots:")
            for hotspot in analysis['hotspots'][:5]:  # Print top 5
                pos = hotspot['position']
                print(f"    #{hotspot['rank']}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm "
                      f"| E/E₀ = {hotspot['enhancement']:.2f}")

    def calculate_near_field_integration(self, field_data_list, config, geometry,
                                          center_only=False, top_percentile_filter=None):
        """
        Calculate near-field integration for all field data.

        Integrates field enhancement and intensity in region near particle surfaces
        (default: 5nm from surface, exterior only).

        Parameters
        ----------
        field_data_list : list of dict
            List of field data dictionaries
        config : dict
            Simulation configuration
        geometry : GeometryCrossSection
            Geometry calculator for particle boundaries
        center_only : bool
            If True, integrate over center sphere only (for cluster structures)
        top_percentile_filter : float, optional
            If set, remove the top N% of enhancement values after filtering.
            E.g., top_percentile_filter=1 removes the top 1%.

        Returns
        -------
        dict or None
            Integration results organized by wavelength and polarization
        """
        structure_type = config.get('structure', 'unknown')

        if not self._is_structure_supported_for_integration(structure_type):
            if self.verbose:
                print(f"  [!] Structure '{structure_type}' not supported for near-field integration")
            return None

        results = {}

        for field_data in field_data_list:
            wl = field_data['wavelength']
            pol_idx = field_data.get('polarization_idx')

            if self.verbose:
                pol_str = f"pol{pol_idx+1}" if pol_idx is not None else "unpolarized"
                mode_str = " (center only)" if center_only else ""
                print(f"\n  Processing λ={wl:.1f} nm, {pol_str}{mode_str}")

            # Calculate integration for this field
            integration_result = self._integrate_single_field(
                field_data, config, geometry, center_only=center_only,
                top_percentile_filter=top_percentile_filter
            )
            
            # Organize results
            if wl not in results:
                results[wl] = {}
            
            if pol_idx is not None:
                key = f'polarization_{pol_idx+1}'
            else:
                key = 'unpolarized'
            
            results[wl][key] = integration_result
        
        return results
    
    def _integrate_single_field(self, field_data, config, geometry, center_only=False,
                                top_percentile_filter=None):
        """
        Integrate field values for a single wavelength/polarization.

        Calculates integration at multiple depths (10nm, 15nm interior).

        Parameters
        ----------
        field_data : dict
            Field data dictionary
        config : dict
            Simulation configuration
        geometry : GeometryCrossSection
            Geometry calculator
        center_only : bool
            If True, integrate over center sphere only
        top_percentile_filter : float, optional
            If set, remove the top N% of enhancement values after filtering.

        Returns
        -------
        dict
            Integration results with strict and conservative filtering for each depth
        """
        enhancement = field_data['enhancement']
        intensity = field_data['intensity']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']

        # Get raw intensities for energy ratio calculation: sum(|E|²)/sum(|E0|²)
        e_sq = field_data.get('e_sq')        # |E|² (raw)
        e0_sq = field_data.get('e0_sq')      # |E0|² (reference)
        e_sq_int = field_data.get('e_sq_int')  # |E|² internal (for chunked version)

        # Handle complex data
        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)
        if e_sq is not None and np.iscomplexobj(e_sq):
            e_sq = np.abs(e_sq)
        if e0_sq is not None and np.iscomplexobj(e0_sq):
            e0_sq = np.abs(e0_sq)
        if e_sq_int is not None and np.iscomplexobj(e_sq_int):
            e_sq_int = np.abs(e_sq_int)

        if self.verbose:
            enh_finite = enhancement[np.isfinite(enhancement)]
            print(f"    [DEBUG] Enhancement array:")
            print(f"      Shape: {enhancement.shape}")
            print(f"      Total points: {enhancement.size}")
            print(f"      Finite points: {len(enh_finite)}")
            print(f"      NaN points: {np.sum(np.isnan(enhancement))}")
            print(f"      Inf points: {np.sum(np.isinf(enhancement))}")
            if len(enh_finite) > 0:
                print(f"      Range: {np.min(enh_finite):.3f} ~ {np.max(enh_finite):.3f}")
                print(f"      Mean: {np.mean(enh_finite):.3f}")

        # Get particle boundaries
        spheres = self._get_sphere_boundaries(config, geometry, center_only=center_only)
        
        if spheres is None or len(spheres) == 0:
            if self.verbose:
                print("    [!] Could not determine sphere boundaries")
            return self._empty_integration_result()
        
        n_spheres = len(spheres)

        if self.verbose:
            print(f"    [DEBUG] Spheres ({n_spheres} total):")
            for i, (cx, cy, cz, r) in enumerate(spheres):
                print(f"      Sphere {i+1}: center=({cx:.1f}, {cy:.1f}, {cz:.1f}), radius={r:.1f} nm")
        
        # Calculate for each depth
        results_by_depth = {}

        for depth in self.near_field_distances:

            if self.verbose:
                print(f"    [DEBUG] Processing depth = {depth:.1f} nm (interior)")
            # Create distance mask for this depth
            distance_mask = self._create_distance_mask(x_grid, y_grid, z_grid, spheres, depth)

            # Calculate with two filtering methods
            result_strict = self._calculate_with_filtering(
                enhancement, intensity, distance_mask, n_spheres,
                e_sq=e_sq, e0_sq=e0_sq, e_sq_int=e_sq_int, method='strict',
                top_percentile_filter=top_percentile_filter
            )

            result_conservative = self._calculate_with_filtering(
                enhancement, intensity, distance_mask, n_spheres,
                e_sq=e_sq, e0_sq=e0_sq, e_sq_int=e_sq_int, method='conservative',
                top_percentile_filter=top_percentile_filter
            )

            results_by_depth[depth] = {
                'strict': result_strict,
                'conservative': result_conservative,
                'n_spheres': n_spheres
            }
        
        # Add grid info (same for all depths)
        grid_info = {
            'total_points': int(np.prod(enhancement.shape)),
            'valid_points': int(np.sum(~np.isnan(enhancement)))
        }
        
        return {
            'depths': results_by_depth,
            'grid_info': grid_info
        }
    
    def _calculate_with_filtering(self, enhancement, intensity, distance_mask, n_spheres,
                                    e_sq=None, e0_sq=None, e_sq_int=None, method='strict',
                                    top_percentile_filter=None):
        """
        Calculate sums with specified filtering method.

        Parameters
        ----------
        enhancement : ndarray
            Field enhancement array
        intensity : ndarray
            Field intensity array
        distance_mask : ndarray (bool)
            Mask for integration region
        n_spheres : int
            Number of spheres
        e_sq : ndarray, optional
            |E|² raw intensity array for energy ratio calculation
        e0_sq : ndarray, optional
            |E0|² reference intensity array for energy ratio calculation
        e_sq_int : ndarray, optional
            |E|² internal intensity array (for chunked version)
        method : str
            'strict' or 'conservative'
        top_percentile_filter : float, optional
            If set, remove the top N% of enhancement values after filtering.
            E.g., top_percentile_filter=1 removes the top 1% (keeps below 99th percentile).

        Returns
        -------
        dict
            Calculated sums and statistics (including per-sphere values and energy_ratio)
        """
        # Start with distance mask
        final_mask = distance_mask.copy()

        if self.verbose:
            print(f"      [DEBUG] {method.upper()} filtering:")
            print(f"        Distance mask: {np.sum(distance_mask)} points")

        # Apply filtering
        if method == 'strict':
            # Remove only Inf and NaN
            valid_enh = np.isfinite(enhancement)

            if self.verbose:
                print(f"        Finite enhancement: {np.sum(valid_enh)} points")

            final_mask = final_mask & valid_enh

            excluded_outliers = 0

        elif method == 'conservative':
            # Remove Inf, NaN, and outliers
            valid_enh = np.isfinite(enhancement)

            if self.verbose:
                print(f"        Finite enhancement: {np.sum(valid_enh)} points")

            if np.sum(valid_enh) > 0:
                # Calculate threshold from valid data
                threshold = np.percentile(enhancement[valid_enh], 99.9)
                outlier_mask = enhancement <= threshold * 10

                if self.verbose:
                    print(f"        99.9%ile threshold: {threshold:.3f}")
                    print(f"        Outlier cutoff: {threshold * 10:.3f}")

                excluded_outliers = int(np.sum(distance_mask & valid_enh & ~outlier_mask))

                final_mask = final_mask & valid_enh & outlier_mask
            else:
                final_mask = final_mask & valid_enh
                excluded_outliers = 0

        # Apply top percentile filter: remove top N% of enhancement values
        excluded_top_percentile = 0
        if top_percentile_filter is not None and top_percentile_filter > 0:
            enh_in_mask = enhancement[final_mask]
            if len(enh_in_mask) > 0:
                keep_percentile = 100.0 - top_percentile_filter
                percentile_threshold = np.percentile(enh_in_mask, keep_percentile)
                top_pct_mask = enhancement <= percentile_threshold
                excluded_top_percentile = int(np.sum(final_mask & ~top_pct_mask))
                final_mask = final_mask & top_pct_mask

                if self.verbose:
                    print(f"        Top {top_percentile_filter}% filter: threshold={percentile_threshold:.3f}")
                    print(f"        Excluded top {top_percentile_filter}%: {excluded_top_percentile} points")

        if self.verbose:
            print(f"        Final mask: {np.sum(final_mask)} points")
        
        # Extract values in region
        enh_in_region = enhancement[final_mask]
        
        if len(enh_in_region) == 0:

            if self.verbose:
                print(f"        [!] No valid points found in region!")
            
            return self._empty_integration_result(method)
        
        # Calculate sums
        enh_sum = float(np.sum(enh_in_region))
        enh_mean = float(np.mean(enh_in_region))

        # Per-sphere statistics
        enh_per_sphere = enh_sum / n_spheres if n_spheres > 0 else 0.0

        if self.verbose:
            print(f"        Enhancement sum: {enh_sum:.3f}")
            print(f"        Enhancement mean: {enh_mean:.3f}")
            print(f"        Per-sphere: {enh_per_sphere:.3f}")

        result = {
            'enhancement_sum': enh_sum,
            'enhancement_mean': enh_mean,
            'enhancement_per_sphere': enh_per_sphere,
            'valid_points': int(np.sum(final_mask)),
        }

        # Add intensity if available
        if intensity is not None:
            int_in_region = intensity[final_mask]
            int_sum = float(np.sum(int_in_region))
            int_mean = float(np.mean(int_in_region))
            int_per_sphere = int_sum / n_spheres if n_spheres > 0 else 0.0

            result['intensity_sum'] = int_sum
            result['intensity_mean'] = int_mean
            result['intensity_per_sphere'] = int_per_sphere
        else:
            result['intensity_sum'] = None
            result['intensity_mean'] = None
            result['intensity_per_sphere'] = None

        # Calculate energy ratio: sum(|E|²) / sum(|E0|²)
        # This is different from intensity_sum which is sum(|E/E0|²)
        # Use e_sq_int if available (internal field from chunked version), otherwise use e_sq
        e_sq_to_use = e_sq_int if e_sq_int is not None else e_sq

        if e_sq_to_use is not None and e0_sq is not None:
            e_sq_in_region = e_sq_to_use[final_mask]
            e0_sq_in_region = e0_sq[final_mask]

            # Filter out NaN/Inf
            valid_e_sq = np.isfinite(e_sq_in_region)
            valid_e0_sq = np.isfinite(e0_sq_in_region)
            valid_both = valid_e_sq & valid_e0_sq

            if np.sum(valid_both) > 0:
                e_sq_sum = float(np.sum(e_sq_in_region[valid_both]))
                e0_sq_sum = float(np.sum(e0_sq_in_region[valid_both]))

                if e0_sq_sum > 1e-20:  # Avoid division by zero
                    energy_ratio = e_sq_sum / e0_sq_sum
                    energy_ratio_per_sphere = energy_ratio / n_spheres if n_spheres > 0 else 0.0
                else:
                    energy_ratio = None
                    energy_ratio_per_sphere = None
                    e_sq_sum = None
                    e0_sq_sum = None

                result['e_sq_sum'] = e_sq_sum
                result['e0_sq_sum'] = e0_sq_sum
                result['energy_ratio'] = energy_ratio
                result['energy_ratio_per_sphere'] = energy_ratio_per_sphere

                if self.verbose:
                    print(f"        Energy ratio: sum(|E|²)/sum(|E0|²) = {energy_ratio:.6f}")
                    print(f"        Per-sphere energy ratio: {energy_ratio_per_sphere:.6f}")
            else:
                result['e_sq_sum'] = None
                result['e0_sq_sum'] = None
                result['energy_ratio'] = None
                result['energy_ratio_per_sphere'] = None
        else:
            result['e_sq_sum'] = None
            result['e0_sq_sum'] = None
            result['energy_ratio'] = None
            result['energy_ratio_per_sphere'] = None

        if method == 'conservative':
            result['excluded_outliers'] = excluded_outliers

        if top_percentile_filter is not None and top_percentile_filter > 0:
            result['excluded_top_percentile'] = excluded_top_percentile
            result['top_percentile_filter'] = top_percentile_filter

        return result
    
    def _create_distance_mask(self, x_grid, y_grid, z_grid, spheres, depth):
        """
        Create mask for integration region - INTERIOR VERSION.
        
        Region criteria:
        1. Inside at least ONE sphere
        2. Within 'depth' nm from surface (measured inward)
        
        This selects the region inside particles, near the surface.
        Overlapping regions between particles are included only once (OR operation).
        
        Parameters
        ----------
        x_grid, y_grid, z_grid : ndarray
            Coordinate grids
        spheres : list of tuple
            List of (center_x, center_y, center_z, radius)
        depth : float
            Integration depth from surface (nm)
        
        Returns
        -------
        ndarray (bool)
            True for points in integration region
        """
        shape = x_grid.shape
        
        # Initialize mask - all False
        integration_mask = np.zeros(shape, dtype=bool)

        if self.verbose:
            print(f"      Grid shape: {shape}, total points: {np.prod(shape)}")
        
        for sphere_idx, (cx, cy, cz, radius) in enumerate(spheres):
            # Calculate distance from sphere center
            dist_from_center = np.sqrt(
                (x_grid - cx)**2 + 
                (y_grid - cy)**2 + 
                (z_grid - cz)**2
            )
            
            # Distance from surface (negative = inside, positive = outside)
            dist_from_surface = dist_from_center - radius
            
            # Points inside this sphere AND within depth from surface (inward)
            # -depth <= dist_from_surface <= 0
            inside_near_surface = (
                (dist_from_surface <= 0) &  # inside sphere
                (dist_from_surface >= -depth)  # within depth from surface
            )

            if self.verbose:
                n_inside_total = np.sum(dist_from_surface <= 0)
                n_inside_near = np.sum(inside_near_surface)
                print(f"      Sphere {sphere_idx+1}: {n_inside_near}/{n_inside_total} points in near-surface region")
            
            # OR operation: add to integration mask (overlaps counted once)
            integration_mask = integration_mask | inside_near_surface
        
        if self.verbose:
            n_total = np.prod(shape)
            n_selected = np.sum(integration_mask)
            print(f"    Integration region ({depth:.1f}nm interior): {n_selected}/{n_total} points ({100*n_selected/n_total:.1f}%)")
        
        return integration_mask
    
    def _get_sphere_boundaries(self, config, geometry, center_only=False):
        """
        Get sphere boundaries from configuration.

        Parameters
        ----------
        config : dict
            Simulation configuration
        geometry : GeometryCrossSection
            Geometry calculator
        center_only : bool
            If True, return only center sphere for cluster structures

        Returns
        -------
        list of tuple
            List of (center_x, center_y, center_z, radius) for each sphere
        """
        structure = config.get('structure', 'unknown')

        if structure in ['sphere_cluster_aggregate', 'sphere_cluster']:
            return self._get_cluster_spheres(config, geometry, center_only=center_only)
        elif structure == 'sphere':
            return self._get_single_sphere(config)
        elif structure in ['dimer_sphere', 'dimer']:
            return self._get_dimer_spheres(config)
        else:
            if self.verbose:
                print(f"    [!] Sphere boundary extraction not implemented for '{structure}'")
            return None
    
    def _get_cluster_spheres(self, config, geometry, center_only=False):
        """Get sphere boundaries for sphere cluster aggregate.

        Parameters
        ----------
        config : dict
            Simulation configuration
        geometry : GeometryCrossSection
            Geometry calculator
        center_only : bool
            If True, return only the first sphere (center sphere at origin
            for N>=4, first sphere for N<4)

        Returns
        -------
        list of tuple
            List of (cx, cy, cz, radius)
        """
        n_spheres = config.get('n_spheres', 1)
        diameter = config.get('diameter', 50.0)
        gap = config.get('gap', -0.1)

        radius = diameter / 2
        spacing = diameter + gap

        # Use geometry calculator to get positions (same as MATLAB)
        positions = geometry._calculate_cluster_positions(n_spheres, spacing)

        # Convert to sphere list
        spheres = [(pos[0], pos[1], pos[2], radius) for pos in positions]

        if center_only:
            spheres = [spheres[0]]
            if self.verbose:
                print(f"    Using center sphere only (r={radius:.1f} nm)")
        else:
            if self.verbose:
                print(f"    Using all {len(spheres)} spheres (r={radius:.1f} nm)")

        return spheres
    
    def _get_single_sphere(self, config):
        """Get boundary for single sphere."""
        diameter = config.get('diameter', 50.0)
        radius = diameter / 2
        center = config.get('center', [0, 0, 0])
        
        return [(center[0], center[1], center[2], radius)]
    
    def _get_dimer_spheres(self, config):
        """Get boundaries for dimer spheres."""
        diameter = config.get('diameter', 50.0)
        gap = config.get('gap', 5.0)
        radius = diameter / 2
        
        spacing = diameter + gap
        offset = spacing / 2
        
        # Assume x-axis dimer
        return [
            (-offset, 0, 0, radius),
            (offset, 0, 0, radius)
        ]
    
    def _is_structure_supported_for_integration(self, structure_type):
        """Check if structure is supported for near-field integration."""
        supported = [
            'sphere',
            'sphere_cluster',
            'sphere_cluster_aggregate',
            'dimer_sphere',
            'dimer'
        ]
        return structure_type in supported
    
    def _empty_integration_result(self, method='strict'):
        """Return empty result when no valid points found."""
        result = {
            'enhancement_sum': 0.0,
            'enhancement_mean': 0.0,
            'enhancement_per_sphere': 0.0,
            'intensity_sum': 0.0,
            'intensity_mean': 0.0,
            'intensity_per_sphere': 0.0,
            'e_sq_sum': None,
            'e0_sq_sum': None,
            'energy_ratio': None,
            'energy_ratio_per_sphere': None,
            'valid_points': 0,
        }
        if method == 'conservative':
            result['excluded_outliers'] = 0
        return result
    
    def save_near_field_results(self, results, config, output_path, center_only=False,
                                top_percentile_filter=None):
        """
        Save near-field integration results to text file.

        Parameters
        ----------
        results : dict
            Integration results from calculate_near_field_integration()
        config : dict
            Configuration dictionary
        output_path : str
            Path to output file
        center_only : bool
            If True, write header indicating center sphere only analysis
        top_percentile_filter : float, optional
            If set, indicate that top N% of enhancement values were removed.
        """
        with open(output_path, 'w') as f:
            self._write_integration_header(f, config, center_only=center_only,
                                           top_percentile_filter=top_percentile_filter)
            self._write_integration_results(f, results, top_percentile_filter=top_percentile_filter)
            self._write_integration_summary(f, results)

        if self.verbose:
            mode_str = " (center sphere only)" if center_only else ""
            filter_str = f" (top {top_percentile_filter}% excluded)" if top_percentile_filter else ""
            print(f"\n[OK] Near-field integration results{mode_str}{filter_str} saved: {output_path}")

    def _write_integration_header(self, f, config, center_only=False, top_percentile_filter=None):
        """Write file header for near-field integration results."""
        f.write("=" * 80 + "\n")
        title_parts = ["Near-Field Integration Analysis (INTERIOR)"]
        if center_only:
            title_parts.append("CENTER SPHERE ONLY")
        if top_percentile_filter is not None and top_percentile_filter > 0:
            keep_pct = 100.0 - top_percentile_filter
            title_parts.append(f"TOP {top_percentile_filter}% EXCLUDED (keep <= {keep_pct:.0f}th percentile)")
        f.write(" - ".join(title_parts) + "\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Integration depths: {', '.join([f'{d:.1f}' for d in self.near_field_distances])} nm from particle surface (interior)\n")
        if top_percentile_filter is not None and top_percentile_filter > 0:
            f.write(f"  Enhancement filter: Top {top_percentile_filter}% of E/E0 values excluded\n")

        structure_type = config.get('structure', 'unknown')
        f.write(f"  Structure: {structure_type}\n")

        # Structure-specific info
        if structure_type in ['sphere_cluster_aggregate', 'sphere_cluster']:
            n_spheres = config.get('n_spheres', 1)
            diameter = config.get('diameter', 50.0)
            gap = config.get('gap', -0.1)
            f.write(f"  Total spheres in cluster: {n_spheres}\n")
            if center_only:
                f.write(f"  Integration target: CENTER SPHERE ONLY (sphere #0)\n")
            else:
                f.write(f"  Integration target: ALL SPHERES\n")
            f.write(f"  Sphere diameter: {diameter:.1f} nm\n")
            f.write(f"  Gap: {gap:.3f} nm\n")

        f.write("\n" + "=" * 80 + "\n\n")
    
    def _write_integration_results(self, f, results, top_percentile_filter=None):
        """Write detailed results for each wavelength/polarization."""
        for wl in sorted(results.keys()):
            f.write(f"Results at wavelength = {wl:.1f} nm:\n")
            f.write("\n" + "-" * 80 + "\n")
            
            wl_results = results[wl]
            
            for pol_key in sorted(wl_results.keys()):
                pol_data = wl_results[pol_key]
                
                # Polarization label
                if pol_key == 'unpolarized':
                    pol_label = "Unpolarized (average)"
                else:
                    pol_num = pol_key.split('_')[1]
                    pol_label = f"Polarization {pol_num}"
                
                f.write(f"{pol_label}\n")
                f.write("-" * 80 + "\n\n")
                
                # Grid info (only once per polarization)
                if 'grid_info' in pol_data:
                    grid_info = pol_data['grid_info']
                    f.write("Grid information:\n")
                    f.write(f"  Total grid points:       {grid_info['total_points']}\n")
                    f.write(f"  Valid points (not NaN):  {grid_info['valid_points']}\n\n")
                
                # Results for each depth
                if 'depths' in pol_data:
                    for depth in sorted(pol_data['depths'].keys()):
                        depth_data = pol_data['depths'][depth]
                        n_spheres = depth_data.get('n_spheres', 1)
                        
                        f.write(f"Integration depth: {depth:.1f} nm (interior)\n")
                        f.write(f"  Number of spheres: {n_spheres}\n\n")
                        
                        # Strict filtering results
                        strict = depth_data['strict']
                        f.write("  Strict filtering (Inf only):\n")
                        f.write(f"    Enhancement sum:         {strict['enhancement_sum']:15.3f}  [sum(|E/E0|)]\n")
                        f.write(f"    Enhancement per sphere:  {strict['enhancement_per_sphere']:15.3f}\n")
                        if strict['intensity_sum'] is not None:
                            f.write(f"    Intensity sum:           {strict['intensity_sum']:15.3f}  [sum(|E/E0|^2)]\n")
                            f.write(f"    Intensity per sphere:    {strict['intensity_per_sphere']:15.3f}\n")
                        # Energy ratio: sum(|E|²)/sum(|E0|²)
                        if strict.get('energy_ratio') is not None:
                            f.write(f"    Energy ratio:            {strict['energy_ratio']:15.6f}  [sum(|E|^2)/sum(|E0|^2)]\n")
                            if strict.get('energy_ratio_per_sphere') is not None:
                                f.write(f"    Energy ratio per sphere: {strict['energy_ratio_per_sphere']:15.6f}\n")
                        f.write(f"    Valid points in region:  {strict['valid_points']:15d}\n")
                        if strict.get('excluded_top_percentile') is not None:
                            f.write(f"    Excluded top {strict['top_percentile_filter']}%:  {strict['excluded_top_percentile']:15d}\n")
                        f.write(f"    Mean enhancement:        {strict['enhancement_mean']:15.3f}\n")
                        if strict['intensity_mean'] is not None:
                            f.write(f"    Mean intensity:          {strict['intensity_mean']:15.3f}\n")
                        f.write("\n")

                        # Conservative filtering results
                        cons = depth_data['conservative']
                        f.write("  Conservative filtering (Inf + outliers):\n")
                        f.write(f"    Enhancement sum:         {cons['enhancement_sum']:15.3f}  [sum(|E/E0|)]\n")
                        f.write(f"    Enhancement per sphere:  {cons['enhancement_per_sphere']:15.3f}\n")
                        if cons['intensity_sum'] is not None:
                            f.write(f"    Intensity sum:           {cons['intensity_sum']:15.3f}  [sum(|E/E0|^2)]\n")
                            f.write(f"    Intensity per sphere:    {cons['intensity_per_sphere']:15.3f}\n")
                        # Energy ratio: sum(|E|²)/sum(|E0|²)
                        if cons.get('energy_ratio') is not None:
                            f.write(f"    Energy ratio:            {cons['energy_ratio']:15.6f}  [sum(|E|^2)/sum(|E0|^2)]\n")
                            if cons.get('energy_ratio_per_sphere') is not None:
                                f.write(f"    Energy ratio per sphere: {cons['energy_ratio_per_sphere']:15.6f}\n")
                        f.write(f"    Valid points in region:  {cons['valid_points']:15d}\n")
                        f.write(f"    Excluded outliers:       {cons['excluded_outliers']:15d}\n")
                        if cons.get('excluded_top_percentile') is not None:
                            f.write(f"    Excluded top {cons['top_percentile_filter']}%:  {cons['excluded_top_percentile']:15d}\n")
                        f.write(f"    Mean enhancement:        {cons['enhancement_mean']:15.3f}\n")
                        if cons['intensity_mean'] is not None:
                            f.write(f"    Mean intensity:          {cons['intensity_mean']:15.3f}\n")
                        f.write("\n" + "-" * 80 + "\n")
            
            f.write("\n")
    
    def _write_integration_summary(self, f, results):
        """Write summary table for near-field integration."""
        f.write("=" * 100 + "\n")
        f.write("Summary (Strict Filtering)\n")
        f.write("=" * 100 + "\n\n")

        # Table header
        f.write(f"{'Wavelength':<12} {'Polarization':<15} {'Depth':<8} {'Enh.Sum':<15} {'Int.Sum':<15} {'Energy Ratio':<15} {'Points':<10}\n")
        f.write("-" * 100 + "\n")

        # Table rows
        for wl in sorted(results.keys()):
            wl_results = results[wl]

            for pol_key in sorted(wl_results.keys()):
                pol_data = wl_results[pol_key]

                # Format polarization
                if pol_key == 'unpolarized':
                    pol_str = "unpolarized"
                else:
                    pol_num = pol_key.split('_')[1]
                    pol_str = f"pol{pol_num}"

                # Results for each depth
                if 'depths' in pol_data:
                    for depth in sorted(pol_data['depths'].keys()):
                        depth_data = pol_data['depths'][depth]
                        strict = depth_data['strict']

                        wl_str = f"{wl:.1f} nm"
                        depth_str = f"{depth:.1f}nm"
                        enh_sum = strict['enhancement_sum']
                        int_sum = strict['intensity_sum'] if strict['intensity_sum'] is not None else 0
                        energy_ratio = strict.get('energy_ratio')
                        energy_ratio_str = f"{energy_ratio:.6f}" if energy_ratio is not None else "N/A"
                        points = strict['valid_points']

                        f.write(f"{wl_str:<12} {pol_str:<15} {depth_str:<8} {enh_sum:<15.3f} {int_sum:<15.3f} {energy_ratio_str:<15} {points:<10d}\n")

        f.write("\n" + "=" * 100 + "\n")