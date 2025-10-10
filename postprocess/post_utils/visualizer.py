"""
Visualization Utilities

Creates plots and visualizations for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = config.get('output_dir', './results')
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', ['png', 'pdf'])
        self.dpi = config.get('plot_dpi', 300)
    
    def create_all_plots(self, data):
        """Create all visualization plots."""
        plots_created = []
        
        # Spectrum plots
        if 'wavelength' in data and 'extinction' in data:
            spectrum_file = self.plot_spectrum(data)
            plots_created.append(spectrum_file)
        
        # Polarization comparison
        if 'wavelength' in data and data['extinction'].shape[1] > 1:
            pol_file = self.plot_polarization_comparison(data)
            plots_created.append(pol_file)
        
        # Field plots (NEW)
        if 'fields' in data:
            field_files = self.plot_fields(data)
            plots_created.extend(field_files)
        
        return plots_created
    
    def plot_spectrum(self, data):
        """Plot extinction, scattering, and absorption spectra."""
        wavelength = data['wavelength']
        extinction = data['extinction'][:, 0]  # First polarization
        scattering = data['scattering'][:, 0]
        absorption = data['absorption'][:, 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(wavelength, extinction, 'b-', linewidth=2, label='Extinction')
        ax.plot(wavelength, scattering, 'r--', linewidth=2, label='Scattering')
        ax.plot(wavelength, absorption, 'g:', linewidth=2, label='Absorption')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Cross Section (nm²)', fontsize=12)
        ax.set_title('Optical Spectra', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        base_filename = 'simulation_spectrum'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files[0] if saved_files else None
    
    def plot_polarization_comparison(self, data):
        """Plot comparison of different polarizations."""
        wavelength = data['wavelength']
        extinction = data['extinction']
        n_pol = extinction.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_pol))
        
        for i in range(n_pol):
            ax.plot(wavelength, extinction[:, i], 
                   color=colors[i], linewidth=2, 
                   label=f'Polarization {i+1}')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        base_filename = 'simulation_polarization_comparison'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files[0] if saved_files else None
    
    def plot_fields(self, data):
        """Plot electromagnetic field distributions."""
        if 'fields' not in data or not data['fields']:
            return []
        
        fields = data['fields']
        saved_files = []
        
        # Plot for each polarization
        for ipol, field_data in enumerate(fields):
            # Enhancement plot
            enhancement_file = self._plot_field_enhancement(field_data, ipol)
            if enhancement_file:
                saved_files.extend(enhancement_file)
            
            # Intensity plot
            intensity_file = self._plot_field_intensity(field_data, ipol)
            if intensity_file:
                saved_files.extend(intensity_file)
            
            # Vector field plot (optional, for 2D slices)
            if self._is_2d_slice(field_data):
                vector_file = self._plot_field_vectors(field_data, ipol)
                if vector_file:
                    saved_files.extend(vector_file)
        
        return saved_files
    
    def _plot_field_enhancement(self, field_data, polarization_idx):
        """Plot field enhancement |E|/|E0|."""
        enhancement = field_data['enhancement']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        
        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        im1 = ax1.imshow(enhancement, extent=extent, origin='lower', 
                        cmap='hot', aspect='auto')
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_title(f'Field Enhancement (Linear)\nλ = {wavelength:.1f} nm, Pol {polarization_idx+1}', 
                     fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|E|/|E₀|', fontsize=11)
        
        # Log scale
        # ✅ FIX: Properly handle vmin and vmax for LogNorm
        enhancement_log = np.maximum(enhancement, 1e-10)
        enh_max = enhancement.max()
        enh_min = enhancement_log[enhancement_log > 0].min() if np.any(enhancement_log > 0) else 1e-10
        
        # Ensure vmin < vmax for LogNorm
        if enh_max > enh_min:
            vmin_log = max(enh_min, 1e-2)  # Use reasonable lower bound
            vmax_log = enh_max
            
            # Ensure vmin < vmax
            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 10
            
            im2 = ax2.imshow(enhancement_log, extent=extent, origin='lower', 
                            cmap='hot', aspect='auto', 
                            norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement (Log Scale)\nλ = {wavelength:.1f} nm, Pol {polarization_idx+1}', 
                         fontsize=12, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('|E|/|E₀|', fontsize=11)
        else:
            # If all values are the same, just show linear scale
            im2 = ax2.imshow(enhancement, extent=extent, origin='lower', 
                            cmap='hot', aspect='auto')
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement\nλ = {wavelength:.1f} nm, Pol {polarization_idx+1}', 
                         fontsize=12, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('|E|/|E₀|', fontsize=11)
        
        plt.tight_layout()
        
        # Save
        base_filename = f'field_enhancement_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files
    
    def _plot_field_intensity(self, field_data, polarization_idx):
        """Plot field intensity |E|²."""
        intensity = field_data['intensity']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        
        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # ✅ FIX: Properly handle vmin and vmax for LogNorm
        intensity_log = np.maximum(intensity, 1e-10)
        int_max = intensity.max()
        int_min = intensity_log[intensity_log > 0].min() if np.any(intensity_log > 0) else 1e-10
        
        # Ensure vmin < vmax for LogNorm
        if int_max > int_min and int_max > 0:
            vmin_log = max(int_min, int_max / 1e6)  # Dynamic range
            vmax_log = int_max
            
            # Ensure vmin < vmax
            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 10
            
            im = ax.imshow(intensity_log, extent=extent, origin='lower', 
                          cmap='viridis', aspect='auto', 
                          norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
        else:
            # Fallback to linear scale if log doesn't work
            im = ax.imshow(intensity, extent=extent, origin='lower', 
                          cmap='viridis', aspect='auto')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Field Intensity |E|² (Log Scale)\nλ = {wavelength:.1f} nm, Pol {polarization_idx+1}', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|E|² (a.u.)', fontsize=11)
        
        plt.tight_layout()
        
        # Save
        base_filename = f'field_intensity_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files
    
    def _plot_field_vectors(self, field_data, polarization_idx):
        """Plot electric field vector arrows (for 2D slices)."""
        e_total = field_data['e_total']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        enhancement = field_data['enhancement']
        
        # Determine plane and extract relevant components
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        
        # Get field components for the plane
        if plane_type == 'xz':
            # xz-plane: use Ex and Ez
            e_x = e_total[:, :, 0].real
            e_z = e_total[:, :, 2].real
            x_coord = x_grid[:, 0]
            y_coord = z_grid[0, :]
        elif plane_type == 'xy':
            # xy-plane: use Ex and Ey
            e_x = e_total[:, :, 0].real
            e_z = e_total[:, :, 1].real  # Actually Ey
            x_coord = x_grid[:, 0]
            y_coord = y_grid[0, :]
        else:
            # Can't plot vectors for 3D
            return []
        
        # Downsample for vector plot (too many arrows)
        skip = max(1, len(x_coord) // 20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Background: enhancement
        im = ax.imshow(enhancement, extent=extent, origin='lower', 
                      cmap='hot', aspect='auto', alpha=0.7)
        
        # Vector field (downsampled)
        X, Y = np.meshgrid(x_coord[::skip], y_coord[::skip])
        U = e_x[::skip, ::skip]
        V = e_z[::skip, ::skip]
        
        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        magnitude_max = magnitude.max()
        if magnitude_max > 0:
            U_norm = U / magnitude_max
            V_norm = V / magnitude_max
        else:
            U_norm = U
            V_norm = V
        
        ax.quiver(X, Y, U_norm, V_norm, magnitude, 
                 cmap='cool', scale=15, width=0.003, alpha=0.8)
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Electric Field Vectors\nλ = {wavelength:.1f} nm, Pol {polarization_idx+1}', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|E|/|E₀|', fontsize=11)
        
        plt.tight_layout()
        
        # Save
        base_filename = f'field_vectors_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files
    
    def _determine_plane(self, x_grid, y_grid, z_grid):
        """Determine which 2D plane is being plotted."""
        # Check which coordinate is constant (single value)
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
        if y_constant:
            # xz-plane
            plane_type = 'xz'
            x_min, x_max = x_grid.min(), x_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            extent = [x_min, x_max, z_min, z_max]
            x_label = 'x (nm)'
            y_label = 'z (nm)'
        elif z_constant:
            # xy-plane
            plane_type = 'xy'
            x_min, x_max = x_grid.min(), x_grid.max()
            y_min, y_max = y_grid.min(), y_grid.max()
            extent = [x_min, x_max, y_min, y_max]
            x_label = 'x (nm)'
            y_label = 'y (nm)'
        elif x_constant:
            # yz-plane
            plane_type = 'yz'
            y_min, y_max = y_grid.min(), y_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            extent = [y_min, y_max, z_min, z_max]
            x_label = 'y (nm)'
            y_label = 'z (nm)'
        else:
            # 3D data or unknown
            plane_type = '3d'
            extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
            x_label = 'x (nm)'
            y_label = 'y (nm)'
        
        return plane_type, extent, x_label, y_label
    
    def _is_2d_slice(self, field_data):
        """Check if field data is a 2D slice (for vector plots)."""
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        
        # Check if any dimension is constant
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
        # It's a 2D slice if exactly one dimension is constant
        return sum([x_constant, y_constant, z_constant]) == 1
    
    def _save_figure(self, fig, base_filename):
        """Save figure in specified formats."""
        saved_files = []
        
        for fmt in self.plot_format:
            filepath = os.path.join(self.output_dir, f'{base_filename}.{fmt}')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(filepath)
            
            if self.verbose:
                print(f"  Saved: {filepath}")
        
        return saved_files