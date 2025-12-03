import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = os.path.join(config.get('output_dir'), config.get('simulation_name'))
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', ['png', 'pdf'])
        self.dpi = config.get('plot_dpi', 300)
        self.polarizations = config.get('polarizations', [])
        self.propagation_dirs = config.get('propagation_dirs', [])
    
    def _format_vector_label(self, vec):
        """Format vector as compact string."""
        if vec is None or len(vec) == 0:
            return ""
        vec_rounded = np.round(vec, 3)
        return f"[{vec_rounded[0]:.0f} {vec_rounded[1]:.0f} {vec_rounded[2]:.0f}]"

    def _get_polarization_label(self, ipol):
        """Get descriptive label for polarization."""
        if ipol < len(self.polarizations) and ipol < len(self.propagation_dirs):
            pol_vec = self.polarizations[ipol]
            prop_vec = self.propagation_dirs[ipol]
            pol_str = self._format_vector_label(pol_vec)
            prop_str = self._format_vector_label(prop_vec)
            return f"Pol{pol_str} Prop{prop_str}"
        else:
            return f"Polarization {ipol+1}"
    
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
        """Plot extinction, scattering, and absorption spectra for each polarization."""
        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        
        # Check x-axis unit preference
        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')
        
        # Convert to energy if requested
        if xaxis_unit == 'energy':
            # E(eV) = 1239.84 / λ(nm)
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            # Reverse order for energy (high energy = short wavelength)
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'
        
        n_pol = extinction.shape[1]
        saved_files_all = []
        
        # Create separate plot for each polarization
        for ipol in range(n_pol):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(xdata, extinction[:, ipol], 'b-', linewidth=2, label='Extinction')
            ax.plot(xdata, scattering[:, ipol], 'r--', linewidth=2, label='Scattering')
            ax.plot(xdata, absorption[:, ipol], 'g:', linewidth=2, label='Absorption')
            
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel('Cross Section (nm²)', fontsize=12)

            # ✅ FIX: Don't duplicate "Polarization"
            pol_label = self._get_polarization_label(ipol)
            ax.set_title(f'Optical Spectra - {pol_label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Reverse x-axis for energy to show high energy on left
            if xaxis_unit == 'energy':
                ax.invert_xaxis()
            
            plt.tight_layout()
            
            # Save plot with polarization index
            base_filename = f'simulation_spectrum_pol{ipol+1}'
            saved_files = self._save_figure(fig, base_filename)
            if saved_files:
                saved_files_all.extend(saved_files)
            plt.close(fig)
        
        return saved_files_all
    
    def plot_polarization_comparison(self, data):
        """Plot comparison of different polarizations for all cross sections."""
        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        n_pol = extinction.shape[1]
        
        # Check x-axis unit preference
        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')
        
        # Convert to energy if requested
        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            # Reverse order for energy
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_pol))
        saved_files_all = []
        
        # ========== Extinction Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            # ✅ FIX: Use polarization label
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, extinction[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Extinction', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)  # Smaller for long labels
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_extinction'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        # ========== Scattering Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, scattering[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Scattering', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_scattering'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        # ========== Absorption Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, absorption[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Absorption', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_absorption'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        return saved_files_all
    
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
        
        # ✅ FIX: Handle scalar enhancement
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([[enhancement]])
        elif enhancement.ndim == 0:
            enhancement = np.array([[enhancement.item()]])
        elif enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)
        
        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        
        # ✅ FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        im1 = ax1.imshow(enhancement, extent=extent, origin='lower', 
                        cmap='hot', aspect='auto')
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_title(f'Field Enhancement (Linear)\nλ = {wavelength:.1f} nm, {pol_label}', 
                     fontsize=11, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|E|/|E₀|', fontsize=11)
        
        # Log scale
        enhancement_log = np.maximum(enhancement, 1e-10)
        enh_max = enhancement.max()
        enh_min = enhancement_log[enhancement_log > 0].min() if np.any(enhancement_log > 0) else 1e-10
        
        if enh_max > enh_min:
            vmin_log = max(enh_min, 1e-2)
            vmax_log = enh_max
            
            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 10
            
            im2 = ax2.imshow(enhancement_log, extent=extent, origin='lower', 
                            cmap='hot', aspect='auto', 
                            norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement (Log Scale)\nλ = {wavelength:.1f} nm, {pol_label}', 
                         fontsize=11, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('|E|/|E₀|', fontsize=11)
        else:
            im2 = ax2.imshow(enhancement, extent=extent, origin='lower', 
                            cmap='hot', aspect='auto')
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement\nλ = {wavelength:.1f} nm, {pol_label}', 
                         fontsize=11, fontweight='bold')
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
        
        # ✅ FIX: Handle scalar intensity
        if not isinstance(intensity, np.ndarray):
            intensity = np.array([[intensity]])
        elif intensity.ndim == 0:
            intensity = np.array([[intensity.item()]])
        elif intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)
        
        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        
        # ✅ FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7))
        
        intensity_log = np.maximum(intensity, 1e-10)
        int_max = intensity.max()
        int_min = intensity_log[intensity_log > 0].min() if np.any(intensity_log > 0) else 1e-10
        
        if int_max > int_min and int_max > 0:
            vmin_log = max(int_min, int_max / 1e6)
            vmax_log = int_max
            
            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 10
            
            im = ax.imshow(intensity_log, extent=extent, origin='lower', 
                          cmap='viridis', aspect='auto', 
                          norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
        else:
            im = ax.imshow(intensity, extent=extent, origin='lower', 
                          cmap='viridis', aspect='auto')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Field Intensity |E|² (Log Scale)\nλ = {wavelength:.1f} nm, {pol_label}', 
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
        
        # ✅ FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)
        
        # Properly extract coordinates and field components
        if plane_type == 'xz':
            x_coord = x_grid[0, :]
            z_coord = z_grid[:, 0]
            e_x = e_total[:, :, 0].real
            e_z = e_total[:, :, 2].real
            x_plot = x_coord
            y_plot = z_coord
            U = e_x
            V = e_z
            
        elif plane_type == 'xy':
            x_coord = x_grid[0, :]
            y_coord = y_grid[:, 0]
            e_x = e_total[:, :, 0].real
            e_y = e_total[:, :, 1].real
            x_plot = x_coord
            y_plot = y_coord
            U = e_x
            V = e_y
            
        elif plane_type == 'yz':
            y_coord = y_grid[:, 0]
            z_coord = z_grid[0, :]
            e_y = e_total[:, :, 1].real
            e_z = e_total[:, :, 2].real
            x_plot = y_coord
            y_plot = z_coord
            U = e_y
            V = e_z
            
        else:
            return []
        
        # Downsample for vector plot
        nx, ny = U.shape
        skip_x = max(1, nx // 15)
        skip_y = max(1, ny // 15)
        
        x_down = x_plot[::skip_y]
        y_down = y_plot[::skip_x]
        
        X, Y = np.meshgrid(x_down, y_down)
        
        U_down = U[::skip_x, ::skip_y]
        V_down = V[::skip_x, ::skip_y]
        
        magnitude = np.sqrt(U_down**2 + V_down**2)
        magnitude_max = np.max(magnitude)
        
        if magnitude_max > 1e-10:
            U_norm = U_down / (magnitude + 1e-10)
            V_norm = V_down / (magnitude + 1e-10)
        else:
            U_norm = U_down
            V_norm = V_down
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(enhancement, extent=extent, origin='lower', 
                      cmap='hot', aspect='auto', alpha=0.7)
        
        q = ax.quiver(X, Y, U_norm, V_norm, magnitude,
                      cmap='cool', scale=25, width=0.004, 
                      alpha=0.9, pivot='middle')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Electric Field Vectors\nλ = {wavelength:.1f} nm, {pol_label}', 
                    fontsize=12, fontweight='bold')
        
        cbar1 = plt.colorbar(im, ax=ax, pad=0.12, label='|E|/|E₀|')
        cbar2 = plt.colorbar(q, ax=ax, label='Field Magnitude')
        
        plt.tight_layout()
        
        # Save
        base_filename = f'field_vectors_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)
        
        return saved_files
    
    def _determine_plane(self, x_grid, y_grid, z_grid):
        """Determine which 2D plane is being plotted."""
        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([[x_grid]])
            y_grid = np.array([[y_grid]])
            z_grid = np.array([[z_grid]])
        
        if x_grid.ndim == 0:
            x_grid = np.array([[x_grid.item()]])
            y_grid = np.array([[y_grid.item()]])
            z_grid = np.array([[z_grid.item()]])
        
        if x_grid.ndim == 1:
            x_grid = x_grid.reshape(1, -1)
            y_grid = y_grid.reshape(1, -1)
            z_grid = z_grid.reshape(1, -1)
        
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
        if y_constant:
            plane_type = 'xz'
            x_min, x_max = x_grid.min(), x_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            # ✅ FIX: Extend extent if single point
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if z_min == z_max:
                z_min -= 0.5
                z_max += 0.5
            extent = [x_min, x_max, z_min, z_max]
            x_label = 'x (nm)'
            y_label = 'z (nm)'
        elif z_constant:
            plane_type = 'xy'
            x_min, x_max = x_grid.min(), x_grid.max()
            y_min, y_max = y_grid.min(), y_grid.max()
            # ✅ FIX: Extend extent if single point
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            extent = [x_min, x_max, y_min, y_max]
            x_label = 'x (nm)'
            y_label = 'y (nm)'
        elif x_constant:
            plane_type = 'yz'
            y_min, y_max = y_grid.min(), y_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            # ✅ FIX: Extend extent if single point
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            if z_min == z_max:
                z_min -= 0.5
                z_max += 0.5
            extent = [y_min, y_max, z_min, z_max]
            x_label = 'y (nm)'
            y_label = 'z (nm)'
        else:
            plane_type = '3d'
            x_min, x_max = x_grid.min(), x_grid.max()
            y_min, y_max = y_grid.min(), y_grid.max()
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            extent = [x_min, x_max, y_min, y_max]
            x_label = 'x (nm)'
            y_label = 'y (nm)'
        
        return plane_type, extent, x_label, y_label

    def _is_2d_slice(self, field_data):
        """Check if field data is a 2D slice (for vector plots)."""
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        
        # ✅ FIX: Handle scalars
        if not isinstance(x_grid, np.ndarray):
            return False  # Single point, not a slice
        
        if x_grid.ndim == 0:
            return False  # Single point
        
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
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

