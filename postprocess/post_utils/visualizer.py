"""
Visualizer

Creates plots and visualizations of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Visualizer:
    """Creates visualizations of simulation results."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = Path(config['output_dir'])
        self.output_prefix = config.get('output_prefix', 'simulation')
        self.plot_formats = config.get('plot_format', ['png'])
        self.dpi = config.get('plot_dpi', 300)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_spectrum(self, data, analysis=None):
        """
        Plot complete optical spectrum.
        
        Args:
            data (dict): Data dictionary
            analysis (dict): Analysis results (optional)
        """
        n_pol = data['n_polarizations']
        wavelength = data['wavelength']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Polarization labels
        pol_labels = self._get_polarization_labels(n_pol)
        colors = plt.cm.tab10(np.linspace(0, 1, n_pol))
        
        # Plot scattering
        ax = axes[0]
        for i in range(n_pol):
            ax.plot(wavelength, data['scattering'][:, i], 
                   label=pol_labels[i], color=colors[i], linewidth=2)
            
            # Mark peaks if analysis provided
            if analysis is not None and 'peak_wavelengths' in analysis:
                peak_wl = analysis['peak_wavelengths'][i]
                peak_val = analysis['peak_values'][i]
                ax.plot(peak_wl, peak_val, 'o', color=colors[i], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize=12)
        ax.set_title('Absorption Spectrum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, 'spectrum')
        
        if self.verbose:
            print("  Spectrum plot saved")
    
    def plot_polarization_comparison(self, data):
        """
        Plot comparison between different polarizations.
        
        Args:
            data (dict): Data dictionary
        """
        if data['n_polarizations'] < 2:
            return
        
        wavelength = data['wavelength']
        n_pol = data['n_polarizations']
        pol_labels = self._get_polarization_labels(n_pol)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_pol))
        
        for i in range(n_pol):
            ax.plot(wavelength, data['scattering'][:, i],
                   label=pol_labels[i], color=colors[i], 
                   linewidth=2.5, marker='o', markersize=3, 
                   markevery=max(1, len(wavelength)//20))
        
        ax.set_xlabel('Wavelength (nm)', fontsize=13)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize=13)
        ax.set_title('Polarization-Dependent Scattering', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add text box with info
        structure = self.config['structure']
        textstr = f"Structure: {structure.replace('_', ' ').title()}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, 'polarization_comparison')
        
        if self.verbose:
            print("  Polarization comparison plot saved")
    
    def plot_enhancement_map(self, data, analysis):
        """
        Plot enhancement factor map (if applicable).
        
        Args:
            data (dict): Data dictionary
            analysis (dict): Analysis results
        """
        if 'enhancement_factors' not in analysis:
            return
        
        enhancement = analysis['enhancement_factors']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = list(enhancement.keys())
        values = list(enhancement.values())
        
        bars = ax.bar(labels, values, color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Enhancement Factor', fontsize=13)
        ax.set_title('Polarization Enhancement Factors', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, 'enhancement_factors')
        
        if self.verbose:
            print("  Enhancement factor plot saved")
    
    def _get_polarization_labels(self, n_pol):
        """Generate polarization labels."""
        if self.config['excitation_type'] == 'planewave':
            # Try to use configured polarizations
            if 'polarizations' in self.config:
                pols = self.config['polarizations']
                labels = []
                for pol in pols:
                    if pol == [1, 0, 0] or pol == (1, 0, 0):
                        labels.append('x-polarization')
                    elif pol == [0, 1, 0] or pol == (0, 1, 0):
                        labels.append('y-polarization')
                    elif pol == [0, 0, 1] or pol == (0, 0, 1):
                        labels.append('z-polarization')
                    else:
                        labels.append(f'pol-{len(labels)+1}')
                return labels
        
        # Default labels
        return [f'Polarization {i+1}' for i in range(n_pol)]
    
    def _save_figure(self, fig, name):
        """Save figure in configured formats."""
        for fmt in self.plot_formats:
            output_file = self.output_dir / f"{self.output_prefix}_{name}.{fmt}"
            
            if fmt in ['png', 'jpg', 'jpeg']:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            else:
                fig.savefig(output_file, bbox_inches='tight')
            
            if self.verbose:
                print(f"    Saved: {output_file}")
        
        plt.close(fig)('Scattering Cross Section (nm²)', fontsize=12)
        ax.set_title('Scattering Spectrum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot extinction
        ax = axes[1]
        for i in range(n_pol):
            ax.plot(wavelength, data['extinction'][:, i],
                   label=pol_labels[i], color=colors[i], linewidth=2)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=12)
        ax.set_title('Extinction Spectrum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot absorption
        ax = axes[2]
        for i in range(n_pol):
            ax.plot(wavelength, data['absorption'][:, i],
                   label=pol_labels[i], color=colors[i], linewidth=2)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel