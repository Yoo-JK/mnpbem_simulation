"""
Data Loader

Handles loading and saving of simulation data.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


class DataLoader:
    """Loads and saves simulation data."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = Path(config['output_dir'])
        self.output_prefix = config.get('output_prefix', 'simulation')
    
    def load(self):
        """Load simulation results from text file."""
        # Try to load from text file first
        result_file = self.output_dir / f"{self.output_prefix}_results.txt"
        
        if not result_file.exists():
            raise FileNotFoundError(f"Results file not found: {result_file}")
        
        if self.verbose:
            print(f"Loading results from: {result_file}")
        
        # Read data
        data = np.loadtxt(result_file, skiprows=1)
        
        # Parse data
        n_wavelengths = data.shape[0]
        n_cols = data.shape[1]
        
        # First column is wavelength
        wavelength = data[:, 0]
        
        # Determine number of polarizations
        # Columns: wavelength, sca1, sca2, ..., ext1, ext2, ..., abs1, abs2, ...
        n_polarizations = (n_cols - 1) // 3
        
        # Extract data
        sca_cols = list(range(1, 1 + n_polarizations))
        ext_cols = list(range(1 + n_polarizations, 1 + 2 * n_polarizations))
        abs_cols = list(range(1 + 2 * n_polarizations, 1 + 3 * n_polarizations))
        
        scattering = data[:, sca_cols]
        extinction = data[:, ext_cols]
        absorption = data[:, abs_cols]
        
        result = {
            'wavelength': wavelength,
            'scattering': scattering,
            'extinction': extinction,
            'absorption': absorption,
            'n_polarizations': n_polarizations
        }
        
        if self.verbose:
            print(f"  Wavelengths: {n_wavelengths}")
            print(f"  Polarizations: {n_polarizations}")
        
        return result
    
    def save_txt(self, data, analysis=None):
        """Save data in text format with analysis."""
        output_file = self.output_dir / f"{self.output_prefix}_processed.txt"
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("# MNPBEM Simulation Results - Processed\n")
            f.write(f"# Structure: {self.config['structure']}\n")
            f.write(f"# Simulation type: {self.config['simulation_type']}\n")
            f.write("#\n")
            
            if analysis is not None:
                f.write("# Analysis Results:\n")
                if 'peak_wavelengths' in analysis:
                    for i, wl in enumerate(analysis['peak_wavelengths']):
                        f.write(f"#   Peak wavelength (pol {i+1}): {wl:.2f} nm\n")
                if 'peak_values' in analysis:
                    for i, val in enumerate(analysis['peak_values']):
                        f.write(f"#   Peak scattering (pol {i+1}): {val:.6e} nm^2\n")
                f.write("#\n")
            
            # Write column headers
            f.write("Wavelength(nm)\t")
            for i in range(data['n_polarizations']):
                f.write(f"Sca{i+1}(nm2)\t")
            for i in range(data['n_polarizations']):
                f.write(f"Ext{i+1}(nm2)\t")
            for i in range(data['n_polarizations']):
                if i < data['n_polarizations'] - 1:
                    f.write(f"Abs{i+1}(nm2)\t")
                else:
                    f.write(f"Abs{i+1}(nm2)\n")
            
            # Write data
            for i in range(len(data['wavelength'])):
                f.write(f"{data['wavelength'][i]:.4f}\t")
                for j in range(data['n_polarizations']):
                    f.write(f"{data['scattering'][i, j]:.6e}\t")
                for j in range(data['n_polarizations']):
                    f.write(f"{data['extinction'][i, j]:.6e}\t")
                for j in range(data['n_polarizations']):
                    if j < data['n_polarizations'] - 1:
                        f.write(f"{data['absorption'][i, j]:.6e}\t")
                    else:
                        f.write(f"{data['absorption'][i, j]:.6e}\n")
        
        if self.verbose:
            print(f"  Saved to: {output_file}")
    
    def save_csv(self, data, analysis=None):
        """Save data in CSV format."""
        output_file = self.output_dir / f"{self.output_prefix}_processed.csv"
        
        # Create DataFrame
        df_dict = {'Wavelength_nm': data['wavelength']}
        
        for i in range(data['n_polarizations']):
            df_dict[f'Scattering_pol{i+1}_nm2'] = data['scattering'][:, i]
            df_dict[f'Extinction_pol{i+1}_nm2'] = data['extinction'][:, i]
            df_dict[f'Absorption_pol{i+1}_nm2'] = data['absorption'][:, i]
        
        df = pd.DataFrame(df_dict)
        df.to_csv(output_file, index=False)
        
        if self.verbose:
            print(f"  Saved to: {output_file}")
    
    def save_json(self, data, analysis=None):
        """Save data and analysis in JSON format."""
        output_file = self.output_dir / f"{self.output_prefix}_processed.json"
        
        # Prepare data for JSON
        json_data = {
            'metadata': {
                'structure': self.config['structure'],
                'simulation_type': self.config['simulation_type'],
                'excitation_type': self.config['excitation_type'],
                'n_wavelengths': len(data['wavelength']),
                'n_polarizations': data['n_polarizations']
            },
            'wavelength_nm': data['wavelength'].tolist(),
            'scattering_nm2': data['scattering'].tolist(),
            'extinction_nm2': data['extinction'].tolist(),
            'absorption_nm2': data['absorption'].tolist()
        }
        
        if analysis is not None:
            json_data['analysis'] = {}
            for key, value in analysis.items():
                if isinstance(value, np.ndarray):
                    json_data['analysis'][key] = value.tolist()
                else:
                    json_data['analysis'][key] = value
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"  Saved to: {output_file}")