"""
Data Loading Utilities

Loads and parses MATLAB output files.
"""

import numpy as np
import scipy.io as sio
import os


class DataLoader:
    """Handles loading of MATLAB output files."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = config.get('output_dir', './results')
    
    def load_simulation_results(self):
        """Load simulation results from MATLAB output."""
        mat_file = os.path.join(self.output_dir, 'simulation_results.mat')
        
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Results file not found: {mat_file}")
        
        if self.verbose:
            print(f"Loading results from: {mat_file}")
        
        # Load MATLAB file
        mat_data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        results = mat_data['results']
        
        # Extract data
        data = {
            'wavelength': self._extract_array(results.wavelength),
            'scattering': self._extract_array(results.scattering),
            'extinction': self._extract_array(results.extinction),
            'absorption': self._extract_array(results.absorption),
            'polarizations': self._extract_array(results.polarizations),
            'propagation_dirs': self._extract_array(results.propagation_dirs),
        }
        
        # Add calculation time if available
        if hasattr(results, 'calculation_time'):
            data['calculation_time'] = float(results.calculation_time)
        
        # Ensure 2D arrays for cross sections
        for key in ['scattering', 'extinction', 'absorption']:
            if data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)
        
        # ✅ FIX: Add n_polarizations to data dictionary
        data['n_polarizations'] = data['scattering'].shape[1]
        
        if self.verbose:
            print(f"  Loaded {len(data['wavelength'])} wavelength points")
            print(f"  Polarizations: {data['n_polarizations']}")
        
        # Load field data if available
        if hasattr(results, 'fields'):
            data['fields'] = self._load_field_data(results.fields)
            if self.verbose and data['fields']:
                print(f"  Field data loaded: {len(data['fields'])} polarization(s)")
        
        return data
    
    def _load_field_data(self, fields_struct):
        """Load electromagnetic field data from MATLAB structure."""
        if fields_struct is None:
            return []
        
        # Handle single polarization vs multiple
        if not isinstance(fields_struct, np.ndarray):
            fields_struct = [fields_struct]
        
        field_data_list = []
        
        for field_item in fields_struct:
            field_dict = {
                'wavelength': float(field_item.wavelength),
                'polarization': self._extract_array(field_item.polarization),
                'x_grid': self._extract_array(field_item.x_grid),
                'y_grid': self._extract_array(field_item.y_grid),
                'z_grid': self._extract_array(field_item.z_grid),
            }
            
            # Electric field components
            if hasattr(field_item, 'e_total'):
                field_dict['e_total'] = self._extract_array(field_item.e_total)
            
            if hasattr(field_item, 'e_induced'):
                field_dict['e_induced'] = self._extract_array(field_item.e_induced)
            
            if hasattr(field_item, 'e_incoming'):
                field_dict['e_incoming'] = self._extract_array(field_item.e_incoming)
            
            # Enhancement and intensity
            if hasattr(field_item, 'enhancement'):
                field_dict['enhancement'] = self._extract_array(field_item.enhancement)
            
            if hasattr(field_item, 'intensity'):
                field_dict['intensity'] = self._extract_array(field_item.intensity)
            
            field_data_list.append(field_dict)
        
        return field_data_list
    
    def _extract_array(self, matlab_array):
        """Extract numpy array from MATLAB data."""
        if matlab_array is None:
            return np.array([])
        
        # Convert to numpy array if not already
        arr = np.array(matlab_array)
        
        # Handle scalar
        if arr.ndim == 0:
            return arr.item()
        
        return arr
    
    def load_text_results(self):
        """Load results from text file (backup method)."""
        txt_file = os.path.join(self.output_dir, 'simulation_results.txt')
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Results file not found: {txt_file}")
        
        if self.verbose:
            print(f"Loading results from text file: {txt_file}")
        
        # Load data
        data_array = np.loadtxt(txt_file, skiprows=1)
        
        # Parse based on number of columns
        n_cols = data_array.shape[1]
        n_pol = (n_cols - 1) // 3
        
        data = {
            'wavelength': data_array[:, 0],
            'scattering': data_array[:, 1:n_pol+1],
            'extinction': data_array[:, n_pol+1:2*n_pol+1],
            'absorption': data_array[:, 2*n_pol+1:],
            'n_polarizations': n_pol,  # ✅ FIX: Add this here too
        }
        
        if self.verbose:
            print(f"  Loaded {len(data['wavelength'])} wavelength points")
            print(f"  Polarizations: {n_pol}")
        
        return data