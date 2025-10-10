"""
Field Data Export Utilities

Exports field data to JSON format.
"""

import json
import numpy as np
import os


class FieldExporter:
    """Exports field data to various formats."""
    
    def __init__(self, output_dir, verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
    
    def export_to_json(self, field_data_list, field_analysis_list):
        """
        Export field data and analysis to JSON.
        
        Parameters
        ----------
        field_data_list : list of dict
            List of field data for each polarization
        field_analysis_list : list of dict
            List of field analysis results
        """
        json_data = {
            'metadata': {
                'num_polarizations': len(field_data_list),
                'description': 'Electromagnetic field distribution data'
            },
            'fields': []
        }
        
        for i, (field_data, analysis) in enumerate(zip(field_data_list, field_analysis_list)):
            field_dict = {
                'polarization_index': i + 1,
                'polarization': field_data['polarization'].tolist() if hasattr(field_data['polarization'], 'tolist') else field_data['polarization'],
                'wavelength_nm': float(field_data['wavelength']),
                
                # Grid information
                'grid': self._extract_grid_info(field_data),
                
                # Analysis results
                'analysis': analysis,
                
                # Note about full field data
                'note': 'Full field arrays (enhancement, intensity, E-field components) are in field_data.mat'
            }
            
            json_data['fields'].append(field_dict)
        
        # Save to file
        filepath = os.path.join(self.output_dir, 'field_analysis.json')
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"  Saved: {filepath}")
        
        return filepath
    
    def export_field_data_arrays(self, field_data_list):
        """
        Export field arrays (enhancement, intensity) to separate JSON.
        WARNING: This can create very large files!
        
        Only exports downsampled data to keep file size reasonable.
        """
        # Downsample factor
        downsample = 4
        
        json_data = {
            'metadata': {
                'warning': 'Data is downsampled by factor of ' + str(downsample),
                'note': 'Full resolution data is in field_data.mat'
            },
            'fields': []
        }
        
        for i, field_data in enumerate(field_data_list):
            # Downsample arrays
            enhancement = field_data['enhancement'][::downsample, ::downsample]
            x_grid = field_data['x_grid'][::downsample, ::downsample]
            y_grid = field_data['y_grid'][::downsample, ::downsample]
            z_grid = field_data['z_grid'][::downsample, ::downsample]
            
            field_dict = {
                'polarization_index': i + 1,
                'wavelength_nm': float(field_data['wavelength']),
                'x_coordinates': x_grid.tolist(),
                'y_coordinates': y_grid.tolist(),
                'z_coordinates': z_grid.tolist(),
                'enhancement': enhancement.tolist(),
            }
            
            json_data['fields'].append(field_dict)
        
        # Save to file
        filepath = os.path.join(self.output_dir, 'field_data_downsampled.json')
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"  Saved (downsampled): {filepath}")
        
        return filepath
    
    def _extract_grid_info(self, field_data):
        """Extract grid information without full arrays."""
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        
        grid_info = {
            'shape': list(x_grid.shape),
            'x_range': [float(x_grid.min()), float(x_grid.max())],
            'y_range': [float(y_grid.min()), float(y_grid.max())],
            'z_range': [float(z_grid.min()), float(z_grid.max())],
        }
        
        # Grid spacing
        if x_grid.ndim == 2 and x_grid.shape[1] > 1:
            grid_info['x_spacing'] = float(np.abs(x_grid[0, 1] - x_grid[0, 0]))
        if y_grid.ndim == 2 and y_grid.shape[0] > 1:
            grid_info['y_spacing'] = float(np.abs(y_grid[1, 0] - y_grid[0, 0]))
        
        return grid_info