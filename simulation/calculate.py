"""
Simulation Manager Class

This class orchestrates the MATLAB code generation process.
"""

import os
from pathlib import Path
from .sim_utils.geometry_generator import GeometryGenerator
from .sim_utils.material_manager import MaterialManager
from .sim_utils.matlab_code_generator import MatlabCodeGenerator


class SimulationManager:
    """Manages the entire simulation generation process."""
    
    def __init__(self, config, verbose=False):
        """
        Initialize simulation manager.
        
        Args:
            config (dict): Configuration dictionary
            verbose (bool): Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        self.matlab_code = None
        
        # Initialize sub-managers
        self.geometry_gen = GeometryGenerator(config, verbose)
        self.material_mgr = MaterialManager(config, verbose)
        self.matlab_gen = MatlabCodeGenerator(config, verbose)
        
        if verbose:
            print("SimulationManager initialized")
    
    def generate_matlab_code(self):
        """Generate complete MATLAB simulation code."""
        if self.verbose:
            print("\n--- Generating MATLAB Code ---")
        
        # Generate geometry code
        if self.verbose:
            print("Generating geometry code...")
        geometry_code = self.geometry_gen.generate()
        
        # Generate material code
        if self.verbose:
            print("Generating material code...")
        material_code = self.material_mgr.generate()
        
        # Generate complete MATLAB script
        if self.verbose:
            print("Assembling complete MATLAB script...")
        self.matlab_code = self.matlab_gen.generate_complete_script(
            geometry_code=geometry_code,
            material_code=material_code
        )
        
        if self.verbose:
            print("MATLAB code generation complete")
        
        return self.matlab_code
    
    def save_matlab_script(self, output_path=None):
        """
        Save MATLAB script to file.
        
        Args:
            output_path (str): Output file path. If None, uses default.
        
        Returns:
            str: Path to saved file
        """
        if self.matlab_code is None:
            raise RuntimeError("MATLAB code not generated yet. Call generate_matlab_code() first.")
        
        # Determine output path
        if output_path is None:
            output_dir = Path(__file__).parent
            output_path = output_dir / 'simulation_script.m'
        else:
            output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(output_path, 'w') as f:
            f.write(self.matlab_code)
        
        if self.verbose:
            print(f"MATLAB script saved to: {output_path}")
        
        return str(output_path)
    
    def get_summary(self):
        """Get summary of simulation configuration."""
        summary = {
            'structure': self.config['structure'],
            'simulation_type': self.config['simulation_type'],
            'excitation': self.config['excitation_type'],
            'wavelength_range': self.config['wavelength_range'],
            'materials': self.config['materials']
        }
        return summary