"""
MNPBEM Simulation Runner

This script generates MATLAB code based on the configuration file
and prepares everything for MATLAB execution.
"""

import argparse
import sys
import os
from pathlib import Path

# Add simulation module to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.calculate import SimulationManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate and prepare MNPBEM simulation'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config file as module
    config_dict = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_dict)
    
    if 'args' not in config_dict:
        raise ValueError("Config file must contain 'args' dictionary")
    
    return config_dict['args']


def validate_config(args):
    """Validate configuration parameters."""
    required_keys = [
        'structure',
        'simulation_type',
        'materials',
        'excitation_type',
        'wavelength_range',
        'output_dir'
    ]
    
    for key in required_keys:
        if key not in args:
            raise ValueError(f"Required configuration key missing: {key}")
    
    # Validate simulation type
    if args['simulation_type'] not in ['stat', 'ret']:
        raise ValueError(f"Invalid simulation_type: {args['simulation_type']}")
    
    # Validate structure type
    valid_structures = [
        'sphere', 'cube', 'rod', 'ellipsoid', 'triangle',
        'dimer_sphere', 'dimer_cube',
        'core_shell_sphere', 'core_shell_cube',
        'dimer_core_shell_cube'
    ]
    if args['structure'] not in valid_structures:
        raise ValueError(f"Invalid structure: {args['structure']}")
    
    # Validate excitation type
    if args['excitation_type'] not in ['planewave', 'dipole', 'eels']:
        raise ValueError(f"Invalid excitation_type: {args['excitation_type']}")
    
    print("✓ Configuration validated successfully")


def main():
    """Main execution function."""
    # Parse arguments
    args_cli = parse_arguments()
    
    print("=" * 60)
    print("MNPBEM Simulation Generator")
    print("=" * 60)
    print()
    
    # Load configuration
    print(f"Loading configuration from: {args_cli.config}")
    try:
        config = load_config(args_cli.config)
        print(f"✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    print("\nValidating configuration...")
    try:
        validate_config(config)
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Create simulation manager
    print("\nInitializing simulation manager...")
    try:
        sim_manager = SimulationManager(config, verbose=args_cli.verbose)
        print("✓ Simulation manager initialized")
    except Exception as e:
        print(f"✗ Error initializing simulation manager: {e}")
        sys.exit(1)
    
    # Generate MATLAB code
    print("\nGenerating MATLAB simulation code...")
    try:
        sim_manager.generate_matlab_code()
        print("✓ MATLAB code generated successfully")
    except Exception as e:
        print(f"✗ Error generating MATLAB code: {e}")
        sys.exit(1)
    
    # Save MATLAB script
    print("\nSaving MATLAB script...")
    try:
        output_path = sim_manager.save_matlab_script()
        print(f"✓ MATLAB script saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving MATLAB script: {e}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Preparation Complete")
    print("=" * 60)
    print(f"Structure:        {config['structure']}")
    print(f"Simulation type:  {config['simulation_type']}")
    print(f"Excitation:       {config['excitation_type']}")
    print(f"Wavelength range: {config['wavelength_range'][0]}-{config['wavelength_range'][1]} nm")
    print(f"Output directory: {config['output_dir']}")
    print()
    print("Ready for MATLAB execution!")
    print("=" * 60)


if __name__ == '__main__':
    main()