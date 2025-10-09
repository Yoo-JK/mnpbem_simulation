"""
MNPBEM Simulation Runner

This script generates MATLAB code based on structure and simulation configuration files.
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
        '--structure',
        type=str,
        required=True,
        help='Path to structure configuration file'
    )
    parser.add_argument(
        '--simulation',
        type=str,
        required=True,
        help='Path to simulation configuration file'
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
        raise ValueError(f"Config file must contain 'args' dictionary: {config_path}")
    
    return config_dict['args']


def merge_configs(structure_path, simulation_path):
    """Merge structure and simulation configs."""
    print(f"Loading structure config: {structure_path}")
    structure_args = load_config(structure_path)
    
    print(f"Loading simulation config: {simulation_path}")
    simulation_args = load_config(simulation_path)
    
    # Merge: simulation settings override structure if there's overlap
    merged = {**structure_args, **simulation_args}
    
    print(f"✓ Configurations loaded and merged successfully")
    
    return merged


def validate_config(args):
    """Validate configuration parameters."""
    required_keys = [
        'structure',
        'simulation_type',
        'excitation_type',
        'wavelength_range',
        'output_dir'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in args:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Required configuration keys missing: {', '.join(missing_keys)}")
    
    # Validate simulation type
    if args['simulation_type'] not in ['stat', 'ret']:
        raise ValueError(f"Invalid simulation_type: {args['simulation_type']}")
    
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
    
    # Load and merge configurations
    try:
        config = merge_configs(args_cli.structure, args_cli.simulation)
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
    print(f"Structure:        {config.get('structure_name', config.get('structure', 'N/A'))}")
    print(f"Structure type:   {config.get('structure', 'N/A')}")
    print(f"Simulation:       {config.get('simulation_name', 'N/A')}")
    print(f"Simulation type:  {config['simulation_type']}")
    print(f"Excitation:       {config['excitation_type']}")
    print(f"Wavelength range: {config['wavelength_range'][0]}-{config['wavelength_range'][1]} nm ({config['wavelength_range'][2]} points)")
    print(f"Output directory: {config['output_dir']}")
    print()
    print("Ready for MATLAB execution!")
    print("=" * 60)


if __name__ == '__main__':
    main()