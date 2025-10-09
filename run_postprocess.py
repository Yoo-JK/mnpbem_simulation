"""
MNPBEM Postprocessing Runner

This script loads simulation results and performs analysis and visualization.
"""

import argparse
import sys
import os
from pathlib import Path

# Add postprocess module to path
sys.path.insert(0, str(Path(__file__).parent))

from postprocess.postprocess import PostprocessManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Postprocess MNPBEM simulation results'
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
    structure_args = load_config(structure_path)
    simulation_args = load_config(simulation_path)
    
    # Merge
    merged = {**structure_args, **simulation_args}
    
    return merged


def main():
    """Main execution function."""
    # Parse arguments
    args_cli = parse_arguments()
    
    print("=" * 60)
    print("MNPBEM Postprocessing")
    print("=" * 60)
    print()
    
    # Load configuration
    print(f"Loading structure config: {args_cli.structure}")
    print(f"Loading simulation config: {args_cli.simulation}")
    try:
        config = merge_configs(args_cli.structure, args_cli.simulation)
        print(f"✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)
    
    # Create postprocess manager
    print("\nInitializing postprocess manager...")
    try:
        postproc_manager = PostprocessManager(config, verbose=args_cli.verbose)
        print("✓ Postprocess manager initialized")
    except Exception as e:
        print(f"✗ Error initializing postprocess manager: {e}")
        sys.exit(1)
    
    # Load results
    print("\nLoading simulation results...")
    try:
        postproc_manager.load_results()
        print("✓ Results loaded successfully")
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        sys.exit(1)
    
    # Analyze spectrum
    print("\nAnalyzing spectrum...")
    try:
        postproc_manager.analyze_spectrum()
        print("✓ Spectrum analysis complete")
    except Exception as e:
        print(f"✗ Error analyzing spectrum: {e}")
        sys.exit(1)
    
    # Save processed data
    print("\nSaving processed data...")
    try:
        postproc_manager.save_processed_data()
        print("✓ Processed data saved")
    except Exception as e:
        print(f"✗ Error saving processed data: {e}")
        sys.exit(1)
    
    # Generate plots
    if config.get('save_plots', True):
        print("\nGenerating plots...")
        try:
            postproc_manager.generate_plots()
            print("✓ Plots generated")
        except Exception as e:
            print(f"✗ Error generating plots: {e}")
            sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Postprocessing Complete")
    print("=" * 60)
    summary = postproc_manager.get_summary()
    print(f"Structure:            {config.get('structure_name', 'N/A')}")
    print(f"Simulation:           {config.get('simulation_name', 'N/A')}")
    print(f"Output directory:     {config['output_dir']}")
    print(f"Number of wavelengths: {summary['n_wavelengths']}")
    print(f"Number of polarizations: {summary['n_polarizations']}")
    
    if 'peak_wavelengths' in summary:
        print("\nPeak wavelengths (nm):")
        for i, wl in enumerate(summary['peak_wavelengths']):
            print(f"  Polarization {i+1}: {wl:.1f} nm")
    
    print("\nAll files saved in:", config['output_dir'])
    print("=" * 60)


if __name__ == '__main__':
    main()