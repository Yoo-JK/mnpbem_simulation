"""
CTP Test Example: Capacitive vs Conductive Junction

이 예제는 동일한 금 나노입자 dimer에서:
1. Capacitive coupling (gap > 0)
2. Conductive junction (gap = 0, CTP mode)

두 가지 모드를 비교하여 CTP 효과를 관측합니다.
"""

def get_structure_config(mode='capacitive'):
    """
    Get structure configuration for capacitive or conductive mode.

    Args:
        mode (str): 'capacitive' or 'conductive'

    Returns:
        dict: Structure configuration
    """
    if mode == 'capacitive':
        gap = 1.0  # 1 nm separation → capacitive coupling
        name_suffix = 'capacitive'
    elif mode == 'conductive':
        gap = 0.0  # Contact → conductive junction (CTP mode)
        name_suffix = 'conductive'
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'capacitive' or 'conductive'")

    args = {}

    # Structure identification
    args['structure_name'] = f'au_dimer_ctp_test_{name_suffix}'

    # Geometry
    args['structure'] = 'dimer_sphere'
    args['diameter'] = 50  # nm
    args['gap'] = gap  # ⭐ This controls capacitive vs conductive

    # Materials
    args['materials'] = ['gold']
    args['medium'] = 'water'

    # Mesh
    args['mesh_density'] = 144  # High density for contact region

    return args


def get_simulation_config(structure_name):
    """
    Get simulation configuration.

    Args:
        structure_name (str): Name from structure config

    Returns:
        dict: Simulation configuration
    """
    args = {}

    # Simulation identification
    args['simulation_name'] = structure_name
    args['simulation_type'] = 'stat'  # Quasi-static for small particles

    # Wavelength range (CTP causes red-shift)
    args['wavelengths'] = [400, 1000, 120]  # [start, end, npoints]

    # Excitation
    args['excitation_type'] = 'planewave'
    args['polarizations'] = [[1, 0, 0]]  # Along dimer axis (x)
    args['propagation_dirs'] = [[0, 0, 1]]  # z direction

    # BEM options
    args['npol'] = 15  # Integration points
    args['calculate_cross_sections'] = True

    # Field calculation (optional)
    args['calculate_field'] = False  # Set to True if you want field maps

    # Output
    args['save_matlab_output'] = True
    args['generate_plots'] = True

    return args


def run_comparison():
    """
    Run both capacitive and conductive simulations and compare.

    Usage:
        python -c "from examples.ctp_test_capacitive_vs_conductive import run_comparison; run_comparison()"
    """
    import sys
    import os

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from simulation.calculate import SimulationManager

    print("=" * 80)
    print("CTP Test: Capacitive vs Conductive Junction")
    print("=" * 80)

    for mode in ['capacitive', 'conductive']:
        print(f"\n{'='*80}")
        print(f"Running: {mode.upper()} mode")
        print(f"{'='*80}\n")

        # Get configs
        structure_config = get_structure_config(mode)
        simulation_config = get_simulation_config(structure_config['structure_name'])

        # Print info
        gap = structure_config['gap']
        print(f"Gap: {gap} nm")
        if gap <= 0:
            print("→ Conductive junction (CTP mode) - AUTO ACTIVATED")
        else:
            print("→ Capacitive coupling (standard mode)")

        # Run simulation
        manager = SimulationManager(structure_config, simulation_config, verbose=True)
        success = manager.run()

        if success:
            print(f"\n✓ {mode.upper()} simulation completed successfully")
            print(f"  Output: output/{structure_config['structure_name']}/")
        else:
            print(f"\n✗ {mode.upper()} simulation failed")

    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)
    print("\nTo compare spectra:")
    print("  1. Check extinction spectra in output/*/extinction_spectrum.png")
    print("  2. Look for red-shift in conductive vs capacitive")
    print("  3. Conductive junction should show BDP (Bonding Dipole Plasmon) mode")
    print("\nExpected results:")
    print("  - Capacitive: ~530 nm peak (standard bonding mode)")
    print("  - Conductive: Red-shifted peak (BDP mode with charge transfer)")


if __name__ == '__main__':
    print(__doc__)
    print("\nThis module provides example configurations for CTP testing.")
    print("\nTo run the comparison test:")
    print('  python -c "from examples.ctp_test_capacitive_vs_conductive import run_comparison; run_comparison()"')
    print("\nOr use the configurations manually:")
    print("  from examples.ctp_test_capacitive_vs_conductive import get_structure_config, get_simulation_config")
    print('  struct_cfg = get_structure_config("conductive")')
    print('  sim_cfg = get_simulation_config(struct_cfg["structure_name"])')
