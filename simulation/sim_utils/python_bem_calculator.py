"""
Python BEM Calculator using pyMNPBEM

This module provides a Python-native alternative to MATLAB-based MNPBEM calculations.
It maps the existing configuration format to pyMNPBEM objects and executes simulations.
"""

import sys
import os
import numpy as np
from pathlib import Path
import json


class PythonBEMCalculator:
    """
    Python-based BEM calculator using pyMNPBEM.

    This class translates configuration dictionaries into pyMNPBEM objects
    and executes BEM simulations entirely in Python, without MATLAB.
    """

    def __init__(self, config, verbose=False):
        """
        Initialize the Python BEM calculator.

        Args:
            config (dict): Configuration dictionary
            verbose (bool): Enable verbose output
        """
        self.config = config
        self.verbose = verbose

        # Add pyMNPBEM to path if specified
        self._setup_pymnpbem_path()

        # Import pyMNPBEM modules
        self._import_pymnpbem()

        # Initialize containers
        self.particle = None
        self.bem_solver = None
        self.excitation = None
        self.results = {}

        if self.verbose:
            print("PythonBEMCalculator initialized")

    def _setup_pymnpbem_path(self):
        """Add pyMNPBEM to Python path if specified in config."""
        pymnpbem_path = self.config.get('pymnpbem_path')

        if pymnpbem_path:
            pymnpbem_path = Path(pymnpbem_path).expanduser()
            if pymnpbem_path.exists():
                sys.path.insert(0, str(pymnpbem_path))
                if self.verbose:
                    print(f"Added pyMNPBEM path: {pymnpbem_path}")
            else:
                print(f"⚠ Warning: pyMNPBEM path not found: {pymnpbem_path}")

    def _import_pymnpbem(self):
        """Import pyMNPBEM modules."""
        try:
            # Import as module-level to make them accessible
            import mnpbem
            self.mnpbem = mnpbem

            # Import specific classes
            from mnpbem import (
                bemoptions, EpsConst, EpsTable, EpsDrude,
                ComParticle, bemsolver, planewave, dipole
            )
            from mnpbem.particles.shapes import trisphere, tricube, trirod, tritorus

            # Store in instance
            self.bemoptions = bemoptions
            self.EpsConst = EpsConst
            self.EpsTable = EpsTable
            self.EpsDrude = EpsDrude
            self.ComParticle = ComParticle
            self.bemsolver = bemsolver
            self.planewave = planewave
            self.dipole = dipole

            # Shapes
            self.trisphere = trisphere
            self.tricube = tricube
            self.trirod = trirod
            self.tritorus = tritorus

            if self.verbose:
                print("✓ pyMNPBEM modules imported successfully")
                print(f"  pyMNPBEM version: {mnpbem.__version__}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import pyMNPBEM. Please ensure it's installed.\n"
                f"Error: {e}\n"
                f"Install with: pip install -e /path/to/pyMNPBEM\n"
                f"Or set 'pymnpbem_path' in config."
            )

    def create_geometry(self):
        """
        Create particle geometry based on configuration.

        Returns:
            list: List of particle shapes
        """
        structure = self.config['structure']

        if self.verbose:
            print(f"\nCreating geometry: {structure}")

        # Single particles
        if structure == 'sphere':
            return self._create_sphere()
        elif structure == 'cube':
            return self._create_cube()
        elif structure == 'rod':
            return self._create_rod()
        elif structure == 'ellipsoid':
            return self._create_ellipsoid()

        # Core-shell structures
        elif structure == 'core_shell_sphere':
            return self._create_core_shell_sphere()
        elif structure == 'core_shell_cube':
            return self._create_core_shell_cube()

        # Dimers
        elif structure == 'dimer_sphere':
            return self._create_dimer_sphere()
        elif structure == 'dimer_cube':
            return self._create_dimer_cube()
        elif structure == 'advanced_dimer_cube':
            return self._create_advanced_dimer_cube()

        else:
            raise ValueError(f"Unsupported structure type: {structure}")

    def _create_sphere(self):
        """Create a single sphere."""
        diameter = self.config['diameter']
        mesh_density = self.config.get('mesh_density', 144)

        sphere = self.trisphere(mesh_density, diameter)

        if self.verbose:
            print(f"  Sphere: diameter={diameter}nm, mesh={mesh_density}")

        return [sphere]

    def _create_cube(self):
        """Create a single cube."""
        size = self.config['size']
        mesh_density = self.config.get('mesh_density', 12)
        rounding = self.config.get('rounding', 0.25)

        cube = self.tricube(mesh_density, size, rounding=rounding)

        if self.verbose:
            print(f"  Cube: size={size}nm, rounding={rounding}, mesh={mesh_density}")

        return [cube]

    def _create_rod(self):
        """Create a cylindrical rod."""
        diameter = self.config['diameter']
        height = self.config['height']

        # Handle different mesh specifications
        if 'rod_mesh' in self.config:
            mesh = self.config['rod_mesh']
            rod = self.trirod(diameter, height, mesh)
            if self.verbose:
                print(f"  Rod: diameter={diameter}nm, height={height}nm, mesh={mesh}")
        else:
            mesh_density = self.config.get('mesh_density', 144)
            rod = self.trirod(diameter, height, mesh_density)
            if self.verbose:
                print(f"  Rod: diameter={diameter}nm, height={height}nm, mesh={mesh_density}")

        return [rod]

    def _create_ellipsoid(self):
        """Create an ellipsoid (using scaled sphere)."""
        axes = self.config['axes']  # [rx, ry, rz]
        mesh_density = self.config.get('mesh_density', 144)

        # Create sphere and scale
        sphere = self.trisphere(mesh_density, 1.0)

        # Scale to ellipsoid
        sphere.verts[:, 0] *= axes[0]
        sphere.verts[:, 1] *= axes[1]
        sphere.verts[:, 2] *= axes[2]

        if self.verbose:
            print(f"  Ellipsoid: axes={axes}nm, mesh={mesh_density}")

        return [sphere]

    def _create_core_shell_sphere(self):
        """Create a core-shell sphere."""
        core_diameter = self.config['core_diameter']
        shell_thickness = self.config['shell_thickness']
        mesh_density = self.config.get('mesh_density', 144)

        # Inner sphere (core)
        inner = self.trisphere(mesh_density, core_diameter)

        # Outer sphere (shell)
        outer_diameter = core_diameter + 2 * shell_thickness
        outer = self.trisphere(mesh_density, outer_diameter)

        if self.verbose:
            print(f"  Core-shell sphere: core={core_diameter}nm, shell={shell_thickness}nm")

        return [outer, inner]

    def _create_core_shell_cube(self):
        """Create a core-shell cube."""
        core_size = self.config['core_size']
        shell_thickness = self.config['shell_thickness']
        mesh_density = self.config.get('mesh_density', 12)
        rounding = self.config.get('rounding', 0.25)

        # Inner cube (core)
        inner = self.tricube(mesh_density, core_size, rounding=rounding)

        # Outer cube (shell)
        outer_size = core_size + 2 * shell_thickness
        outer = self.tricube(mesh_density, outer_size, rounding=rounding)

        if self.verbose:
            print(f"  Core-shell cube: core={core_size}nm, shell={shell_thickness}nm")

        return [outer, inner]

    def _create_dimer_sphere(self):
        """Create a dimer of two spheres."""
        diameter = self.config['diameter']
        gap = self.config['gap']
        mesh_density = self.config.get('mesh_density', 144)

        # Create two spheres
        sphere1 = self.trisphere(mesh_density, diameter)
        sphere2 = self.trisphere(mesh_density, diameter)

        # Position them with gap
        shift = diameter / 2 + gap + diameter / 2
        sphere1.verts[:, 0] -= shift / 2
        sphere2.verts[:, 0] += shift / 2

        if self.verbose:
            print(f"  Dimer sphere: diameter={diameter}nm, gap={gap}nm")

        return [sphere1, sphere2]

    def _create_dimer_cube(self):
        """Create a dimer of two cubes."""
        size = self.config['size']
        gap = self.config['gap']
        mesh_density = self.config.get('mesh_density', 12)
        rounding = self.config.get('rounding', 0.25)

        # Create two cubes
        cube1 = self.tricube(mesh_density, size, rounding=rounding)
        cube2 = self.tricube(mesh_density, size, rounding=rounding)

        # Position them with gap
        shift = size / 2 + gap + size / 2
        cube1.verts[:, 0] -= shift / 2
        cube2.verts[:, 0] += shift / 2

        if self.verbose:
            print(f"  Dimer cube: size={size}nm, gap={gap}nm")

        return [cube1, cube2]

    def _create_advanced_dimer_cube(self):
        """Create advanced dimer cube with multiple shells and transformations."""
        # This is a placeholder - would need full implementation
        # For now, fall back to simple dimer
        if self.verbose:
            print("  ⚠ Advanced dimer cube: using simplified version")

        return self._create_dimer_cube()

    def create_materials(self):
        """
        Create material epsilon objects.

        Returns:
            list: List of epsilon objects [eps_medium, eps_mat1, eps_mat2, ...]
        """
        if self.verbose:
            print("\nCreating materials...")

        epstab = []

        # Medium (surrounding environment)
        medium = self.config.get('medium', 'vacuum')
        eps_medium = self._create_single_material(medium, is_medium=True)
        epstab.append(eps_medium)

        if self.verbose:
            print(f"  Medium: {medium}")

        # Particle materials
        materials = self.config.get('materials', [])
        for mat in materials:
            eps_mat = self._create_single_material(mat)
            epstab.append(eps_mat)
            if self.verbose:
                print(f"  Material: {mat}")

        return epstab

    def _create_single_material(self, material, is_medium=False):
        """
        Create a single epsilon object for a material.

        Args:
            material: Material name (str) or dict with custom properties
            is_medium: Whether this is the surrounding medium

        Returns:
            Epsilon object (EpsConst, EpsTable, or EpsDrude)
        """
        # Handle dict format (custom material)
        if isinstance(material, dict):
            if material.get('type') == 'constant':
                return self.EpsConst(material['epsilon'])
            elif material.get('type') == 'drude':
                return self.EpsDrude(
                    material.get('eps_inf', 1.0),
                    material.get('omega_p'),
                    material.get('gamma')
                )
            else:
                raise ValueError(f"Unknown material type: {material.get('type')}")

        # Handle string format (named material)
        material_lower = material.lower()

        # Constant materials
        if material_lower in ['vacuum', 'air']:
            return self.EpsConst(1.0)
        elif material_lower == 'water':
            return self.EpsConst(1.77)  # n=1.33
        elif material_lower == 'glass':
            return self.EpsConst(2.25)  # n=1.5

        # Table-based materials (from data files)
        else:
            # Try to find material data file
            material_file = self._find_material_file(material_lower)

            if material_file:
                return self.EpsTable(material_file)
            else:
                raise ValueError(
                    f"Material '{material}' not found. "
                    f"Please provide a data file or use constant epsilon."
                )

    def _find_material_file(self, material_name):
        """
        Find material data file.

        Args:
            material_name: Material name (e.g., 'gold', 'silver')

        Returns:
            str: Path to material file, or None if not found
        """
        # Check custom paths first
        custom_paths = self.config.get('refractive_index_paths', {})
        if material_name in custom_paths:
            return custom_paths[material_name]

        # Check pyMNPBEM data directory
        pymnpbem_path = self.config.get('pymnpbem_path')
        if pymnpbem_path:
            data_dir = Path(pymnpbem_path) / 'data'
            material_file = data_dir / f'{material_name}.dat'
            if material_file.exists():
                return str(material_file)

        # Try relative to pyMNPBEM package
        try:
            import mnpbem
            package_dir = Path(mnpbem.__file__).parent
            data_dir = package_dir / 'data'
            material_file = data_dir / f'{material_name}.dat'
            if material_file.exists():
                return str(material_file)
        except:
            pass

        return None

    def create_comparticle(self, shapes, epstab):
        """
        Create ComParticle from shapes and materials.

        Args:
            shapes: List of particle shapes
            epstab: List of epsilon objects

        Returns:
            ComParticle object
        """
        if self.verbose:
            print("\nCreating ComParticle...")

        # Determine inout array based on structure
        inout = self._determine_inout(shapes, epstab)

        # Check if closed
        closed = self._is_closed_structure()

        # Create particle
        particle = self.ComParticle(epstab, shapes, inout, closed=closed)

        if self.verbose:
            print(f"  Number of faces: {particle.n_faces}")
            print(f"  Closed: {closed}")

        return particle

    def _determine_inout(self, shapes, epstab):
        """
        Determine inout array for ComParticle.

        Args:
            shapes: List of shapes
            epstab: List of epsilon objects

        Returns:
            list: inout array [[inside_idx, outside_idx], ...]
        """
        structure = self.config['structure']
        n_materials = len(self.config.get('materials', []))

        # Single particle: inside=material[0], outside=medium
        if structure in ['sphere', 'cube', 'rod', 'ellipsoid', 'triangle']:
            return [[2, 1]]  # inside=eps[1]=material[0], outside=eps[0]=medium

        # Core-shell: outer=[mat[0], medium], inner=[mat[1], mat[0]]
        elif structure in ['core_shell_sphere', 'core_shell_cube']:
            return [
                [2, 1],  # outer: inside=mat[0], outside=medium
                [3, 2]   # inner: inside=mat[1], outside=mat[0]
            ]

        # Dimer: two particles, both same material
        elif structure in ['dimer_sphere', 'dimer_cube']:
            return [
                [2, 1],  # particle 1
                [2, 1]   # particle 2
            ]

        # Advanced dimer with multiple shells
        elif structure == 'advanced_dimer_cube':
            # For now, simplified
            return [[2, 1], [2, 1]]

        else:
            # Default: single material
            return [[2, 1]]

    def _is_closed_structure(self):
        """Check if structure is closed (1) or open (0)."""
        structure = self.config['structure']

        # Most structures are closed
        closed_structures = [
            'sphere', 'cube', 'rod', 'ellipsoid',
            'core_shell_sphere', 'core_shell_cube',
            'dimer_sphere', 'dimer_cube', 'advanced_dimer_cube'
        ]

        return 1 if structure in closed_structures else 0

    def create_bem_options(self):
        """
        Create BEM options.

        Returns:
            BEMOptions object
        """
        sim_type = self.config.get('simulation_type', 'stat')
        interp = self.config.get('interp', 'curv')
        waitbar = self.config.get('waitbar', 0)

        op = self.bemoptions(sim=sim_type, interp=interp, waitbar=waitbar)

        # Add refine if specified
        if 'refine' in self.config:
            op.refine = self.config['refine']

        if self.verbose:
            print(f"\nBEM options: sim={sim_type}, interp={interp}")

        return op

    def create_bem_solver(self, particle, options):
        """
        Create BEM solver.

        Args:
            particle: ComParticle object
            options: BEMOptions object

        Returns:
            BEM solver object
        """
        if self.verbose:
            print("\nCreating BEM solver...")

        bem = self.bemsolver(particle, options)

        if self.verbose:
            print(f"  Solver type: {type(bem).__name__}")

        return bem

    def create_excitation(self, options):
        """
        Create excitation object.

        Args:
            options: BEMOptions object

        Returns:
            Excitation object (planewave or dipole)
        """
        exc_type = self.config.get('excitation_type', 'planewave')

        if self.verbose:
            print(f"\nCreating excitation: {exc_type}")

        if exc_type == 'planewave':
            pols = self.config.get('polarizations', [[1, 0, 0]])
            dirs = self.config.get('propagation_dirs', [[0, 0, 1]])

            return self.planewave(pols, dirs, options)

        elif exc_type == 'dipole':
            pos = self.config.get('dipole_position', [0, 0, 10])
            moment = self.config.get('dipole_moment', [0, 0, 1])

            return self.dipole(pos, moment, options)

        else:
            raise ValueError(f"Unsupported excitation type: {exc_type}")

    def run_simulation(self):
        """
        Run the complete BEM simulation.

        Returns:
            dict: Results dictionary with spectra and field data
        """
        print("\n" + "="*60)
        print("Starting Python BEM Simulation")
        print("="*60)

        # Create geometry
        shapes = self.create_geometry()

        # Create materials
        epstab = self.create_materials()

        # Create particle
        self.particle = self.create_comparticle(shapes, epstab)

        # Create BEM options
        options = self.create_bem_options()

        # Create BEM solver
        self.bem_solver = self.create_bem_solver(self.particle, options)

        # Create excitation
        self.excitation = self.create_excitation(options)

        # Get wavelength range
        wl_range = self.config['wavelength_range']
        wavelengths = np.linspace(wl_range[0], wl_range[1], int(wl_range[2]))

        print(f"\nComputing spectra for {len(wavelengths)} wavelengths...")
        print(f"Range: {wl_range[0]}-{wl_range[1]} nm")

        # Allocate result arrays
        n_pol = len(self.config.get('polarizations', [[1, 0, 0]]))
        sca = np.zeros((len(wavelengths), n_pol))
        abs_cross = np.zeros((len(wavelengths), n_pol))
        ext = np.zeros((len(wavelengths), n_pol))

        # Wavelength loop
        for i, wl in enumerate(wavelengths):
            # Get excitation potential
            exc_pot = self.excitation(self.particle, wl)

            # Solve BEM equations
            sig = self.bem_solver.solve(exc_pot)

            # Compute cross sections
            if hasattr(self.excitation, 'sca'):
                sca[i, :] = self.excitation.sca(sig)
            if hasattr(self.excitation, 'ext'):
                ext[i, :] = self.excitation.ext(sig)

            # Absorption = Extinction - Scattering
            abs_cross[i, :] = ext[i, :] - sca[i, :]

            if (i + 1) % 20 == 0 or (i + 1) == len(wavelengths):
                print(f"  Processed {i + 1}/{len(wavelengths)} wavelengths")

        print("✓ BEM calculation complete!")

        # Store results
        self.results = {
            'wavelengths': wavelengths,
            'scattering': sca,
            'absorption': abs_cross,
            'extinction': ext,
        }

        return self.results

    def save_results(self, output_dir):
        """
        Save results to files (compatible with MATLAB format).

        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to: {output_dir}")

        # Save spectra to text file (MATLAB compatible)
        spec_file = output_dir / 'cross_sections.txt'
        wavelengths = self.results['wavelengths']
        sca = self.results['scattering']
        abs_cross = self.results['absorption']
        ext = self.results['extinction']

        # Write header
        with open(spec_file, 'w') as f:
            f.write("# Wavelength(nm) Scattering(nm^2) Absorption(nm^2) Extinction(nm^2)\n")
            for i in range(len(wavelengths)):
                f.write(f"{wavelengths[i]:.6f} {sca[i, 0]:.6e} "
                       f"{abs_cross[i, 0]:.6e} {ext[i, 0]:.6e}\n")

        print(f"  ✓ Saved: {spec_file}")

        # Save as JSON
        json_file = output_dir / 'results.json'
        json_data = {
            'wavelengths': wavelengths.tolist(),
            'scattering': sca.tolist(),
            'absorption': abs_cross.tolist(),
            'extinction': ext.tolist(),
            'config': {
                'structure': self.config['structure'],
                'simulation_type': self.config['simulation_type'],
                'excitation_type': self.config['excitation_type'],
            }
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"  ✓ Saved: {json_file}")

        # Try to save as numpy
        try:
            npz_file = output_dir / 'results.npz'
            np.savez(
                npz_file,
                wavelengths=wavelengths,
                scattering=sca,
                absorption=abs_cross,
                extinction=ext
            )
            print(f"  ✓ Saved: {npz_file}")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Could not save NPZ: {e}")

        print("✓ Results saved successfully")
