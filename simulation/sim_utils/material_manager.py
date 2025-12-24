"""
Material Manager

Manages material definitions and generates corresponding MATLAB code.
Supports:
  - Medium, materials, and substrate as separate configs
  - Enhanced 'table' type with automatic interpolation
  - Custom refractive_index_paths from config
  - Nonlocal quantum corrections for metals
"""

import numpy as np
from pathlib import Path
from .refractive_index_loader import RefractiveIndexLoader
from .nonlocal_generator import NonlocalGenerator


class MaterialManager:
    """Manages material definitions and dielectric functions."""

    # List of recognized metal names
    METALS = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al', 'copper', 'cu']

    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        self.nonlocal_gen = NonlocalGenerator(config, verbose)
        self.complete_materials = self._build_complete_material_list()
        self.table_materials_data = {}

    def _is_metal(self, material):
        """Check if a material is a metal."""
        if isinstance(material, str):
            mat_lower = material.lower()
            return any(metal in mat_lower for metal in self.METALS)
        return False

    def _find_outermost_metal(self, materials):
        """
        Find the outermost metal in the materials list.

        In core-shell structures, materials are ordered from core to shell,
        so the outermost is the LAST item in the list.

        Returns:
            tuple: (index, material_name) or (None, None) if no metal found
        """
        # Check from the outermost (last) to innermost (first)
        for i in range(len(materials) - 1, -1, -1):
            if self._is_metal(materials[i]):
                return i, materials[i]
        return None, None
    
    def _build_complete_material_list(self):
        """
        Build complete ordered material list from medium, materials, substrate.
        
        Returns:
            list: Complete material list in correct order for MNPBEM
        """
        materials_list = []
        
        # 1. Medium (always first)
        medium = self.config.get('medium', 'air')
        materials_list.append(medium)
        
        # 2. Particle materials
        particle_materials = self.config.get('materials', [])
        if not isinstance(particle_materials, list):
            particle_materials = [particle_materials]
        materials_list.extend(particle_materials)
        
        # 3. Substrate (if used, always last)
        if self.config.get('use_substrate', False):
            substrate = self.config.get('substrate', {})
            substrate_material = substrate.get('material', 'glass')
            materials_list.append(substrate_material)
        
        if self.verbose:
            print(f"Complete material list: {materials_list}")
        
        return materials_list
    
    def generate(self):
        """Generate material-related MATLAB code."""
        use_substrate = self.config.get('use_substrate', False)
        use_nonlocal = self.nonlocal_gen.is_needed()
        
        if use_substrate:
            return self._generate_substrate_materials()
        
        if use_nonlocal:
            # Use nonlocal-aware generation
            epstab_code = self._generate_nonlocal_epstab()
            inout_code = self._generate_inout_nonlocal()
            closed_code = self._generate_closed()
            
            code = f"""
%% Materials and Dielectric Functions (Nonlocal Mode)
{epstab_code}

%% Material Mapping
{inout_code}

%% Closed Surfaces
{closed_code}
"""
        else:
            # Standard generation
            epstab_code = self._generate_epstab()
            inout_code = self._generate_inout()
            closed_code = self._generate_closed()
            
            code = f"""
%% Materials and Dielectric Functions
{epstab_code}

%% Material Mapping
{inout_code}

%% Closed Surfaces
{closed_code}
"""
        
        return code
    
    def _generate_nonlocal_epstab(self):
        """Generate dielectric function table with nonlocal corrections.

        IMPORTANT: Nonlocal corrections are applied ONLY to the outermost metal.
        Examples:
          - Au nanocube dimer: Au gets nonlocal
          - Au@Ag nanocube dimer: Only Ag (shell) gets nonlocal
          - Au@Ag@AgCl nanocube dimer: AgCl is not metal, no nonlocal applied
        """
        if not self.nonlocal_gen.is_needed():
            return self._generate_epstab()

        materials_list = self.complete_materials
        particle_materials = self.config.get('materials', [])

        # Find the outermost metal in particle materials
        outermost_idx, outermost_metal = self._find_outermost_metal(particle_materials)

        if self.verbose:
            print("\n=== Generating Nonlocal Materials ===")
            if outermost_metal:
                print(f"  Outermost metal: {outermost_metal} (index {outermost_idx} in particle materials)")
            else:
                print("  No metal found in outermost layer - nonlocal will NOT be applied")

        # If no outermost metal, fall back to standard generation
        if outermost_metal is None:
            if self.verbose:
                print("  → Falling back to standard (non-nonlocal) material generation")
            return self._generate_epstab()

        epstab_entries = []
        material_descriptions = []

        # complete_materials = [medium, particle_mat1, particle_mat2, ..., (substrate)]
        # outermost_idx is relative to particle_materials, so in complete_materials it's outermost_idx + 1
        outermost_complete_idx = outermost_idx + 1  # +1 because medium is at index 0

        for i, mat in enumerate(materials_list):
            mat_idx_display = i + 1  # 1-based for display

            if i == outermost_complete_idx:
                # This is the outermost metal - apply nonlocal
                material_descriptions.append(f"% Material {mat_idx_display}: {mat} (Drude + Nonlocal) [OUTERMOST]")
                epstab_entries.append(f"eps_{mat}_drude")
                epstab_entries.append(f"eps_{mat}_nonlocal")

                if self.verbose:
                    print(f"  Material {mat_idx_display}: {mat} → Drude + Nonlocal (outermost)")
            else:
                # All other materials (including inner metals) - standard treatment
                material_descriptions.append(f"% Material {mat_idx_display}: {mat}")
                var_name = f"eps_mat{mat_idx_display}"
                epstab_entries.append(var_name)

                if self.verbose:
                    print(f"  Material {mat_idx_display}: {mat} → Standard")

        materials_code = "\n".join(material_descriptions)
        epstab_code = "epstab = { " + ", ".join(epstab_entries) + " };"

        full_code = f"""
%% Dielectric Functions with Nonlocal Corrections
%% NOTE: Nonlocal applied ONLY to outermost metal: {outermost_metal}
{materials_code}

% Medium
eps_mat1 = {self._generate_single_material(materials_list[0], 1)};

% Generate artificial nonlocal dielectric function for outermost metal only
"""

        # Add artificial epsilon ONLY for the outermost metal
        full_code += self.nonlocal_gen.generate_artificial_epsilon(outermost_metal)
        full_code += "\n"

        # Add standard material definitions for all other materials
        for i, mat in enumerate(materials_list):
            mat_idx_display = i + 1

            # Skip medium (already defined) and outermost metal (has drude+nonlocal)
            if i == 0:  # medium
                continue
            if i == outermost_complete_idx:  # outermost metal
                continue

            mat_def = self._generate_single_material(mat, mat_idx_display)
            full_code += f"eps_mat{mat_idx_display} = {mat_def};  % {mat}\n"

        full_code += f"\n{epstab_code}\n"

        return full_code
    
    def _generate_inout_nonlocal(self):
        """Generate inout matrix for nonlocal structure.

        IMPORTANT: Nonlocal is applied ONLY to the outermost metal.
        - For core-shell structures: outermost = last in materials list
        - Only that layer gets Drude+Nonlocal treatment
        - Inner metals are treated as standard materials
        """
        structure = self.config.get('structure', '')
        materials = self.config.get('materials', [])

        # Find outermost metal
        outermost_idx, outermost_metal = self._find_outermost_metal(materials)

        if self.verbose:
            print(f"\n=== Generating Inout (Nonlocal Mode) ===")
            if outermost_metal:
                print(f"  Outermost metal: {outermost_metal} at index {outermost_idx}")
            else:
                print(f"  No outermost metal - using standard inout")

        # If no outermost metal, use standard inout
        if outermost_metal is None:
            return self._generate_inout()

        if 'dimer' in structure:
            n_particles = 2
        else:
            n_particles = 1

        # Build epstab index mapping
        # epstab = [medium, mat1, mat2, ..., outermost_drude, outermost_nonlocal, ...]
        # Only the outermost metal gets drude+nonlocal (2 entries), others get 1 entry
        mat_indices = {}
        epstab_idx = 2  # Start from 2 (1 is medium)

        for i, mat in enumerate(materials):
            if i == outermost_idx:
                # Outermost metal: drude + nonlocal (2 entries)
                mat_indices[i] = {
                    'drude': epstab_idx,
                    'nonlocal': epstab_idx + 1,
                    'is_outermost_metal': True
                }
                epstab_idx += 2
            else:
                # Standard material (1 entry)
                mat_indices[i] = {
                    'standard': epstab_idx,
                    'is_outermost_metal': False
                }
                epstab_idx += 1

        # Generate inout rows
        # For core-shell: layers are ordered [core, shell1, shell2, ...]
        # Boundaries go from innermost to outermost
        inout_rows = []
        n_layers = len(materials)

        for particle_idx in range(n_particles):
            for layer_idx in range(n_layers):
                is_outermost_layer = (layer_idx == n_layers - 1)
                is_outermost_metal = mat_indices[layer_idx].get('is_outermost_metal', False)

                if is_outermost_metal:
                    # This is the outermost metal - has two boundaries (outer + inner)
                    nonlocal_idx = mat_indices[layer_idx]['nonlocal']
                    drude_idx = mat_indices[layer_idx]['drude']

                    # Outer boundary: nonlocal inside, medium(1) outside
                    inout_rows.append(f"{nonlocal_idx}, 1")

                    # Inner boundary: drude inside, nonlocal outside
                    inout_rows.append(f"{drude_idx}, {nonlocal_idx}")
                else:
                    # Standard material - single boundary
                    std_idx = mat_indices[layer_idx]['standard']

                    if layer_idx == n_layers - 1:
                        # Outermost layer (but not metal): inside=material, outside=medium
                        inout_rows.append(f"{std_idx}, 1")
                    elif layer_idx == 0:
                        # Core: inside=core, outside=next_layer
                        next_layer_idx = layer_idx + 1
                        if mat_indices[next_layer_idx].get('is_outermost_metal'):
                            # Next layer is outermost metal → use its drude index
                            outside_idx = mat_indices[next_layer_idx]['drude']
                        else:
                            outside_idx = mat_indices[next_layer_idx]['standard']
                        inout_rows.append(f"{std_idx}, {outside_idx}")
                    else:
                        # Middle layer: inside=this, outside=next
                        next_layer_idx = layer_idx + 1
                        if mat_indices[next_layer_idx].get('is_outermost_metal'):
                            outside_idx = mat_indices[next_layer_idx]['drude']
                        else:
                            outside_idx = mat_indices[next_layer_idx]['standard']
                        inout_rows.append(f"{std_idx}, {outside_idx}")

        inout_str = "; ...\n         ".join(inout_rows)

        code = f"""
%% Material Mapping (with Nonlocal on outermost metal: {outermost_metal})
% inout(i, :) = [material_inside, material_outside] for boundary i
% Nonlocal correction applied ONLY to outermost metal layer
inout = [ {inout_str} ];

fprintf('  Total boundaries: %d\\n', size(inout, 1));
"""
        return code
    
    def _generate_substrate_materials(self):
        """Generate materials code for substrate configuration."""
        substrate = self.config.get('substrate', {})
        z_interface = substrate.get('position', 0)
        
        epstab_code = self._generate_epstab()
        substrate_idx = len(self.complete_materials)
        
        code = f"""
%% Materials and Dielectric Functions (with Substrate)
{epstab_code}

%% Layer Structure Setup
fprintf('Setting up layer structure...\\n');
z_interface = {z_interface};

if exist('layerstructure', 'class')
    op_layer = layerstructure.options;
    layer = layerstructure(epstab, [1, {substrate_idx}], z_interface, op_layer);
    op.layer = layer;
    fprintf('  Layer structure created at z=%.2f nm\\n', z_interface);
else
    warning('layerstructure class not found. Running without substrate.');
end

%% Material Mapping
{self._generate_inout()}

%% Closed Surfaces
{self._generate_closed()}
"""
        return code
    
    def _generate_epstab(self):
        """Generate epstab (dielectric function table)."""
        eps_list = []
        
        for i, material in enumerate(self.complete_materials):
            eps_code = self._material_to_eps(material, material_index=i)
            eps_list.append(eps_code)
        
        eps_str = ', '.join(eps_list)
        code = f"epstab = {{ {eps_str} }};"
        return code
    
    def _generate_single_material(self, material, material_index):
        """Generate single material definition."""
        return self._material_to_eps(material, material_index)
    
    def _material_to_eps(self, material, material_index=0):
        """Convert material specification to MATLAB epsilon code."""
        material_map = {
            'air': "epsconst(1)",
            'vacuum': "epsconst(1)",
            'water': "epsconst(1.33^2)",
            'glass': "epsconst(2.25)",
            'silicon': "epsconst(11.7)",
            'sapphire': "epsconst(3.13)",
            'sio2': "epsconst(2.25)",
            'agcl': "epsconst(2.02)",
            'gold': "epstable('gold.dat')",
            'silver': "epstable('silver.dat')",
            'aluminum': "epstable('aluminum.dat')"
        }
        
        if isinstance(material, dict):
            mat_type = material.get('type', 'constant')
            
            if mat_type == 'constant':
                epsilon = material['epsilon']
                return f"epsconst({epsilon})"
            
            elif mat_type == 'table':
                return self._handle_table_material(material, material_index)
            
            elif mat_type == 'function':
                formula = material['formula']
                unit = material.get('unit', 'nm')
                
                if unit == 'eV':
                    return f"epsfun(@(w) {formula}, 'eV')"
                else:
                    return f"epsfun(@(enei) {formula})"
            
            else:
                raise ValueError(f"Unknown custom material type: {mat_type}")
        
        elif isinstance(material, str):
            material_lower = material.lower()
            
            # Check for custom refractive index paths
            refractive_index_paths = self.config.get('refractive_index_paths', {})
            
            if material_lower in refractive_index_paths:
                custom_value = refractive_index_paths[material_lower]
                
                if isinstance(custom_value, dict):
                    if self.verbose:
                        print(f"Using custom material definition for '{material}' from refractive_index_paths")
                    return self._material_to_eps(custom_value, material_index)
                
                elif isinstance(custom_value, str):
                    custom_path = str(Path(custom_value).expanduser())
                    if self.verbose:
                        print(f"Using custom refractive index path for '{material}': {custom_path}")
                    return f"epstable('{custom_path}')"
                
                else:
                    raise ValueError(
                        f"Invalid value in refractive_index_paths for '{material}': "
                        f"expected string (file path) or dict (material definition), "
                        f"got {type(custom_value)}"
                    )
            
            if material_lower in material_map:
                return material_map[material_lower]
            else:
                raise ValueError(f"Unknown material: {material}")
        
        else:
            raise ValueError(f"Invalid material specification: {material}")
    
    def _handle_table_material(self, material, material_index):
        """Handle 'table' type material with automatic interpolation."""
        filepath = material['file']
        filepath = Path(filepath).expanduser()
        
        if self.verbose:
            print(f"\n--- Processing table material (index {material_index}) ---")
            print(f"File: {filepath}")
        
        try:
            loader = RefractiveIndexLoader(filepath, verbose=self.verbose)
            
            wavelength_range = self.config.get('wavelength_range', [400, 800, 80])
            target_wavelengths = np.linspace(
                wavelength_range[0],
                wavelength_range[1],
                wavelength_range[2]
            )
            
            n_interp, k_interp = loader.interpolate(target_wavelengths)
            
            refractive_index = n_interp + 1j * k_interp
            epsilon_complex = refractive_index ** 2
            
            self.table_materials_data[material_index] = {
                'wavelengths': target_wavelengths,
                'n': n_interp,
                'k': k_interp,
                'epsilon': epsilon_complex
            }
            
            epsilon_str = self._format_complex_array(epsilon_complex)
            matlab_code = f"epsconst({epsilon_str})"
            
            if self.verbose:
                print(f"Generated MATLAB code with {len(epsilon_complex)} wavelength points")
            
            return matlab_code
        
        except Exception as e:
            raise RuntimeError(f"Error processing table material '{filepath}': {e}")
    
    def _format_complex_array(self, complex_array):
        """Format complex numpy array for MATLAB."""
        values = []
        for val in complex_array:
            real_part = val.real
            imag_part = val.imag
            
            if imag_part >= 0:
                values.append(f"{real_part:.6f}+{imag_part:.6f}i")
            else:
                values.append(f"{real_part:.6f}{imag_part:.6f}i")
        
        return "[" + ", ".join(values) + "]"
    
    def _generate_inout(self):
        """Generate inout matrix based on structure."""
        structure_inout_map = {
            'sphere': self._inout_single,
            'cube': self._inout_single,
            'rod': self._inout_single,
            'ellipsoid': self._inout_single,
            'triangle': self._inout_single,
            'dimer_sphere': self._inout_dimer,
            'sphere_cluster_aggregate': self._inout_sphere_cluster_aggregate,
            'dimer_cube': self._inout_dimer,
            'core_shell_sphere': self._inout_core_shell_single,
            'core_shell_cube': self._inout_core_shell_single,
            'core_shell_rod': self._inout_core_shell_single,
            'dimer_core_shell_cube': self._inout_dimer_core_shell,
            'advanced_dimer_cube': self._inout_advanced_dimer_cube,
            'from_shape': self._inout_from_shape
        }
        
        if self.structure not in structure_inout_map:
            raise ValueError(f"Unknown structure: {self.structure}")
        
        return structure_inout_map[self.structure]()
    
    def _inout_single(self):
        """Inout for single particle."""
        code = "inout = [2, 1];"
        return code
    
    def _inout_dimer(self):
        """Inout for dimer (two identical particles)."""
        code = """inout = [
    2, 1;  % Particle 1
    2, 1   % Particle 2
];"""
        return code
    
    def _inout_core_shell_single(self):
        """Inout for single core-shell particle."""
        code = """inout = [
    2, 3;  % Core (particles{1}): inside=core(2), outside=shell(3)
    3, 1   % Shell (particles{2}): inside=shell(3), outside=medium(1)
];"""
        return code
    
    def _inout_dimer_core_shell(self):
        """Inout for dimer of core-shell particles."""
        code = """inout = [
    2, 3;  % P1-Core: inside=core(2), outside=shell(3)
    3, 1;  % P1-Shell: inside=shell(3), outside=medium(1)
    2, 3;  % P2-Core: inside=core(2), outside=shell(3)
    3, 1   % P2-Shell: inside=shell(3), outside=medium(1)
];"""
        return code

    def _inout_sphere_cluster_aggregate(self):
        """Inout for sphere cluster aggregate."""
        n_spheres = self.config.get('n_spheres', 1)
        
        inout_lines = []
        for i in range(n_spheres):
            if i < n_spheres - 1:
                inout_lines.append(f"    2, 1;  % Sphere {i+1}")
            else:
                inout_lines.append(f"    2, 1   % Sphere {i+1}")
        
        code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
        return code
    
    def _inout_advanced_dimer_cube(self):
        """Inout for advanced dimer cube with multi-shell structure.

        Nonlocal is applied ONLY to the outermost metal layer.
        """
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        n_shells = len(shell_layers)
        n_layers = len(materials)  # 1 (core) + n_shells
        use_nonlocal = self.nonlocal_gen.is_needed()

        # Find outermost metal
        outermost_idx, outermost_metal = self._find_outermost_metal(materials)

        if use_nonlocal:
            if outermost_metal is None:
                # No metal in outermost layer - fall back to standard
                use_nonlocal = False
                if self.verbose:
                    print("  No outermost metal - nonlocal disabled for advanced_dimer_cube")

        if not use_nonlocal:
            # Standard inout (no nonlocal)
            inout_lines = []

            # Particle 1
            if n_shells == 0:
                inout_lines.append(f"    2, 1;  % P1-Core")
            else:
                inout_lines.append(f"    2, 3;  % P1-Core: outside=shell1")

            for i in range(n_shells):
                shell_num = i + 1
                mat_idx = 2 + shell_num

                if i == n_shells - 1:
                    inout_lines.append(f"    {mat_idx}, 1;  % P1-Shell{shell_num}: outside=medium")
                else:
                    next_shell_mat = mat_idx + 1
                    inout_lines.append(f"    {mat_idx}, {next_shell_mat};  % P1-Shell{shell_num}")

            # Particle 2
            if n_shells == 0:
                inout_lines.append(f"    2, 1;  % P2-Core")
            else:
                inout_lines.append(f"    2, 3;  % P2-Core: outside=shell1")

            for i in range(n_shells):
                shell_num = i + 1
                mat_idx = 2 + shell_num

                if i == n_shells - 1:
                    inout_lines.append(f"    {mat_idx}, 1;  % P2-Shell{shell_num}: outside=medium")
                else:
                    next_shell_mat = mat_idx + 1
                    inout_lines.append(f"    {mat_idx}, {next_shell_mat};  % P2-Shell{shell_num}")

            if inout_lines:
                inout_lines[-1] = inout_lines[-1].rstrip(';')

            code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
            return code

        # === Nonlocal mode: apply only to outermost metal ===
        # Build epstab index mapping (only outermost metal gets 2 indices)
        mat_indices = {}
        epstab_idx = 2  # 1 is medium

        for i in range(n_layers):
            if i == outermost_idx:
                mat_indices[i] = {
                    'drude': epstab_idx,
                    'nonlocal': epstab_idx + 1,
                    'is_outermost_metal': True
                }
                epstab_idx += 2
            else:
                mat_indices[i] = {
                    'standard': epstab_idx,
                    'is_outermost_metal': False
                }
                epstab_idx += 1

        inout_lines = []

        for particle_name in ['P1', 'P2']:
            for layer_idx in range(n_layers):
                layer_info = mat_indices[layer_idx]
                is_outermost_layer = (layer_idx == n_layers - 1)

                if layer_info.get('is_outermost_metal'):
                    # Outermost metal: 2 boundaries
                    nonlocal_idx = layer_info['nonlocal']
                    drude_idx = layer_info['drude']
                    inout_lines.append(f"    {nonlocal_idx}, 1;  % {particle_name}-Outer: nonlocal inside, medium outside")
                    inout_lines.append(f"    {drude_idx}, {nonlocal_idx};  % {particle_name}-Inner: drude inside, nonlocal outside")
                else:
                    # Standard material: 1 boundary
                    std_idx = layer_info['standard']

                    if is_outermost_layer:
                        # Outermost but not metal
                        inout_lines.append(f"    {std_idx}, 1;  % {particle_name}-Layer{layer_idx}: outside=medium")
                    else:
                        # Get next layer's inside index
                        next_layer = mat_indices[layer_idx + 1]
                        if next_layer.get('is_outermost_metal'):
                            outside_idx = next_layer['drude']
                        else:
                            outside_idx = next_layer['standard']

                        layer_name = "Core" if layer_idx == 0 else f"Shell{layer_idx}"
                        inout_lines.append(f"    {std_idx}, {outside_idx};  % {particle_name}-{layer_name}")

        if inout_lines:
            inout_lines[-1] = inout_lines[-1].rstrip(';')

        code = f"% Nonlocal applied to outermost metal: {outermost_metal}\n"
        code += "inout = [\n" + "\n".join(inout_lines) + "\n];"
        return code
    
    def _inout_from_shape(self):
        """Inout for DDA shape file with multiple materials."""
        n_materials = len(self.config.get('materials', []))
        
        if n_materials == 1:
            code = "inout = [2, 1];"
        elif n_materials == 2:
            code = """inout = [
    2, 1;  % Material 1
    3, 1   % Material 2
];"""
        else:
            inout_lines = []
            for i in range(n_materials):
                mat_idx = i + 2
                inout_lines.append(f"    {mat_idx}, 1;  % Material {i+1}")
            
            if inout_lines:
                inout_lines[-1] = inout_lines[-1].rstrip(';')
            
            code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
        
        return code
    
    def _generate_closed(self):
        """Generate closed surfaces specification.

        For nonlocal mode, only the outermost metal layer gets an additional boundary.
        """
        use_nonlocal = self.nonlocal_gen.is_needed()
        materials = self.config.get('materials', [])

        # Check if outermost layer is a metal
        outermost_idx, outermost_metal = self._find_outermost_metal(materials)

        # If nonlocal enabled but no outermost metal, disable nonlocal for closed calculation
        if use_nonlocal and outermost_metal is None:
            use_nonlocal = False

        structure_closed_map = {
            'sphere': "closed = 1;",
            'cube': "closed = 1;",
            'rod': "closed = 1;",
            'ellipsoid': "closed = 1;",
            'triangle': "closed = 1;",
            'dimer_sphere': "closed = [1, 2];",
            'dimer_cube': "closed = [1, 2];",
            'core_shell_sphere': "closed = [1, 2];",
            'core_shell_cube': "closed = [1, 2];",
            'core_shell_rod': "closed = [1, 2];",
            'dimer_core_shell_cube': "closed = [1, 2, 3, 4];",
            'sphere_cluster_aggregate': self._closed_sphere_cluster_aggregate,
            'advanced_dimer_cube': self._closed_advanced_dimer_cube,
            'from_shape': self._closed_from_shape
        }

        if self.structure not in structure_closed_map:
            raise ValueError(f"Unknown structure: {self.structure}")

        result = structure_closed_map[self.structure]

        if use_nonlocal and isinstance(result, str):
            # Calculate closed indices for nonlocal mode
            # Only the outermost layer gets 2 boundaries, others get 1
            return self._calculate_closed_for_nonlocal(outermost_idx)

        if callable(result):
            if self.structure == 'advanced_dimer_cube':
                return result(use_nonlocal=use_nonlocal, outermost_idx=outermost_idx)
            else:
                return result()
        else:
            return result

    def _calculate_closed_for_nonlocal(self, outermost_idx):
        """Calculate closed indices when nonlocal is applied to outermost metal only."""
        materials = self.config.get('materials', [])
        n_layers = len(materials)
        structure = self.structure

        if 'dimer' in structure:
            n_particles = 2
        else:
            n_particles = 1

        # Calculate total boundaries per particle:
        # - Each layer has 1 boundary
        # - Except outermost metal which has 2 boundaries (outer + inner)
        boundaries_per_particle = n_layers + 1  # +1 for the extra boundary on outermost metal

        total_boundaries = n_particles * boundaries_per_particle
        closed_indices = list(range(1, total_boundaries + 1))

        return f"closed = [{', '.join(map(str, closed_indices))}];"

    
    def _closed_advanced_dimer_cube(self, use_nonlocal=False, outermost_idx=None):
        """Closed surfaces for advanced dimer cube.

        When nonlocal is enabled, only the outermost metal layer gets an extra boundary.
        """
        materials = self.config.get('materials', [])
        n_layers = len(materials)  # 1 (core) + n_shells
        n_particles = 2  # dimer

        if use_nonlocal and outermost_idx is not None:
            # Only outermost metal gets an extra boundary
            # Total boundaries per particle = n_layers + 1 (for the extra cover layer)
            boundaries_per_particle = n_layers + 1
            n_boundaries_total = n_particles * boundaries_per_particle

            if self.verbose:
                print(f"  ✓ Advanced dimer with nonlocal on outermost metal only:")
                print(f"    - {n_layers} layers per particle + 1 extra for nonlocal = {boundaries_per_particle}")
                print(f"    - Total boundaries: {n_boundaries_total}")
        else:
            # Standard mode: each layer = 1 boundary
            boundaries_per_particle = n_layers
            n_boundaries_total = n_particles * boundaries_per_particle

        closed_indices = list(range(1, n_boundaries_total + 1))
        return f"closed = [{', '.join(map(str, closed_indices))}];"
    
    def _closed_from_shape(self):
        """Closed surfaces for DDA shape file."""
        n_materials = len(self.config.get('materials', []))
        
        if n_materials == 0:
            raise ValueError("No materials specified for DDA shape file")
        elif n_materials == 1:
            return "closed = 1;"
        else:
            closed_indices = list(range(1, n_materials + 1))
            return f"closed = [{', '.join(map(str, closed_indices))}];"

    def _closed_sphere_cluster_aggregate(self):
        """Closed surfaces for sphere cluster aggregate."""
        n_spheres = self.config.get('n_spheres', 1)
        closed_indices = list(range(1, n_spheres + 1))
        return f"closed = [{', '.join(map(str, closed_indices))}];"
