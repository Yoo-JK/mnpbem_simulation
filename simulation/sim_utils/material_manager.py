"""
Material Manager

Manages material definitions and generates corresponding MATLAB code.
NEW: Supports medium, materials, and substrate as separate configs.
"""


class MaterialManager:
    """Manages material definitions and dielectric functions."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        
        # NEW: Build complete material list from separate configs
        self.complete_materials = self._build_complete_material_list()
    
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
        # Check if substrate is used
        use_substrate = self.config.get('use_substrate', False)
        
        if use_substrate:
            # Generate substrate-specific code
            return self._generate_substrate_materials()
        else:
            # Generate normal materials code
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
    
    def _generate_substrate_materials(self):
        """Generate materials code for substrate configuration."""
        substrate = self.config.get('substrate', {})
        z_interface = substrate.get('position', 0)
        
        # Generate epstab with all materials
        epstab_code = self._generate_epstab()
        
        # Medium index = 1, Substrate index = last
        substrate_idx = len(self.complete_materials)
        
        # Generate layer structure code
        code = f"""
%% Materials and Dielectric Functions (with Substrate)
{epstab_code}

%% Layer Structure Setup
fprintf('Setting up layer structure...\\n');
z_interface = {z_interface};

% Default options for layer structure
if exist('layerstructure', 'class')
    op_layer = layerstructure.options;
    % Set up layer structure: [medium, substrate]
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
            eps_code = self._material_to_eps(material)
            eps_list.append(eps_code)
        
        # Join with commas
        eps_str = ', '.join(eps_list)
        
        code = f"epstab = {{ {eps_str} }};"
        return code
    
    def _material_to_eps(self, material):
        """
        Convert material specification to MATLAB epsilon code.
        
        Supports:
        1. Built-in materials (string)
        2. Constant epsilon (dict with 'constant')
        3. Tabulated data file (dict with 'table')
        4. Custom function (dict with 'function')
        """
        # Built-in materials
        material_map = {
            'air': "epsconst(1)",
            'vacuum': "epsconst(1)",
            'water': "epsconst(1.33^2)",
            'glass': "epsconst(2.25)",
            'silicon': "epsconst(11.7)",
            'sapphire': "epsconst(3.13)",
            'sio2': "epsconst(2.25)",
            'gold': "epstable('gold.dat')",
            'silver': "epstable('silver.dat')",
            'aluminum': "epstable('aluminum.dat')"
        }
        
        if isinstance(material, dict):
            # Custom material
            mat_type = material.get('type', 'constant')
            
            if mat_type == 'constant':
                # Constant dielectric function
                epsilon = material['epsilon']
                return f"epsconst({epsilon})"
            
            elif mat_type == 'table':
                # Wavelength-dependent from data file
                filename = material['file']
                return f"epstable('{filename}')"
            
            elif mat_type == 'function':
                # Custom function (advanced)
                formula = material['formula']
                unit = material.get('unit', 'nm')
                
                if unit == 'eV':
                    return f"epsfun(@(w) {formula}, 'eV')"
                else:
                    return f"epsfun(@(enei) {formula})"
            
            else:
                raise ValueError(f"Unknown custom material type: {mat_type}")
        
        elif isinstance(material, str):
            # Built-in material
            if material.lower() in material_map:
                return material_map[material.lower()]
            else:
                raise ValueError(f"Unknown material: {material}")
        
        else:
            raise ValueError(f"Invalid material specification: {material}")
    
    def _generate_inout(self):
        """Generate inout matrix based on structure."""
        structure_inout_map = {
            'sphere': self._inout_single(),
            'cube': self._inout_single(),
            'rod': self._inout_single(),
            'ellipsoid': self._inout_single(),
            'triangle': self._inout_single(),
            'dimer_sphere': self._inout_dimer(),
            'dimer_cube': self._inout_dimer(),
            'core_shell_sphere': self._inout_core_shell_single(),
            'core_shell_cube': self._inout_core_shell_single(),
            'dimer_core_shell_cube': self._inout_dimer_core_shell()
        }
        
        if self.structure not in structure_inout_map:
            raise ValueError(f"Unknown structure: {self.structure}")
        
        return structure_inout_map[self.structure]
    
    def _inout_single(self):
        """Inout for single particle."""
        # complete_materials[0]: medium
        # complete_materials[1]: particle material
        code = "inout = [2, 1];"
        return code
    
    def _inout_dimer(self):
        """Inout for dimer (two identical particles)."""
        # complete_materials[0]: medium
        # complete_materials[1]: particle material
        code = """inout = [
    2, 1;  % Particle 1: inside=material, outside=medium
    2, 1   % Particle 2: inside=material, outside=medium
];"""
        return code
    
    def _inout_core_shell_single(self):
        """Inout for single core-shell particle."""
        # complete_materials[0]: medium
        # complete_materials[1]: shell material
        # complete_materials[2]: core material
        code = """inout = [
    2, 1;  % Shell: inside=shell_material, outside=medium
    3, 2   % Core:  inside=core_material, outside=shell
];"""
        return code
    
    def _inout_dimer_core_shell(self):
        """Inout for dimer of core-shell particles."""
        # complete_materials[0]: medium
        # complete_materials[1]: shell material
        # complete_materials[2]: core material
        code = """inout = [
    2, 1;  % Shell1: inside=shell, outside=medium
    3, 2;  % Core1:  inside=core, outside=shell
    2, 1;  % Shell2: inside=shell, outside=medium
    3, 2   % Core2:  inside=core, outside=shell
];"""
        return code
    
    def _generate_closed(self):
        """Generate closed surfaces specification."""
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
            'dimer_core_shell_cube': "closed = [1, 2, 3, 4];"
        }
        
        if self.structure not in structure_closed_map:
            raise ValueError(f"Unknown structure: {self.structure}")
        
        return structure_closed_map[self.structure]