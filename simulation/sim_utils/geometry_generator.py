"""
Geometry Generator

Generates MATLAB code for creating various nanoparticle geometries.
"""


class GeometryGenerator:
    """Generates geometry-related MATLAB code."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
    
    def generate(self):
        """Generate geometry code based on structure type."""
        structure_map = {
            'sphere': self._sphere,
            'cube': self._cube,
            'rod': self._rod,
            'ellipsoid': self._ellipsoid,
            'triangle': self._triangle,
            'dimer_sphere': self._dimer_sphere,
            'dimer_cube': self._dimer_cube,
            'core_shell_sphere': self._core_shell_sphere,
            'core_shell_cube': self._core_shell_cube,
            'dimer_core_shell_cube': self._dimer_core_shell_cube
        }
        
        if self.structure not in structure_map:
            raise ValueError(f"Unknown structure type: {self.structure}")
        
        return structure_map[self.structure]()
    
    def _sphere(self):
        """Generate code for single sphere."""
        diameter = self.config.get('diameter', 10)
        mesh = self.config.get('mesh_density', 144)
        
        code = f"""
%% Geometry: Single Sphere
diameter = {diameter};
p = trisphere({mesh}, diameter);
particles = {{p}};
"""
        return code
    
    def _cube(self):
        """Generate code for single cube."""
        size = self.config.get('size', 20)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        
        code = f"""
%% Geometry: Single Cube
cube_size = {size};
rounding_param = {rounding};
p = tricube({mesh}, cube_size, 'e', rounding_param);
particles = {{p}};
"""
        return code
    
    def _rod(self):
        """Generate code for rod/cylinder."""
        diameter = self.config.get('diameter', 10)
        height = self.config.get('height', 50)
        
        code = f"""
%% Geometry: Rod
diameter = {diameter};
height = {height};
p = trirod(diameter, height, [15, 20, 20]);
particles = {{p}};
"""
        return code
    
    def _ellipsoid(self):
        """Generate code for ellipsoid."""
        axes = self.config.get('axes', [10, 15, 20])
        mesh = self.config.get('mesh_density', 144)
        
        code = f"""
%% Geometry: Ellipsoid
axes = [{axes[0]}, {axes[1]}, {axes[2]}];
p = trispherescale(trisphere({mesh}, 1), axes);
particles = {{p}};
"""
        return code
    
    def _triangle(self):
        """Generate code for triangular nanoparticle."""
        side_length = self.config.get('side_length', 30)
        thickness = self.config.get('thickness', 5)
        
        code = f"""
%% Geometry: Triangle
side_length = {side_length};
thickness = {thickness};
poly = round(polygon(3, 'size', [side_length, side_length * 2/sqrt(3)]));
edge = edgeprofile(thickness, 11);
p = tripolygon(poly, edge);
particles = {{p}};
"""
        return code
    
    def _dimer_sphere(self):
        """Generate code for two coupled spheres."""
        diameter = self.config.get('diameter', 10)
        gap = self.config.get('gap', 5)
        mesh = self.config.get('mesh_density', 144)
        shift_dist = (diameter + gap) / 2
        
        code = f"""
%% Geometry: Dimer - Two Spheres
diameter = {diameter};
gap = {gap};
shift_distance = (diameter + gap) / 2;

% First sphere
p1 = trisphere({mesh}, diameter);
p1 = shift(p1, [-shift_distance, 0, 0]);

% Second sphere
p2 = trisphere({mesh}, diameter);
p2 = shift(p2, [shift_distance, 0, 0]);

particles = {{p1, p2}};
"""
        return code
    
    def _dimer_cube(self):
        """Generate code for two coupled cubes."""
        size = self.config.get('size', 20)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        shift_dist = (size + gap) / 2
        
        code = f"""
%% Geometry: Dimer - Two Cubes
cube_size = {size};
gap = {gap};
rounding_param = {rounding};
shift_distance = (cube_size + gap) / 2;

% First cube
p1 = tricube({mesh}, cube_size, 'e', rounding_param);
p1 = shift(p1, [-shift_distance, 0, 0]);

% Second cube
p2 = tricube({mesh}, cube_size, 'e', rounding_param);
p2 = shift(p2, [shift_distance, 0, 0]);

particles = {{p1, p2}};
"""
        return code
    
    def _core_shell_sphere(self):
        """Generate code for core-shell sphere."""
        core_diameter = self.config.get('core_diameter', 10)
        shell_thickness = self.config.get('shell_thickness', 5)
        mesh = self.config.get('mesh_density', 144)
        shell_diameter = core_diameter + 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Sphere
core_diameter = {core_diameter};
shell_thickness = {shell_thickness};
shell_diameter = core_diameter + 2 * shell_thickness;

% Core
p_core = trisphere({mesh}, core_diameter);

% Shell
p_shell = trisphere({mesh}, shell_diameter);

particles = {{p_shell, p_core}};
"""
        return code
    
    def _core_shell_cube(self):
        """Generate code for core-shell cube."""
        core_size = self.config.get('core_size', 15)
        shell_thickness = self.config.get('shell_thickness', 5)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        shell_size = core_size + 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Cube
core_size = {core_size};
shell_thickness = {shell_thickness};
shell_size = core_size + 2 * shell_thickness;
rounding_param = {rounding};

% Core
p_core = tricube({mesh}, core_size, 'e', rounding_param);

% Shell
p_shell = tricube({mesh}, shell_size, 'e', rounding_param);

particles = {{p_shell, p_core}};
"""
        return code
    
    def _dimer_core_shell_cube(self):
        """Generate code for two core-shell cubes (dimer)."""
        core_size = self.config.get('core_size', 20)
        shell_thickness = self.config.get('shell_thickness', 5)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        shell_size = core_size + 2 * shell_thickness
        shift_dist = (shell_size + gap) / 2
        
        code = f"""
%% Geometry: Dimer Core-Shell Cubes
core_size = {core_size};
shell_thickness = {shell_thickness};
shell_size = core_size + 2 * shell_thickness;
gap = {gap};
rounding_param = {rounding};
shift_distance = (shell_size + gap) / 2;

% First cube - Core
core1 = tricube({mesh}, core_size, 'e', rounding_param);
core1 = shift(core1, [-shift_distance, 0, 0]);

% First cube - Shell
shell1 = tricube({mesh}, shell_size, 'e', rounding_param);
shell1 = shift(shell1, [-shift_distance, 0, 0]);

% Second cube - Core
core2 = tricube({mesh}, core_size, 'e', rounding_param);
core2 = shift(core2, [shift_distance, 0, 0]);

% Second cube - Shell
shell2 = tricube({mesh}, shell_size, 'e', rounding_param);
shell2 = shift(shell2, [shift_distance, 0, 0]);

particles = {{shell1, core1, shell2, core2}};
"""
        return code