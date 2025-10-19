"""
Geometry Generator

Generates MATLAB code for creating various nanoparticle geometries.
Supports:
  - Built-in structures (MNPBEM predefined shapes)
  - DDA .shape files (with material indices)
"""

import numpy as np
from pathlib import Path


# ============================================================================
# DDA Shape File Loader
# ============================================================================

class ShapeFileLoader:
    """Load and process DDA .shape files with material indices."""
    
    def __init__(self, shape_path, voxel_size=1.0, method='surface', verbose=False):
        """
        Initialize shape file loader.
        
        Args:
            shape_path: Path to DDA .shape file
            voxel_size: Physical size of each voxel (nm)
            method: 'surface' (fast) or 'cube' (accurate)
            verbose: Print debug information
        """
        self.shape_path = Path(shape_path)
        self.voxel_size = voxel_size
        self.method = method
        self.verbose = verbose
        
        if not self.shape_path.exists():
            raise FileNotFoundError(f"Shape file not found: {self.shape_path}")
        
        if method not in ['surface', 'cube']:
            raise ValueError(f"method must be 'surface' or 'cube', got '{method}'")
        
        self.voxel_data = None  # Will store [i, j, k, mat_idx]
        self.unique_materials = None
        self.material_particles = {}  # {mat_idx: {'vertices': ..., 'faces': ...}}
    
    def load(self):
        """Load shape file and extract voxel data with materials."""
        if self.verbose:
            print(f"  Loading DDA shape file: {self.shape_path}")
        
        # Load shape file: expected format [i, j, k, mat_type]
        # Some DDA files have additional columns (Jx, Jy, Jz), we only need first 4
        data = np.loadtxt(self.shape_path, dtype=int)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 4:
            raise ValueError(
                f"Shape file must have at least 4 columns [i,j,k,mat_type], "
                f"got {data.shape[1]} columns"
            )
        
        # Extract [i, j, k, mat_type]
        self.voxel_data = data[:, :4]
        
        # Find unique materials
        self.unique_materials = np.unique(self.voxel_data[:, 3])
        
        if self.verbose:
            print(f"    Total voxels: {len(self.voxel_data)}")
            print(f"    Unique materials: {self.unique_materials.tolist()}")
            for mat_idx in self.unique_materials:
                count = np.sum(self.voxel_data[:, 3] == mat_idx)
                print(f"      Material {mat_idx}: {count} voxels")
        
        # Convert each material to mesh
        for mat_idx in self.unique_materials:
            # Get voxels for this material
            mat_voxels = self.voxel_data[self.voxel_data[:, 3] == mat_idx][:, :3]
            
            if self.verbose:
                print(f"    Converting material {mat_idx}...")
            
            # Convert to mesh
            if self.method == 'surface':
                vertices, faces = self._voxels_to_surface_mesh(mat_voxels)
            else:
                vertices, faces = self._voxels_to_cube_mesh(mat_voxels)
            
            self.material_particles[mat_idx] = {
                'vertices': vertices,
                'faces': faces
            }
            
            if self.verbose:
                print(f"      → {len(vertices)} vertices, {len(faces)} faces")
        
        return self.material_particles
    
    def _voxels_to_surface_mesh(self, voxel_coords):
        """Convert voxels to surface mesh (only external faces)."""
        voxel_set = set(map(tuple, voxel_coords))
        
        vertices_list = []
        faces_list = []
        vertex_map = {}
        
        # Cube face definitions
        cube_face_offsets = [
            [[0,0,0], [1,0,0], [1,1,0], [0,1,0]],  # bottom
            [[0,0,1], [0,1,1], [1,1,1], [1,0,1]],  # top
            [[0,0,0], [0,1,0], [0,1,1], [0,0,1]],  # left
            [[1,0,0], [1,0,1], [1,1,1], [1,1,0]],  # right
            [[0,0,0], [0,0,1], [1,0,1], [1,0,0]],  # front
            [[0,1,0], [1,1,0], [1,1,1], [0,1,1]]   # back
        ]
        
        neighbors = [
            [0, 0, -1], [0, 0, 1], [-1, 0, 0],
            [1, 0, 0], [0, -1, 0], [0, 1, 0]
        ]
        
        for voxel in voxel_coords:
            i, j, k = voxel
            
            for face_idx, neighbor_offset in enumerate(neighbors):
                neighbor = (i + neighbor_offset[0],
                           j + neighbor_offset[1],
                           k + neighbor_offset[2])
                
                if neighbor not in voxel_set:
                    face_verts_offsets = cube_face_offsets[face_idx]
                    vert_indices = []
                    
                    for vert_offset in face_verts_offsets:
                        vx = (i + vert_offset[0]) * self.voxel_size
                        vy = (j + vert_offset[1]) * self.voxel_size
                        vz = (k + vert_offset[2]) * self.voxel_size
                        vert_key = (vx, vy, vz)
                        
                        if vert_key not in vertex_map:
                            vertex_map[vert_key] = len(vertices_list)
                            vertices_list.append([vx, vy, vz])
                        
                        vert_indices.append(vertex_map[vert_key] + 1)  # MATLAB 1-indexing
                    
                    # Split quad to triangles
                    faces_list.append([vert_indices[0], vert_indices[1], vert_indices[2]])
                    faces_list.append([vert_indices[0], vert_indices[2], vert_indices[3]])
        
        return np.array(vertices_list), np.array(faces_list)
    
    def _voxels_to_cube_mesh(self, voxel_coords):
        """Convert each voxel to a cube mesh."""
        cube_vert_template = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        cube_face_template = np.array([
            [1, 2, 3], [1, 3, 4],  # bottom
            [5, 8, 7], [5, 7, 6],  # top
            [1, 5, 6], [1, 6, 2],  # front
            [4, 3, 7], [4, 7, 8],  # back
            [1, 4, 8], [1, 8, 5],  # left
            [2, 6, 7], [2, 7, 3]   # right
        ])
        
        all_verts = []
        all_faces = []
        
        for voxel in voxel_coords:
            i, j, k = voxel
            
            cube_verts = cube_vert_template * self.voxel_size
            cube_verts[:, 0] += i * self.voxel_size
            cube_verts[:, 1] += j * self.voxel_size
            cube_verts[:, 2] += k * self.voxel_size
            
            vert_offset = len(all_verts)
            all_verts.extend(cube_verts)
            all_faces.extend(cube_face_template + vert_offset)
        
        return np.array(all_verts), np.array(all_faces)
    
    def generate_matlab_code(self, material_names):
        """
        Generate MATLAB code for all material particles.
        
        Args:
            material_names: List or dict of material names
                           If list: materials[0] corresponds to mat_idx=1, etc.
                           If dict: {mat_idx: material_name}
        
        Returns:
            str: MATLAB code for creating all particles
        """
        if self.material_particles is None:
            raise RuntimeError("Shape file not loaded. Call load() first.")
        
        # Convert material_names to dict if it's a list
        if isinstance(material_names, list):
            # Map: mat_idx (1-based in DDA) → material name
            mat_name_dict = {i+1: name for i, name in enumerate(material_names)}
        else:
            mat_name_dict = material_names
        
        code = """
%% Geometry: From DDA Shape File
fprintf('Creating particles from DDA shape file...\\n');
fprintf('  Voxel size: %.2f nm\\n', {voxel_size});
fprintf('  Method: {method}\\n');
fprintf('  Number of materials: %d\\n', {n_materials});

""".format(
            voxel_size=self.voxel_size,
            method=self.method,
            n_materials=len(self.unique_materials)
        )
        
        # Generate code for each material
        particles_list = []
        
        for mat_idx in sorted(self.unique_materials):
            data = self.material_particles[mat_idx]
            vertices = data['vertices']
            faces = data['faces']
            
            mat_name = mat_name_dict.get(mat_idx, f'material_{mat_idx}')
            
            # Generate MATLAB arrays
            verts_str, faces_str = self._arrays_to_matlab(vertices, faces)
            
            code += f"""
% Material index {mat_idx}: {mat_name}
verts_{mat_idx} = {verts_str};
faces_{mat_idx} = {faces_str};
p{mat_idx} = particle(verts_{mat_idx}, faces_{mat_idx});
fprintf('  Material {mat_idx} ({mat_name}): %d vertices, %d faces\\n', ...
        size(verts_{mat_idx}, 1), size(faces_{mat_idx}, 1));
"""
            particles_list.append(f'p{mat_idx}')
        
        # Create particles cell array
        particles_str = ', '.join(particles_list)
        code += f"\nparticles = {{{particles_str}}};\n"
        
        return code
    
    def _arrays_to_matlab(self, vertices, faces):
        """Convert numpy arrays to MATLAB format."""
        # Vertices
        verts_str = "[\n"
        for v in vertices:
            verts_str += f"    {v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f};\n"
        verts_str += "]"
        
        # Faces
        faces_str = "[\n"
        for f in faces:
            faces_str += f"    {f[0]}, {f[1]}, {f[2]};\n"
        faces_str += "]"
        
        return verts_str, faces_str


# ============================================================================
# Geometry Generator
# ============================================================================

class GeometryGenerator:
    """Generates geometry-related MATLAB code."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
    
    def generate(self):
        """Generate geometry code based on structure type."""
        structure_map = {
            # Built-in structures
            'sphere': self._sphere,
            'cube': self._cube,
            'rod': self._rod,
            'ellipsoid': self._ellipsoid,
            'triangle': self._triangle,
            'dimer_sphere': self._dimer_sphere,
            'dimer_cube': self._dimer_cube,
            'core_shell_sphere': self._core_shell_sphere,
            'core_shell_cube': self._core_shell_cube,
            'dimer_core_shell_cube': self._dimer_core_shell_cube,
            # DDA shape file
            'from_shape': self._from_shape,
        }
        
        if self.structure not in structure_map:
            raise ValueError(f"Unknown structure type: {self.structure}")
        
        return structure_map[self.structure]()
    
    # ========================================================================
    # Built-in Structures
    # ========================================================================
    
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
p = trisphere({mesh}, 1);
p.verts(:, 1) = p.verts(:, 1) * {axes[0]};
p.verts(:, 2) = p.verts(:, 2) * {axes[1]};
p.verts(:, 3) = p.verts(:, 3) * {axes[2]};
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
        
        code = f"""
%% Geometry: Dimer - Two Spheres
diameter = {diameter};
gap = {gap};
shift_distance = (diameter + gap) / 2;

p1 = trisphere({mesh}, diameter);
p1 = shift(p1, [-shift_distance, 0, 0]);

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
        
        code = f"""
%% Geometry: Dimer - Two Cubes
cube_size = {size};
gap = {gap};
rounding_param = {rounding};
shift_distance = (cube_size + gap) / 2;

p1 = tricube({mesh}, cube_size, 'e', rounding_param);
p1 = shift(p1, [-shift_distance, 0, 0]);

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

p_core = trisphere({mesh}, core_diameter);
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

p_core = tricube({mesh}, core_size, 'e', rounding_param);
p_shell = tricube({mesh}, shell_size, 'e', rounding_param);

particles = {{p_shell, p_core}};
"""
        return code
    
    def _dimer_core_shell_cube(self):
        """Generate code for two core-shell cubes."""
        core_size = self.config.get('core_size', 20)
        shell_thickness = self.config.get('shell_thickness', 5)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        shell_size = core_size + 2 * shell_thickness
        
        code = f"""
%% Geometry: Dimer Core-Shell Cubes
core_size = {core_size};
shell_thickness = {shell_thickness};
shell_size = core_size + 2 * shell_thickness;
gap = {gap};
rounding_param = {rounding};
shift_distance = (shell_size + gap) / 2;

core1 = tricube({mesh}, core_size, 'e', rounding_param);
core1 = shift(core1, [-shift_distance, 0, 0]);

shell1 = tricube({mesh}, shell_size, 'e', rounding_param);
shell1 = shift(shell1, [-shift_distance, 0, 0]);

core2 = tricube({mesh}, core_size, 'e', rounding_param);
core2 = shift(core2, [shift_distance, 0, 0]);

shell2 = tricube({mesh}, shell_size, 'e', rounding_param);
shell2 = shift(shell2, [shift_distance, 0, 0]);

particles = {{shell1, core1, shell2, core2}};
"""
        return code
    
    # ========================================================================
    # DDA Shape File
    # ========================================================================
    
    def _from_shape(self):
        """Generate code for DDA shape file import."""
        shape_file = self.config.get('shape_file')
        if not shape_file:
            raise ValueError("'shape_file' must be specified for 'from_shape' structure")
        
        voxel_size = self.config.get('voxel_size', 1.0)
        method = self.config.get('voxel_method', 'surface')
        materials = self.config.get('materials', [])
        
        if not materials:
            raise ValueError("'materials' list must be specified for DDA shape files")
        
        if self.verbose:
            print(f"Loading DDA shape file...")
        
        # Load shape file using Python
        loader = ShapeFileLoader(shape_file, voxel_size=voxel_size, method=method, verbose=self.verbose)
        loader.load()
        
        # Generate MATLAB code with material names
        code = loader.generate_matlab_code(materials)
        
        return code