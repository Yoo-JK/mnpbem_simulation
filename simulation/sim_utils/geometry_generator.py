"""
Geometry Generator

Generates MATLAB code for creating various nanoparticle geometries.
Supports:
  - Built-in structures (MNPBEM predefined shapes)
  - DDA .shape files (with material indices)
  - Large mesh support via .mat file export
  - Nonlocal quantum corrections (cover layers)
"""

import numpy as np
from pathlib import Path
from .nonlocal_generator import NonlocalGenerator


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
        
        # Read file and skip non-numeric lines
        data_lines = []
        with open(self.shape_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not (line[0].isdigit() or line[0] == '-'):
                    if self.verbose:
                        print(f"    Skipping header/comment line: {line}")
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        i, j, k, mat_type = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        data_lines.append([i, j, k, mat_type])
                except (ValueError, IndexError):
                    if self.verbose:
                        print(f"    Skipping invalid line: {line}")
                    continue
        
        if not data_lines:
            raise ValueError(f"No valid voxel data found in {self.shape_path}")
        
        data = np.array(data_lines, dtype=int)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 4:
            raise ValueError(
                f"Shape file must have at least 4 columns [i,j,k,mat_type], "
                f"got {data.shape[1]} columns"
            )
        
        self.voxel_data = data[:, :4]
        self.unique_materials = np.unique(self.voxel_data[:, 3])
        
        if self.verbose:
            print(f"    Total voxels: {len(self.voxel_data)}")
            print(f"    Unique materials: {self.unique_materials.tolist()}")
            for mat_idx in self.unique_materials:
                count = np.sum(self.voxel_data[:, 3] == mat_idx)
                print(f"      Material {mat_idx}: {count} voxels")
        
        # Convert each material to mesh
        for mat_idx in self.unique_materials:
            mat_voxels = self.voxel_data[self.voxel_data[:, 3] == mat_idx][:, :3]
            
            if self.verbose:
                print(f"    Converting material {mat_idx}...")
            
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
                        
                        vert_indices.append(vertex_map[vert_key] + 1)
                    
                    faces_list.append([vert_indices[0], vert_indices[1], vert_indices[2], np.nan])
                    faces_list.append([vert_indices[0], vert_indices[2], vert_indices[3], np.nan])
        
        return np.array(vertices_list), np.array(faces_list)
    
    def _voxels_to_cube_mesh(self, voxel_coords):
        """Convert each voxel to a cube mesh."""
        cube_vert_template = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        cube_face_template = np.array([
            [1, 2, 3, np.nan], [1, 3, 4, np.nan],
            [5, 8, 7, np.nan], [5, 7, 6, np.nan],
            [1, 5, 6, np.nan], [1, 6, 2, np.nan],
            [4, 3, 7, np.nan], [4, 7, 8, np.nan],
            [1, 4, 8, np.nan], [1, 8, 5, np.nan],
            [2, 6, 7, np.nan], [2, 7, 3, np.nan]
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
    
    def generate_matlab_code(self, material_names, output_dir=None):
        """Generate MATLAB code for all material particles."""
        if self.material_particles is None:
            raise RuntimeError("Shape file not loaded. Call load() first.")
        
        if isinstance(material_names, list):
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
        
        use_mat_files = False
        total_vertices = sum(len(data['vertices']) for data in self.material_particles.values())
        
        if total_vertices > 100000:
            use_mat_files = True
            if output_dir is None:
                raise ValueError("output_dir must be provided for large meshes")
            
            if self.verbose:
                print(f"  Large mesh detected ({total_vertices} vertices)")
                print(f"  Saving geometry data to .mat files in {output_dir}")
        
        particles_list = []
        
        for mat_idx in sorted(self.unique_materials):
            data = self.material_particles[mat_idx]
            vertices = data['vertices']
            faces = data['faces']
            
            mat_name = mat_name_dict.get(mat_idx, f'material_{mat_idx}')
            
            if use_mat_files:
                try:
                    import scipy.io as sio
                except ImportError:
                    raise ImportError("scipy is required for saving large meshes")
                
                mat_filename = f'geometry_mat{mat_idx}.mat'
                mat_filepath = Path(output_dir) / mat_filename
                
                faces_matlab = faces.copy()
                if faces_matlab.min() == 0:
                    faces_matlab = faces_matlab + 1
                
                sio.savemat(
                    str(mat_filepath),
                    {
                        f'verts_{mat_idx}': vertices,
                        f'faces_{mat_idx}': faces_matlab
                    },
                    do_compression=True
                )
                
                if self.verbose:
                    print(f"    Saved material {mat_idx} to {mat_filename}")
                
                code += f"""
% Material index {mat_idx}: {mat_name}
fprintf('  Loading material {mat_idx} ({mat_name}) from file...\\n');
geom_data_{mat_idx} = load('{mat_filename}');
verts_{mat_idx} = geom_data_{mat_idx}.verts_{mat_idx};
faces_{mat_idx} = geom_data_{mat_idx}.faces_{mat_idx};

% DDA meshes use flat triangular faces only
p{mat_idx} = particle(verts_{mat_idx}, faces_{mat_idx}, op, 'interp', 'flat');
fprintf('  Material {mat_idx} ({mat_name}): %d vertices, %d faces\\n', ...
        size(verts_{mat_idx}, 1), size(faces_{mat_idx}, 1));
"""
            else:
                verts_str, faces_str = self._arrays_to_matlab(vertices, faces)
                
                code += f"""
% Material index {mat_idx}: {mat_name}
verts_{mat_idx} = {verts_str};
faces_{mat_idx} = {faces_str};

% DDA meshes use flat triangular faces only
p{mat_idx} = particle(verts_{mat_idx}, faces_{mat_idx}, op, 'interp', 'flat');
fprintf('  Material {mat_idx} ({mat_name}): %d vertices, %d faces\\n', ...
        size(verts_{mat_idx}, 1), size(faces_{mat_idx}, 1));
"""
            
            particles_list.append(f'p{mat_idx}')
        
        particles_str = ', '.join(particles_list)
        code += f"\nparticles = {{{particles_str}}};\n"
        
        return code
    
    def _arrays_to_matlab(self, vertices, faces):
        """Convert numpy arrays to MATLAB format."""
        verts_str = "[\n"
        for v in vertices:
            verts_str += f"    {v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f};\n"
        verts_str += "]"
        
        faces_str = "[\n"
        for f in faces:
            if len(f) >= 4:
                if np.isnan(f[3]):
                    faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, NaN;\n"
                else:
                    faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, {int(f[3])};\n"
            else:
                faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, NaN;\n"
        faces_str += "]"
        
        return verts_str, faces_str


# ============================================================================
# Adaptive Cube Mesh Generator
# ============================================================================

class AdaptiveCubeMesh:
    """
    Generate adaptive mesh for rounded cube with per-face density control.

    Uses edge-unified approach: all edges share the same density (max of adjacent faces),
    while face interiors can have different densities with proper transition triangles.
    """

    def __init__(self, size, rounding=0.2, verbose=False):
        """
        Initialize adaptive cube mesh generator.

        Args:
            size: Cube edge length (nm)
            rounding: Edge rounding parameter (0-1, 0=sharp, 1=sphere-like)
            verbose: Print debug information
        """
        self.size = size
        self.rounding = rounding
        self.verbose = verbose
        self.half_size = size / 2
        self.r = rounding * self.half_size * 0.5

    def generate(self, densities):
        """
        Generate cube mesh with per-face densities using gradual adaptive approach.

        Each face has its own edge density, and shared edges use max of adjacent faces.
        This allows gradual transition from gap (fine) to back (coarse).

        Args:
            densities: dict with keys '+x', '-x', '+y', '-y', '+z', '-z'
                      Each value is the density for that face (both edge and interior)

        Returns:
            vertices: (N, 3) array of vertex coordinates
            faces: (M, 4) array of face indices (1-indexed, 4th column is NaN)
        """
        # Define which faces share each edge
        # Each edge is shared by exactly 2 faces
        # Format: edge_key -> (face1, face2)
        edge_adjacency = {
            # Edges along x-axis (at corners of y-z plane)
            'x_pp': ('+y', '+z'),  # y=+h, z=+h
            'x_pm': ('+y', '-z'),  # y=+h, z=-h
            'x_mp': ('-y', '+z'),  # y=-h, z=+h
            'x_mm': ('-y', '-z'),  # y=-h, z=-h
            # Edges along y-axis (at corners of x-z plane)
            'y_pp': ('+x', '+z'),  # x=+h, z=+h
            'y_pm': ('+x', '-z'),  # x=+h, z=-h
            'y_mp': ('-x', '+z'),  # x=-h, z=+h
            'y_mm': ('-x', '-z'),  # x=-h, z=-h
            # Edges along z-axis (at corners of x-y plane)
            'z_pp': ('+x', '+y'),  # x=+h, y=+h
            'z_pm': ('+x', '-y'),  # x=+h, y=-h
            'z_mp': ('-x', '+y'),  # x=-h, y=+h
            'z_mm': ('-x', '-y'),  # x=-h, y=-h
        }

        # Calculate density for each edge (max of two adjacent faces)
        edge_densities = {}
        for edge_key, (face1, face2) in edge_adjacency.items():
            d1 = densities.get(face1, 12)
            d2 = densities.get(face2, 12)
            edge_densities[edge_key] = max(d1, d2)

        # Map each face to its 4 edges
        # Order: bottom, right, top, left (going around the face)
        # Based on _generate_face_gradual logic:
        #   - bottom: ax2 = -h, ax1 varies from -h to +h
        #   - right:  ax1 = +h, ax2 varies from -h to +h
        #   - top:    ax2 = +h, ax1 varies from +h to -h
        #   - left:   ax1 = -h, ax2 varies from +h to -h
        face_edges = {
            # +x face: ax1=y, ax2=z, face at x=+h
            '+x': ['y_pm', 'z_pp', 'y_pp', 'z_pm'],
            # -x face: ax1=y, ax2=z, face at x=-h
            '-x': ['y_mm', 'z_mp', 'y_mp', 'z_mm'],
            # +y face: ax1=x, ax2=z, face at y=+h
            '+y': ['x_pm', 'z_pp', 'x_pp', 'z_mp'],
            # -y face: ax1=x, ax2=z, face at y=-h
            '-y': ['x_mm', 'z_pm', 'x_mp', 'z_mm'],
            # +z face: ax1=x, ax2=y, face at z=+h
            '+z': ['x_mp', 'y_pp', 'x_pp', 'y_mp'],
            # -z face: ax1=x, ax2=y, face at z=-h
            '-z': ['x_mm', 'y_pm', 'x_pm', 'y_mm'],
        }

        if self.verbose:
            print(f"  Gradual adaptive mesh:")
            for face_name, d in densities.items():
                edges = face_edges[face_name]
                edge_d = [edge_densities[e] for e in edges]
                print(f"    {face_name}: interior={d}, edges={edge_d}")

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Face definitions
        face_defs = {
            '+x': (0, +1),
            '-x': (0, -1),
            '+y': (1, +1),
            '-y': (1, -1),
            '+z': (2, +1),
            '-z': (2, -1),
        }

        for face_name, (axis, sign) in face_defs.items():
            inner_density = densities.get(face_name, 12)

            # Get edge densities for this face's 4 edges
            edges = face_edges[face_name]
            face_edge_densities = [edge_densities[e] for e in edges]

            # Generate face with gradual approach
            verts, faces = self._generate_face_gradual(
                axis, sign, face_edge_densities, inner_density
            )

            # Apply rounding
            verts = self._apply_rounding(verts)

            # Offset face indices
            faces_offset = faces + vertex_offset

            all_vertices.append(verts)
            all_faces.append(faces_offset)

            vertex_offset += len(verts)

        # Concatenate
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        # Merge duplicate vertices at shared edges
        vertices, faces = self._merge_vertices(vertices, faces)

        # Remove degenerate triangles (where two or more vertices are the same)
        # This can happen when rounding moves nearby vertices to the same position
        valid_mask = []
        for face in faces:
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            is_valid = (v1 != v2) and (v2 != v3) and (v1 != v3)
            valid_mask.append(is_valid)

        faces = faces[valid_mask]

        if self.verbose:
            n_removed = sum(1 for v in valid_mask if not v)
            if n_removed > 0:
                print(f"  Removed {n_removed} degenerate triangles")
            print(f"  Total: {len(vertices)} vertices, {len(faces)} faces")

        return vertices, faces

    def _generate_face_gradual(self, axis, sign, edge_densities, inner_n):
        """
        Generate a face with per-edge densities and interior density.

        Args:
            axis: 0=x, 1=y, 2=z (normal axis)
            sign: +1 or -1 (direction of normal)
            edge_densities: [bottom, right, top, left] densities for 4 edges
            inner_n: density for face interior

        Returns:
            vertices, faces arrays
        """
        h = self.half_size

        # Check if all edges and interior have same density
        if all(d == inner_n for d in edge_densities):
            return self._generate_face_simple(axis, sign, inner_n)

        # Get the two axes that span this face
        if axis == 0:  # x-face, spans y and z
            ax1, ax2 = 1, 2
        elif axis == 1:  # y-face, spans x and z
            ax1, ax2 = 0, 2
        else:  # z-face, spans x and y
            ax1, ax2 = 0, 1

        vertices = []
        faces_list = []

        # Edge densities: [bottom, right, top, left]
        n_bottom, n_right, n_top, n_left = edge_densities

        # === 1. Generate boundary vertices for each edge ===
        boundary_verts = []

        # Bottom edge: ax2 = -h, ax1 varies from -h to +h
        bottom_coords = np.linspace(-h, h, n_bottom + 1)
        for i in range(n_bottom):  # Exclude last to avoid duplicate corner
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = bottom_coords[i]
            v[ax2] = -h
            boundary_verts.append(v)
        bottom_start = 0
        bottom_count = n_bottom

        # Right edge: ax1 = +h, ax2 varies from -h to +h
        right_coords = np.linspace(-h, h, n_right + 1)
        for i in range(n_right):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = h
            v[ax2] = right_coords[i]
            boundary_verts.append(v)
        right_start = bottom_count
        right_count = n_right

        # Top edge: ax2 = +h, ax1 varies from +h to -h
        top_coords = np.linspace(h, -h, n_top + 1)
        for i in range(n_top):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = top_coords[i]
            v[ax2] = h
            boundary_verts.append(v)
        top_start = right_start + right_count
        top_count = n_top

        # Left edge: ax1 = -h, ax2 varies from +h to -h
        left_coords = np.linspace(h, -h, n_left + 1)
        for i in range(n_left):
            v = [0, 0, 0]
            v[axis] = sign * h
            v[ax1] = -h
            v[ax2] = left_coords[i]
            boundary_verts.append(v)
        left_start = top_start + top_count
        left_count = n_left

        n_boundary = len(boundary_verts)

        # === 2. Generate interior grid ===
        # Shrink interior to create transition zone
        margin = h * 0.15  # 15% margin for transition
        inner_h = h - margin
        inner_coords = np.linspace(-inner_h, inner_h, inner_n + 1)

        interior_verts = []
        for i in range(inner_n + 1):
            for j in range(inner_n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h
                v[ax1] = inner_coords[i]
                v[ax2] = inner_coords[j]
                interior_verts.append(v)

        n_interior = (inner_n + 1) ** 2

        # Combine all vertices
        vertices = boundary_verts + interior_verts

        # === 3. Generate interior grid faces ===
        for i in range(inner_n):
            for j in range(inner_n):
                v00 = n_boundary + i * (inner_n + 1) + j
                v10 = n_boundary + (i + 1) * (inner_n + 1) + j
                v01 = n_boundary + i * (inner_n + 1) + (j + 1)
                v11 = n_boundary + (i + 1) * (inner_n + 1) + (j + 1)

                if sign > 0:
                    faces_list.append([v00 + 1, v10 + 1, v11 + 1, np.nan])
                    faces_list.append([v00 + 1, v11 + 1, v01 + 1, np.nan])
                else:
                    faces_list.append([v00 + 1, v11 + 1, v10 + 1, np.nan])
                    faces_list.append([v00 + 1, v01 + 1, v11 + 1, np.nan])

        # === 4. Generate transition triangles for each edge ===
        # Bottom edge to interior bottom
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            bottom_start, bottom_count, 'bottom'
        )
        # Right edge to interior right
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            right_start, right_count, 'right'
        )
        # Top edge to interior top
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            top_start, top_count, 'top'
        )
        # Left edge to interior left
        self._add_transition_gradual(
            faces_list, boundary_verts, n_boundary, inner_n, sign,
            left_start, left_count, 'left'
        )

        return np.array(vertices), np.array(faces_list)

    def _add_transition_gradual(self, faces, boundary_verts, n_boundary, inner_n, sign,
                                 edge_start, edge_count, side):
        """Add transition triangles between a boundary edge and interior grid edge."""
        # Get boundary indices for this edge
        b_indices = [(edge_start + i) % n_boundary for i in range(edge_count + 1)]
        if b_indices[-1] == 0:
            b_indices[-1] = n_boundary  # Wrap around

        # Actually, we need edge_count + 1 indices, but we only stored edge_count vertices
        # The last vertex is the first vertex of the next edge
        b_indices = []
        for i in range(edge_count):
            b_indices.append(edge_start + i)
        # Add the first vertex of the next edge (or wrap to 0)
        next_start = (edge_start + edge_count) % n_boundary
        b_indices.append(next_start)

        # Get interior indices for this side
        if side == 'bottom':
            i_indices = [n_boundary + j for j in range(inner_n + 1)]
        elif side == 'right':
            i_indices = [n_boundary + inner_n + j * (inner_n + 1) for j in range(inner_n + 1)]
        elif side == 'top':
            i_indices = [n_boundary + (inner_n + 1) * inner_n + (inner_n - j) for j in range(inner_n + 1)]
        else:  # left
            i_indices = [n_boundary + (inner_n - j) * (inner_n + 1) for j in range(inner_n + 1)]

        n_b = len(b_indices)
        n_i = len(i_indices)

        # Create transition triangles using ratio-based advancing
        bi, ii = 0, 0
        while bi < n_b - 1 or ii < n_i - 1:
            if bi >= n_b - 1:
                if ii < n_i - 1:
                    v1, v2, v3 = b_indices[-1], i_indices[ii], i_indices[ii + 1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    ii += 1
            elif ii >= n_i - 1:
                if bi < n_b - 1:
                    v1, v2, v3 = b_indices[bi], b_indices[bi + 1], i_indices[-1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    bi += 1
            else:
                b_ratio = bi / max(n_b - 1, 1)
                i_ratio = ii / max(n_i - 1, 1)

                if b_ratio <= i_ratio:
                    v1, v2, v3 = b_indices[bi], b_indices[bi + 1], i_indices[ii]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    bi += 1
                else:
                    v1, v2, v3 = b_indices[bi], i_indices[ii], i_indices[ii + 1]
                    if sign > 0:
                        faces.append([v1 + 1, v2 + 1, v3 + 1, np.nan])
                    else:
                        faces.append([v1 + 1, v3 + 1, v2 + 1, np.nan])
                    ii += 1

    def _generate_face_simple(self, axis, sign, n):
        """Generate a simple uniform-density face (original method)."""
        h = self.half_size

        u = np.linspace(-h, h, n + 1)
        v = np.linspace(-h, h, n + 1)
        uu, vv = np.meshgrid(u, v)

        uu = uu.flatten()
        vv = vv.flatten()

        vertices = np.zeros((len(uu), 3))

        if axis == 0:
            vertices[:, 0] = sign * h
            vertices[:, 1] = uu
            vertices[:, 2] = vv
        elif axis == 1:
            vertices[:, 0] = uu
            vertices[:, 1] = sign * h
            vertices[:, 2] = vv
        else:
            vertices[:, 0] = uu
            vertices[:, 1] = vv
            vertices[:, 2] = sign * h

        faces = []
        for i in range(n):
            for j in range(n):
                v00 = i * (n + 1) + j
                v10 = (i + 1) * (n + 1) + j
                v01 = i * (n + 1) + (j + 1)
                v11 = (i + 1) * (n + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1, np.nan])
                    faces.append([v00 + 1, v11 + 1, v01 + 1, np.nan])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1, np.nan])
                    faces.append([v00 + 1, v01 + 1, v11 + 1, np.nan])

        return vertices, np.array(faces)

    def _apply_rounding(self, vertices):
        """
        Apply rounding to vertices near edges and corners.

        Vertices near cube edges/corners are moved inward and then
        projected onto a rounded surface.
        """
        if self.rounding <= 0:
            return vertices

        h = self.half_size
        r = self.r

        # Threshold for being "near edge"
        edge_threshold = h - r

        rounded_verts = vertices.copy()

        for i, v in enumerate(vertices):
            x, y, z = v

            # Count how many coordinates are near the edge
            near_x = abs(abs(x) - h) < 1e-10
            near_y = abs(abs(y) - h) < 1e-10
            near_z = abs(abs(z) - h) < 1e-10
            n_near = near_x + near_y + near_z

            if n_near == 1:
                # On a face, check if near an edge within the face
                if not near_x:
                    if abs(y) > edge_threshold or abs(z) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
                elif not near_y:
                    if abs(x) > edge_threshold or abs(z) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
                else:  # not near_z
                    if abs(x) > edge_threshold or abs(y) > edge_threshold:
                        rounded_verts[i] = self._round_edge_vertex(v, r, h)
            elif n_near >= 2:
                # On an edge or corner - apply rounding
                rounded_verts[i] = self._round_corner_vertex(v, r, h)

        return rounded_verts

    def _round_edge_vertex(self, v, r, h):
        """Round a vertex that's on a face but near an edge."""
        x, y, z = v

        # Find which axis is the face normal
        if abs(abs(x) - h) < 1e-10:
            # On x-face, round y and z if near edge
            new_y, new_z = self._round_2d(y, z, r, h)
            return np.array([x, new_y, new_z])
        elif abs(abs(y) - h) < 1e-10:
            new_x, new_z = self._round_2d(x, z, r, h)
            return np.array([new_x, y, new_z])
        else:
            new_x, new_y = self._round_2d(x, y, r, h)
            return np.array([new_x, new_y, z])

    def _round_2d(self, u, v, r, h):
        """Round in 2D plane."""
        edge_h = h - r

        new_u, new_v = u, v

        # Check if in corner region
        in_corner_u = abs(u) > edge_h
        in_corner_v = abs(v) > edge_h

        if in_corner_u and in_corner_v:
            # Corner region - project onto circle
            su = np.sign(u)
            sv = np.sign(v)

            # Local coords relative to corner center
            lu = abs(u) - edge_h
            lv = abs(v) - edge_h

            # Project onto circle of radius r
            dist = np.sqrt(lu**2 + lv**2)
            if dist > 0:
                new_u = su * (edge_h + r * lu / dist)
                new_v = sv * (edge_h + r * lv / dist)
        elif in_corner_u:
            # Edge in u direction only
            su = np.sign(u)
            new_u = su * h  # Keep at boundary
        elif in_corner_v:
            sv = np.sign(v)
            new_v = sv * h

        return new_u, new_v

    def _round_corner_vertex(self, v, r, h):
        """Round a vertex at an edge or corner of the cube."""
        x, y, z = v
        edge_h = h - r

        # Determine which edges/corners
        sx = np.sign(x) if abs(x) > edge_h else 0
        sy = np.sign(y) if abs(y) > edge_h else 0
        sz = np.sign(z) if abs(z) > edge_h else 0

        if sx != 0 and sy != 0 and sz != 0:
            # True corner - project onto sphere
            cx = sx * edge_h
            cy = sy * edge_h
            cz = sz * edge_h

            dx = x - cx
            dy = y - cy
            dz = z - cz

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 0:
                return np.array([
                    cx + r * dx / dist,
                    cy + r * dy / dist,
                    cz + r * dz / dist
                ])
        elif sx != 0 and sy != 0:
            # Edge along z
            new_x, new_y = self._round_2d(x, y, r, h)
            return np.array([new_x, new_y, z])
        elif sx != 0 and sz != 0:
            new_x, new_z = self._round_2d(x, z, r, h)
            return np.array([new_x, y, new_z])
        elif sy != 0 and sz != 0:
            new_y, new_z = self._round_2d(y, z, r, h)
            return np.array([x, new_y, new_z])

        return v

    def _merge_vertices(self, vertices, faces, tol=1e-6):
        """
        Merge duplicate vertices and update face indices.

        This is important because faces share vertices at edges.
        """
        n = len(vertices)

        # Find unique vertices
        unique_verts = []
        index_map = np.zeros(n, dtype=int)

        for i, v in enumerate(vertices):
            found = False
            for j, uv in enumerate(unique_verts):
                if np.linalg.norm(v - uv) < tol:
                    index_map[i] = j + 1  # 1-indexed
                    found = True
                    break
            if not found:
                unique_verts.append(v)
                index_map[i] = len(unique_verts)  # 1-indexed

        # Update face indices
        new_faces = faces.copy()
        for i in range(len(faces)):
            for j in range(3):  # Only first 3 columns are indices
                old_idx = int(faces[i, j])
                new_faces[i, j] = index_map[old_idx - 1]  # Convert to 0-indexed, then back

        return np.array(unique_verts), new_faces

    def save_to_mat(self, vertices, faces, filepath):
        """Save mesh to .mat file for MATLAB."""
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required for saving mesh to .mat file")

        sio.savemat(
            str(filepath),
            {
                'vertices': vertices,
                'faces': faces
            },
            do_compression=True
        )

        if self.verbose:
            print(f"  Saved adaptive mesh to {filepath}")

    def generate_proper_rounded(self, densities):
        """
        Generate proper rounded cube mesh with globally shared corners and edges.

        Unlike the per-face approach, this generates:
        - 8 corners ONCE (each 1/8 sphere, shared by 3 faces)
        - 12 edges ONCE (each quarter cylinder, shared by 2 faces)
        - 6 face centers (flat regions with edge-matching boundaries)

        This avoids the overlap problem at cube corners.

        Args:
            densities: dict with keys '+x', '-x', '+y', '-y', '+z', '-z'

        Returns:
            vertices: (N, 3) array of vertex coordinates
            faces: (M, 4) array of face indices (1-indexed, 4th column is NaN)
        """
        h = self.half_size
        r = self.r
        inner = h - r

        if r <= 0:
            return self.generate(densities)

        # Angular divisions for curved regions
        n_arc = max(3, int(self.rounding * 8))

        # Edge info: (axis along edge, sign1, sign2, face1, face2)
        # face1 connects at angle=0, face2 connects at angle=π/2
        edge_info = {
            'x_pp': (0, +1, +1, '+y', '+z'),
            'x_pm': (0, +1, -1, '+y', '-z'),
            'x_mp': (0, -1, +1, '-y', '+z'),
            'x_mm': (0, -1, -1, '-y', '-z'),
            'y_pp': (1, +1, +1, '+x', '+z'),
            'y_pm': (1, +1, -1, '+x', '-z'),
            'y_mp': (1, -1, +1, '-x', '+z'),
            'y_mm': (1, -1, -1, '-x', '-z'),
            'z_pp': (2, +1, +1, '+x', '+y'),
            'z_pm': (2, +1, -1, '+x', '-y'),
            'z_mp': (2, -1, +1, '-x', '+y'),
            'z_mm': (2, -1, -1, '-x', '-y'),
        }

        corner_info = {
            'ppp': (+1, +1, +1, ('+x', '+y', '+z')),
            'ppm': (+1, +1, -1, ('+x', '+y', '-z')),
            'pmp': (+1, -1, +1, ('+x', '-y', '+z')),
            'pmm': (+1, -1, -1, ('+x', '-y', '-z')),
            'mpp': (-1, +1, +1, ('-x', '+y', '+z')),
            'mpm': (-1, +1, -1, ('-x', '+y', '-z')),
            'mmp': (-1, -1, +1, ('-x', '-y', '+z')),
            'mmm': (-1, -1, -1, ('-x', '-y', '-z')),
        }

        # Calculate edge densities (max of adjacent faces)
        edge_densities = {}
        for edge_name, (_, _, _, f1, f2) in edge_info.items():
            edge_densities[edge_name] = max(densities.get(f1, 12), densities.get(f2, 12))

        # Determine which edges touch each face and their densities
        # Each face has 4 boundaries that connect to 4 edges
        # Face boundary names: bottom (ax2=-inner), top (ax2=+inner),
        #                      left (ax1=-inner), right (ax1=+inner)
        face_edge_map = {
            '+x': {  # axis=0, ax1=y, ax2=z
                'bottom': ('y_pm', 0),  # z=-inner, edge at angle=0
                'top': ('y_pp', 0),     # z=+inner, edge at angle=0
                'left': ('z_pm', 0),    # y=-inner, edge at angle=0
                'right': ('z_pp', 0),   # y=+inner, edge at angle=0
            },
            '-x': {  # axis=0, ax1=y, ax2=z
                'bottom': ('y_mm', 0),  # z=-inner
                'top': ('y_mp', 0),     # z=+inner
                'left': ('z_mm', 0),    # y=-inner
                'right': ('z_mp', 0),   # y=+inner
            },
            '+y': {  # axis=1, ax1=x, ax2=z
                'bottom': ('x_pm', 0),  # z=-inner
                'top': ('x_pp', 0),     # z=+inner
                'left': ('z_mp', 1),    # x=-inner, edge at angle=π/2
                'right': ('z_pp', 1),   # x=+inner
            },
            '-y': {  # axis=1, ax1=x, ax2=z
                'bottom': ('x_mm', 0),  # z=-inner
                'top': ('x_mp', 0),     # z=+inner
                'left': ('z_mm', 1),    # x=-inner
                'right': ('z_pm', 1),   # x=+inner
            },
            '+z': {  # axis=2, ax1=x, ax2=y
                'bottom': ('x_mp', 1),  # y=-inner, edge at angle=π/2
                'top': ('x_pp', 1),     # y=+inner
                'left': ('y_mp', 1),    # x=-inner
                'right': ('y_pp', 1),   # x=+inner
            },
            '-z': {  # axis=2, ax1=x, ax2=y
                'bottom': ('x_mm', 1),  # y=-inner
                'top': ('x_pm', 1),     # y=+inner
                'left': ('y_mm', 1),    # x=-inner
                'right': ('y_pm', 1),   # x=+inner
            },
        }

        # Get boundary densities for each face (from adjacent edges)
        face_boundary_densities = {}
        for face_name in ['+x', '-x', '+y', '-y', '+z', '-z']:
            face_boundary_densities[face_name] = {
                side: edge_densities[edge_name]
                for side, (edge_name, _) in face_edge_map[face_name].items()
            }

        all_vertices = []
        all_faces = []
        offset = 0

        # ============ 1. Generate 8 corners (1/8 spheres) ============
        corner_data = {}
        for corner_name, (sx, sy, sz, _) in corner_info.items():
            verts, faces = self._gen_global_corner(sx, sy, sz, n_arc, h, r, inner)
            corner_data[corner_name] = {'offset': offset, 'n_verts': len(verts)}
            all_vertices.extend(verts)
            for f in faces:
                all_faces.append([f[0] + offset, f[1] + offset, f[2] + offset, np.nan])
            offset += len(verts)

        # ============ 2. Generate 12 edges (quarter cylinders) ============
        edge_data = {}
        for edge_name, (axis, s1, s2, f1, f2) in edge_info.items():
            n = edge_densities[edge_name]
            verts, faces = self._gen_global_edge(axis, s1, s2, n, n_arc, h, r, inner)
            edge_data[edge_name] = {'offset': offset, 'n_verts': len(verts), 'n': n}
            all_vertices.extend(verts)
            for f in faces:
                all_faces.append([f[0] + offset, f[1] + offset, f[2] + offset, np.nan])
            offset += len(verts)

        # ============ 3. Generate 6 face centers with edge-matching boundaries ============
        face_data = {}
        for face_name in ['+x', '-x', '+y', '-y', '+z', '-z']:
            n = densities.get(face_name, 12)
            axis = {'+x': 0, '-x': 0, '+y': 1, '-y': 1, '+z': 2, '-z': 2}[face_name]
            sign = +1 if face_name[0] == '+' else -1
            boundary_n = face_boundary_densities[face_name]

            verts, faces = self._gen_face_center_with_transitions(
                axis, sign, n, boundary_n, h, inner
            )
            face_data[face_name] = {'offset': offset, 'n_verts': len(verts), 'n': n}
            all_vertices.extend(verts)
            for f in faces:
                all_faces.append([f[0] + offset, f[1] + offset, f[2] + offset, np.nan])
            offset += len(verts)

        # Convert to arrays
        vertices = np.array(all_vertices)
        faces = np.array(all_faces)

        # Merge duplicate vertices
        vertices, faces = self._merge_vertices(vertices, faces)

        # Remove degenerate triangles
        valid_mask = []
        for face in faces:
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            valid_mask.append((v1 != v2) and (v2 != v3) and (v1 != v3))

        n_degenerate = sum(1 for v in valid_mask if not v)
        if n_degenerate > 0:
            faces = faces[valid_mask]
            if self.verbose:
                print(f"  Removed {n_degenerate} degenerate triangles")

        if self.verbose:
            print(f"  Proper rounded mesh: {len(vertices)} vertices, {len(faces)} faces")

        return vertices, faces

    def _gen_face_center_with_transitions(self, axis, sign, n, boundary_n, h, inner):
        """
        Generate face center with edge-matching boundaries and internal transitions.

        The face has:
        - Internal grid at density n
        - Boundary vertices at edge densities (may differ per side)
        - Transition triangles connecting boundaries to internal grid

        Args:
            axis: 0, 1, or 2 (face normal direction)
            sign: +1 or -1
            n: internal grid density
            boundary_n: dict with 'bottom', 'top', 'left', 'right' densities
            h, inner: geometry parameters
        """
        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1

        # Check if any boundary has different density
        needs_transition = any(boundary_n[side] != n for side in boundary_n)

        if not needs_transition:
            # Simple case: all boundaries match internal density
            return self._gen_global_face_center(axis, sign, n, h, inner)

        # Generate with boundary transitions
        all_verts = []
        all_faces = []

        # 1. Generate internal grid (slightly smaller to leave room for boundary)
        # We use the internal density for an inner region, then add boundary vertices
        internal_coords = np.linspace(-inner, inner, n + 1)

        # Generate internal vertices
        for i in range(n + 1):
            for j in range(n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h
                v[ax1] = internal_coords[i]
                v[ax2] = internal_coords[j]
                all_verts.append(v)

        # Generate internal faces
        for i in range(n):
            for j in range(n):
                v00 = i * (n + 1) + j
                v10 = (i + 1) * (n + 1) + j
                v01 = i * (n + 1) + (j + 1)
                v11 = (i + 1) * (n + 1) + (j + 1)

                if sign > 0:
                    all_faces.append([v00 + 1, v10 + 1, v11 + 1])
                    all_faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    all_faces.append([v00 + 1, v11 + 1, v10 + 1])
                    all_faces.append([v00 + 1, v01 + 1, v11 + 1])

        # Internal grid boundaries (vertex indices, 1-indexed)
        # i indexes ax1, j indexes ax2
        internal_boundary = {
            'bottom': [i * (n + 1) + 1 for i in range(n + 1)],  # j=0
            'top': [i * (n + 1) + (n + 1) for i in range(n + 1)],  # j=n
            'left': [1 + j for j in range(n + 1)],  # i=0
            'right': [n * (n + 1) + 1 + j for j in range(n + 1)],  # i=n
        }

        # 2. For each boundary that needs different density, add boundary vertices
        # and transition triangles
        offset = len(all_verts)

        for side in ['bottom', 'top', 'left', 'right']:
            edge_n = boundary_n[side]
            if edge_n == n:
                continue  # No transition needed

            # Generate boundary vertices at edge density
            edge_coords = np.linspace(-inner, inner, edge_n + 1)
            boundary_verts = []

            for k in range(edge_n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h

                if side == 'bottom':
                    v[ax1] = edge_coords[k]
                    v[ax2] = -inner
                elif side == 'top':
                    v[ax1] = edge_coords[k]
                    v[ax2] = inner
                elif side == 'left':
                    v[ax1] = -inner
                    v[ax2] = edge_coords[k]
                else:  # right
                    v[ax1] = inner
                    v[ax2] = edge_coords[k]

                boundary_verts.append(v)
                all_verts.append(v)

            # Boundary vertex indices (1-indexed)
            boundary_idx = [offset + k + 1 for k in range(edge_n + 1)]
            offset += edge_n + 1

            # Internal edge vertices
            internal_idx = internal_boundary[side]

            # Reverse one of the arrays if needed for proper matching
            if side in ['top', 'left']:
                # These edges run in opposite direction
                pass  # Keep as is after testing

            # Create transition triangles
            trans = self._create_transition_strip(
                internal_idx, boundary_idx, sign, side
            )
            all_faces.extend(trans)

        return all_verts, all_faces

    def _create_transition_strip(self, idx1, idx2, sign, side):
        """
        Create transition triangles between two vertex sequences.

        idx1: internal boundary vertex indices
        idx2: edge boundary vertex indices
        sign: face normal direction
        side: 'bottom', 'top', 'left', or 'right'
        """
        faces = []
        n1, n2 = len(idx1), len(idx2)

        # Determine winding based on side and sign
        # For proper outward normals, we need correct vertex ordering
        if side in ['bottom', 'left']:
            # idx2 is "outside" (at boundary)
            outer, inner = idx2, idx1
            flip = (sign < 0)
        else:
            outer, inner = idx2, idx1
            flip = (sign > 0)

        # March through both sequences creating triangles
        i, j = 0, 0
        while i < len(inner) - 1 or j < len(outer) - 1:
            if i >= len(inner) - 1:
                # Only advance outer
                if flip:
                    faces.append([inner[-1], outer[j+1], outer[j]])
                else:
                    faces.append([inner[-1], outer[j], outer[j+1]])
                j += 1
            elif j >= len(outer) - 1:
                # Only advance inner
                if flip:
                    faces.append([inner[i], outer[-1], inner[i+1]])
                else:
                    faces.append([inner[i], inner[i+1], outer[-1]])
                i += 1
            else:
                # Advance based on parametric position
                t1 = i / max(len(inner) - 1, 1)
                t2 = j / max(len(outer) - 1, 1)

                if t1 <= t2:
                    if flip:
                        faces.append([inner[i], outer[j], inner[i+1]])
                    else:
                        faces.append([inner[i], inner[i+1], outer[j]])
                    i += 1
                else:
                    if flip:
                        faces.append([inner[i], outer[j+1], outer[j]])
                    else:
                        faces.append([inner[i], outer[j], outer[j+1]])
                    j += 1

        return faces

    def _gen_global_corner(self, sx, sy, sz, n_arc, h, r, inner):
        """Generate a 1/8 sphere at cube corner (sx*h, sy*h, sz*h)."""
        # Sphere center at (sx*(h-r), sy*(h-r), sz*(h-r))
        cx, cy, cz = sx * (h - r), sy * (h - r), sz * (h - r)

        vertices = []
        # Parametrize the 1/8 sphere using two angles
        # theta: 0 to pi/2 (from z-axis toward xy-plane)
        # phi: angle in xy-plane covering the octant
        for i in range(n_arc + 1):
            theta = i * (np.pi / 2) / n_arc
            for j in range(n_arc + 1):
                phi = j * (np.pi / 2) / n_arc

                # Local spherical coordinates
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)

                # Global position
                x = cx + sx * dx
                y = cy + sy * dy
                z = cz + sz * dz
                vertices.append([x, y, z])

        # Generate faces
        faces = []
        for i in range(n_arc):
            for j in range(n_arc):
                v00 = i * (n_arc + 1) + j
                v10 = (i + 1) * (n_arc + 1) + j
                v01 = i * (n_arc + 1) + (j + 1)
                v11 = (i + 1) * (n_arc + 1) + (j + 1)

                # Winding order for outward normal
                if sx * sy * sz > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        return vertices, faces

    def _gen_global_edge(self, axis, s1, s2, n, n_arc, h, r, inner):
        """
        Generate a quarter-cylinder edge.

        axis: 0=x, 1=y, 2=z (direction of edge)
        s1, s2: signs for the other two axes
        n: number of divisions along edge
        """
        # Determine the two perpendicular axes
        if axis == 0:  # edge along x
            ax1, ax2 = 1, 2
        elif axis == 1:  # edge along y
            ax1, ax2 = 0, 2
        else:  # edge along z
            ax1, ax2 = 0, 1

        # Cylinder center line: at (ax1_pos, ax2_pos) = (s1*(h-r), s2*(h-r))
        # Edge runs from -inner to +inner along 'axis'
        edge_coords = np.linspace(-inner, inner, n + 1)
        arc_angles = np.linspace(0, np.pi/2, n_arc + 1)

        vertices = []
        for i in range(n + 1):
            pos = edge_coords[i]  # position along edge
            for j in range(n_arc + 1):
                angle = arc_angles[j]

                # Cylinder surface: curves in ax1-ax2 plane
                # At angle=0: on ax1 side, at angle=pi/2: on ax2 side
                ax1_offset = r * np.cos(angle)
                ax2_offset = r * np.sin(angle)

                v = [0, 0, 0]
                v[axis] = pos
                v[ax1] = s1 * (h - r + ax1_offset)
                v[ax2] = s2 * (h - r + ax2_offset)
                vertices.append(v)

        # Generate faces
        faces = []
        for i in range(n):
            for j in range(n_arc):
                v00 = i * (n_arc + 1) + j
                v10 = (i + 1) * (n_arc + 1) + j
                v01 = i * (n_arc + 1) + (j + 1)
                v11 = (i + 1) * (n_arc + 1) + (j + 1)

                # Winding for outward normal
                if s1 * s2 > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        return vertices, faces

    def _gen_global_face_center(self, axis, sign, n, h, inner):
        """Generate flat center region of a face."""
        coords = np.linspace(-inner, inner, n + 1)

        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1

        vertices = []
        for i in range(n + 1):
            for j in range(n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h
                v[ax1] = coords[i]
                v[ax2] = coords[j]
                vertices.append(v)

        faces = []
        for i in range(n):
            for j in range(n):
                v00 = i * (n + 1) + j
                v10 = (i + 1) * (n + 1) + j
                v01 = i * (n + 1) + (j + 1)
                v11 = (i + 1) * (n + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        return vertices, faces

    # Keep old methods for backwards compatibility (marked as deprecated)
    def _generate_face_proper_rounded(self, axis, sign, face_n, edge_d, corner_d,
                                       n_arc, h, r, inner):
        """
        Generate a single face with proper rounded edges and corners.

        The face is divided into 9 regions:
        - 1 flat center
        - 4 cylindrical edge strips
        - 4 spherical corner patches

        Args:
            axis: 0=x, 1=y, 2=z (normal axis)
            sign: +1 or -1 (direction of normal)
            face_n: density for flat center
            edge_d: dict with 'bottom', 'right', 'top', 'left' densities
            corner_d: dict with 'bl', 'br', 'tr', 'tl' densities
            n_arc: number of angular divisions for curved regions
            h, r, inner: geometry parameters

        Returns:
            vertices, faces arrays
        """
        # Determine the two axes that span this face
        if axis == 0:  # x-face, spans y and z
            ax1, ax2 = 1, 2
        elif axis == 1:  # y-face, spans x and z
            ax1, ax2 = 0, 2
        else:  # z-face, spans x and y
            ax1, ax2 = 0, 1

        all_verts = []
        all_faces = []
        vert_offset = 0

        # ========== 1. Flat center region ==========
        center_verts, center_faces, center_boundary = self._gen_flat_center(
            axis, sign, ax1, ax2, face_n, h, inner
        )
        all_verts.extend(center_verts)
        for f in center_faces:
            all_faces.append([f[0] + vert_offset, f[1] + vert_offset,
                             f[2] + vert_offset, np.nan])
        center_offset = vert_offset
        vert_offset += len(center_verts)

        # ========== 2. Edge strips (cylindrical) ==========
        edge_strips = {}
        for edge_name in ['bottom', 'right', 'top', 'left']:
            edge_n = edge_d[edge_name]
            verts, faces, boundary = self._gen_edge_strip(
                axis, sign, ax1, ax2, edge_name, edge_n, n_arc, h, r, inner
            )
            edge_strips[edge_name] = {
                'verts': verts,
                'faces': faces,
                'boundary': boundary,
                'offset': vert_offset
            }
            all_verts.extend(verts)
            for f in faces:
                all_faces.append([f[0] + vert_offset, f[1] + vert_offset,
                                 f[2] + vert_offset, np.nan])
            vert_offset += len(verts)

        # ========== 3. Corner patches (spherical) ==========
        corner_patches = {}
        for corner_name in ['bl', 'br', 'tr', 'tl']:
            corner_n = corner_d[corner_name]
            verts, faces = self._gen_corner_patch(
                axis, sign, ax1, ax2, corner_name, corner_n, n_arc, h, r, inner
            )
            corner_patches[corner_name] = {
                'verts': verts,
                'faces': faces,
                'offset': vert_offset
            }
            all_verts.extend(verts)
            for f in faces:
                all_faces.append([f[0] + vert_offset, f[1] + vert_offset,
                                 f[2] + vert_offset, np.nan])
            vert_offset += len(verts)

        # ========== 4. Transition triangles ==========
        # Connect center to edge strips
        trans_faces = self._gen_center_edge_transitions(
            center_boundary, center_offset, edge_strips, face_n, edge_d, sign
        )
        all_faces.extend(trans_faces)

        # Connect edge strips to corner patches
        corner_trans = self._gen_edge_corner_transitions(
            edge_strips, corner_patches, n_arc, sign
        )
        all_faces.extend(corner_trans)

        vertices = np.array(all_verts)
        faces = np.array(all_faces)

        return vertices, faces

    def _gen_flat_center(self, axis, sign, ax1, ax2, n, h, inner):
        """Generate flat center region of a face."""
        coords = np.linspace(-inner, inner, n + 1)

        vertices = []
        for i in range(n + 1):
            for j in range(n + 1):
                v = [0, 0, 0]
                v[axis] = sign * h
                v[ax1] = coords[i]
                v[ax2] = coords[j]
                vertices.append(v)

        faces = []
        for i in range(n):
            for j in range(n):
                v00 = i * (n + 1) + j
                v10 = (i + 1) * (n + 1) + j
                v01 = i * (n + 1) + (j + 1)
                v11 = (i + 1) * (n + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        # Boundary indices: edges of center region
        # Grid layout: i indexes ax1, j indexes ax2
        # i=0 → ax1=-inner (left edge in ax1 direction)
        # i=n → ax1=+inner (right edge in ax1 direction)
        # j=0 → ax2=-inner (bottom edge in ax2 direction)
        # j=n → ax2=+inner (top edge in ax2 direction)
        boundary = {
            'bottom': [i * (n + 1) for i in range(n + 1)],  # j=0, ax2=-inner
            'top': [i * (n + 1) + n for i in range(n + 1)],  # j=n, ax2=+inner
            'left': [j for j in range(n + 1)],  # i=0, ax1=-inner
            'right': [n * (n + 1) + j for j in range(n + 1)],  # i=n, ax1=+inner
        }

        return vertices, faces, boundary

    def _gen_edge_strip(self, axis, sign, ax1, ax2, edge_name, n, n_arc, h, r, inner):
        """
        Generate cylindrical edge strip.

        The strip is a quarter-cylinder connecting the flat center to the cube outer edge.

        For bottom edge of +x face:
        - Cylinder axis along ax1 (y-direction)
        - Cylinder center at (h-r, y, -inner)
        - At angle=0: connects to flat center at (h, y, -inner)
        - At angle=π/2: connects to corner at (h-r, y, -h)
        """
        # Edge coords: along the edge direction (ax1 for bottom/top, ax2 for left/right)
        # Arc angles: 0 = inner (connects to flat center), π/2 = outer (connects to corner)
        arc_angles = np.linspace(0, np.pi/2, n_arc + 1)

        # Determine edge orientation
        if edge_name == 'bottom':
            edge_coords = np.linspace(-inner, inner, n + 1)  # ax1 varies
            ax2_sign = -1  # ax2 goes toward negative
        elif edge_name == 'top':
            edge_coords = np.linspace(inner, -inner, n + 1)  # ax1 varies, reversed
            ax2_sign = +1  # ax2 goes toward positive
        elif edge_name == 'left':
            edge_coords = np.linspace(inner, -inner, n + 1)  # ax2 varies
            ax2_sign = None  # use ax1 instead
            ax1_sign = -1  # ax1 goes toward negative
        else:  # right
            edge_coords = np.linspace(-inner, inner, n + 1)  # ax2 varies
            ax2_sign = None
            ax1_sign = +1  # ax1 goes toward positive

        vertices = []
        for i in range(n + 1):
            for j in range(n_arc + 1):
                angle = arc_angles[j]
                edge_pos = edge_coords[i]

                # Cylinder geometry:
                # - At angle=0: on the flat face (x=h, curved_ax at ±inner)
                # - At angle=π/2: at cube outer (x=h-r, curved_ax at ±h)
                if edge_name in ['bottom', 'top']:
                    # Cylinder axis along ax1 (y), curves in ax2 (z) and axis (x)
                    ax1_val = edge_pos
                    ax2_val = ax2_sign * (inner + r * np.sin(angle))
                    normal_val = sign * (h - r * (1 - np.cos(angle)))
                else:
                    # Cylinder axis along ax2 (z for +x face), curves in ax1 (y) and axis (x)
                    ax2_val = edge_pos
                    ax1_val = ax1_sign * (inner + r * np.sin(angle))
                    normal_val = sign * (h - r * (1 - np.cos(angle)))

                v = [0, 0, 0]
                v[axis] = normal_val
                v[ax1] = ax1_val
                v[ax2] = ax2_val
                vertices.append(v)

        faces = []
        for i in range(n):
            for j in range(n_arc):
                v00 = i * (n_arc + 1) + j
                v10 = (i + 1) * (n_arc + 1) + j
                v01 = i * (n_arc + 1) + (j + 1)
                v11 = (i + 1) * (n_arc + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        # Boundary indices:
        # - inner (j=0): connects to flat center
        # - outer (j=n_arc): connects to corner
        # - start (i=0): connects to corner patch
        # - end (i=n): connects to corner patch
        boundary = {
            'inner': [i * (n_arc + 1) for i in range(n + 1)],  # j=0
            'outer': [i * (n_arc + 1) + n_arc for i in range(n + 1)],  # j=n_arc
            'start': [j for j in range(n_arc + 1)],  # i=0
            'end': [n * (n_arc + 1) + j for j in range(n_arc + 1)],  # i=n
        }

        return vertices, faces, boundary

    def _gen_corner_patch(self, axis, sign, ax1, ax2, corner_name, n, n_arc, h, r, inner):
        """
        Generate spherical corner patch.

        The patch is a 1/8 sphere section at the cube corner.
        """
        # Determine corner position
        if corner_name == 'bl':  # bottom-left
            ax1_sign, ax2_sign = -1, -1
        elif corner_name == 'br':  # bottom-right
            ax1_sign, ax2_sign = +1, -1
        elif corner_name == 'tr':  # top-right
            ax1_sign, ax2_sign = +1, +1
        else:  # tl, top-left
            ax1_sign, ax2_sign = -1, +1

        # Sphere center
        center = [0, 0, 0]
        center[axis] = sign * (h - r)
        center[ax1] = ax1_sign * inner
        center[ax2] = ax2_sign * inner

        # Generate vertices on spherical surface
        # Use spherical coordinates: theta (0 to pi/2), phi (0 to pi/2)
        vertices = []

        # Angular divisions
        theta_vals = np.linspace(0, np.pi/2, n_arc + 1)  # from pole to equator
        phi_vals = np.linspace(0, np.pi/2, n_arc + 1)  # around the corner

        for i in range(n_arc + 1):
            theta = theta_vals[i]
            for j in range(n_arc + 1):
                phi = phi_vals[j]

                # Spherical to Cartesian (local coordinates)
                # The "pole" points toward the face normal
                local_normal = r * np.cos(theta)
                local_ax1 = r * np.sin(theta) * np.cos(phi)
                local_ax2 = r * np.sin(theta) * np.sin(phi)

                v = [0, 0, 0]
                v[axis] = center[axis] + sign * local_normal
                v[ax1] = center[ax1] + ax1_sign * local_ax1
                v[ax2] = center[ax2] + ax2_sign * local_ax2
                vertices.append(v)

        faces = []
        for i in range(n_arc):
            for j in range(n_arc):
                v00 = i * (n_arc + 1) + j
                v10 = (i + 1) * (n_arc + 1) + j
                v01 = i * (n_arc + 1) + (j + 1)
                v11 = (i + 1) * (n_arc + 1) + (j + 1)

                if sign > 0:
                    faces.append([v00 + 1, v10 + 1, v11 + 1])
                    faces.append([v00 + 1, v11 + 1, v01 + 1])
                else:
                    faces.append([v00 + 1, v11 + 1, v10 + 1])
                    faces.append([v00 + 1, v01 + 1, v11 + 1])

        return vertices, faces

    def _gen_center_edge_transitions(self, center_boundary, center_offset,
                                      edge_strips, face_n, edge_d, sign):
        """Generate triangles connecting center to edge strips.

        Note: When center and edge have the same density, their boundary vertices
        are at identical positions and will be merged. In this case, no transition
        triangles are needed - the faces from center and edge will naturally share
        the merged vertices.

        Transition triangles are only needed when densities differ.
        """
        faces = []

        for edge_name in ['bottom', 'right', 'top', 'left']:
            center_n = len(center_boundary[edge_name])  # = face_n + 1
            edge_n = len(edge_strips[edge_name]['boundary']['inner'])  # = edge_density + 1

            # If densities match, vertices will merge - no transition needed
            if center_n == edge_n:
                continue

            # Different densities - need transition triangles
            center_indices = center_boundary[edge_name]
            edge_info = edge_strips[edge_name]
            edge_inner = edge_info['boundary']['inner']
            edge_offset = edge_info['offset']

            c_idx = [center_offset + i + 1 for i in center_indices]
            e_idx = [edge_offset + i + 1 for i in edge_inner]

            if edge_name in ['top', 'left']:
                e_idx = e_idx[::-1]

            trans = self._create_transition_triangles(c_idx, e_idx, sign)
            faces.extend(trans)

        return faces

    def _gen_edge_corner_transitions(self, edge_strips, corner_patches, n_arc, sign):
        """Generate triangles connecting edge strips to corner patches.

        Note: Edge strip boundaries ('start', 'end') and corner patch boundaries
        both have n_arc+1 vertices. When they're at the same positions, they will
        merge automatically - no transition triangles needed.
        """
        # Edge 'start' and 'end' boundaries have n_arc+1 vertices
        # Corner boundaries also have n_arc+1 vertices
        # If they're at the same positions (which they should be), they merge.
        # No transition triangles needed - just return empty list.
        return []

    def _create_transition_triangles(self, idx1, idx2, sign):
        """Create triangles connecting two vertex sequences."""
        faces = []
        n1, n2 = len(idx1), len(idx2)

        i, j = 0, 0
        while i < n1 - 1 or j < n2 - 1:
            if i >= n1 - 1:
                # Only advance j
                if sign > 0:
                    faces.append([idx1[-1], idx2[j], idx2[j + 1], np.nan])
                else:
                    faces.append([idx1[-1], idx2[j + 1], idx2[j], np.nan])
                j += 1
            elif j >= n2 - 1:
                # Only advance i
                if sign > 0:
                    faces.append([idx1[i], idx1[i + 1], idx2[-1], np.nan])
                else:
                    faces.append([idx1[i], idx2[-1], idx1[i + 1], np.nan])
                i += 1
            else:
                # Advance based on ratio
                r1 = i / max(n1 - 1, 1)
                r2 = j / max(n2 - 1, 1)

                if r1 <= r2:
                    if sign > 0:
                        faces.append([idx1[i], idx1[i + 1], idx2[j], np.nan])
                    else:
                        faces.append([idx1[i], idx2[j], idx1[i + 1], np.nan])
                    i += 1
                else:
                    if sign > 0:
                        faces.append([idx1[i], idx2[j], idx2[j + 1], np.nan])
                    else:
                        faces.append([idx1[i], idx2[j + 1], idx2[j], np.nan])
                    j += 1

        return faces


# ============================================================================
# Geometry Generator
# ============================================================================

class GeometryGenerator:
    """Generates geometry-related MATLAB code."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        self.nonlocal_gen = NonlocalGenerator(config, verbose)
    
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
            'core_shell_rod': self._core_shell_rod,
            'dimer_core_shell_cube': self._dimer_core_shell_cube,
            'advanced_dimer_cube': self._advanced_dimer_cube,
            'sphere_cluster_aggregate': self._sphere_cluster_aggregate,
            'from_shape': self._from_shape,
        }
        
        if self.structure not in structure_map:
            raise ValueError(f"Unknown structure type: {self.structure}")
        
        # Generate base geometry
        base_code = structure_map[self.structure]()

        # Apply mirror symmetry selection if enabled
        mirror_code = self._get_mirror_selection_code()
        if mirror_code:
            base_code = base_code + mirror_code

        # Apply nonlocal cover layers if enabled
        if self.nonlocal_gen.is_needed():
            base_code = self._apply_nonlocal_coverlayer(base_code)

        return base_code

    def _get_mirror_selection_code(self):
        """
        Generate MATLAB code for mirror symmetry selection.

        NOTE: This function now returns empty string because comparticlemirror
        handles mirror symmetry internally. Pre-selecting particles with select()
        before passing to comparticlemirror causes errors because:

        1. comparticlemirror expects FULL particles
        2. It internally uses flip() to create mirror copies
        3. It builds symtable based on the full geometry

        If we pre-select (e.g., y >= 0), comparticlemirror receives half-particles
        and cannot properly initialize pfull, causing "Dot indexing not supported"
        errors in the closed() method.

        The op.sym parameter tells comparticlemirror which symmetry to use,
        and it handles everything internally.
        """
        # Do NOT generate pre-selection code - comparticlemirror handles this internally
        return ""

    def _mesh_density_to_n_rod(self, mesh_density):
        """
        Convert single mesh_density value to [nphi, ntheta, nz] for trirod.
        
        MNPBEM's trirod default is [15, 20, 20] which gives ~900 vertices.
        This method maintains similar aspect ratios for different densities.
        
        Args:
            mesh_density (int): Target mesh density (similar to sphere)
            
        Returns:
            list: [nphi, ntheta, nz] for trirod
        """
        
        # Predefined mapping for common values
        MESH_MAPPING = {
            32:   [8,   10,  10],   # Very coarse (~240 vertices)
            60:   [10,  12,  12],   # Coarse (~360 vertices)
            144:  [15,  20,  20],   # Standard (~900 vertices) - MNPBEM default
            256:  [20,  25,  25],   # Medium (~1400 vertices)
            400:  [25,  30,  30],   # Fine (~2100 vertices)
            576:  [30,  35,  35],   # Very fine (~3150 vertices)
            900:  [35,  40,  40],   # Ultra fine (~4200 vertices)
        }
        
        # Return exact match if available
        if mesh_density in MESH_MAPPING:
            if self.verbose:
                print(f"    mesh_density {mesh_density} → {MESH_MAPPING[mesh_density]}")
            return MESH_MAPPING[mesh_density]
        
        # Find closest match (within 15%)
        closest = min(MESH_MAPPING.keys(), key=lambda x: abs(x - mesh_density))
        if abs(closest - mesh_density) < mesh_density * 0.15:
            if self.verbose:
                print(f"    mesh_density {mesh_density} rounded to {closest} → {MESH_MAPPING[closest]}")
            return MESH_MAPPING[closest]
        
        # Calculate for custom values
        # Total vertices ≈ nphi * (2*ntheta + nz)
        # Maintain ratio nphi:ntheta:nz = 15:20:20 = 3:4:4
        k = np.sqrt(mesh_density / 4.0)
        nphi = max(8, int(np.round(k)))
        ntheta = max(10, int(np.round(k * 4.0 / 3.0)))
        nz = max(10, int(np.round(k * 4.0 / 3.0)))
        
        result = [nphi, ntheta, nz]
        if self.verbose:
            approx_verts = nphi * (2 * ntheta + nz)
            print(f"    mesh_density {mesh_density} (custom) → {result} (~{approx_verts} vertices)")
        
        return result
    
    def _apply_nonlocal_coverlayer(self, base_geometry_code):
        """Apply nonlocal cover layer to existing geometry.

        IMPORTANT: Cover layer is applied ONLY to the outermost metal layer.
        If the outermost layer is not a metal, no cover layer is applied.
        """
        if not self.nonlocal_gen.is_needed():
            return base_geometry_code

        # Check if outermost layer is a metal
        materials = self.config.get('materials', [])
        metals = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al', 'copper', 'cu']

        outermost_metal = None
        outermost_idx = None
        for i in range(len(materials) - 1, -1, -1):
            mat_name = materials[i].lower() if isinstance(materials[i], str) else ''
            if any(metal in mat_name for metal in metals):
                outermost_metal = materials[i]
                outermost_idx = i
                break

        if outermost_metal is None:
            if self.verbose:
                print("  No outermost metal found - skipping cover layer application")
            return base_geometry_code

        is_applicable, warnings = self.nonlocal_gen.check_applicability()
        if warnings:
            warning_str = "\n".join([f"%   - {w}" for w in warnings])
            warning_code = f"""
%% [!] Nonlocal Warnings:
{warning_str}
"""
        else:
            warning_code = ""

        structure = self.config.get('structure', '')

        if 'sphere' in structure:
            cover_code = self._apply_coverlayer_sphere(outermost_idx, outermost_metal)
        elif 'cube' in structure:
            cover_code = self._apply_coverlayer_cube(outermost_idx, outermost_metal)
        else:
            if self.verbose:
                print(f"  ⚠ Warning: Nonlocal not fully implemented for structure '{structure}'")
            cover_code = self._apply_coverlayer_manual()

        combined_code = f"""
{base_geometry_code}

{warning_code}

%% Apply Nonlocal Cover Layers (ONLY to outermost metal: {outermost_metal})
fprintf('\\n=== Applying Nonlocal Cover Layers ===\\n');
fprintf('  Outermost metal: {outermost_metal}\\n');
{cover_code}
"""
        return combined_code
    
    def _apply_coverlayer_sphere(self, outermost_idx=None, outermost_metal=None):
        """Apply cover layer to sphere structures.

        For core-shell structures, only the outermost layer (shell) gets a cover layer.
        For single sphere, the sphere gets a cover layer.
        """
        d = self.nonlocal_gen.cover_thickness
        materials = self.config.get('materials', [])
        n_layers = len(materials)

        if n_layers == 1:
            # Single sphere - apply cover layer
            code = f"""
% Apply nonlocal cover layer to single sphere ({outermost_metal})
d_cover = {d};
particles_with_cover = {{}};

for i = 1:length(particles)
    p_inner = particles{{i}};
    p_outer = coverlayer.shift( p_inner, d_cover );
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;
    fprintf('  [OK] Particle %d: added %.3f nm cover layer\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total particles after cover layers: %d\\n', length(particles));
"""
        else:
            # Core-shell structure - only apply cover layer to outermost layer
            # particles list: [core1, shell1, ..., core2, shell2, ...] for dimer
            # or [core, shell, ...] for single
            code = f"""
% Apply nonlocal cover layer ONLY to outermost layer ({outermost_metal})
% Inner layers (core, inner shells) are NOT modified
d_cover = {d};
n_layers = {n_layers};  % layers per particle
particles_with_cover = {{}};

n_particles = length(particles) / n_layers;  % number of particles (1 for single, 2 for dimer)

for p_idx = 1:n_particles
    base_idx = (p_idx - 1) * n_layers;

    % Copy inner layers as-is (no cover layer)
    for layer_idx = 1:(n_layers-1)
        particles_with_cover{{end+1}} = particles{{base_idx + layer_idx}};
    end

    % Apply cover layer ONLY to outermost layer
    p_outer_layer = particles{{base_idx + n_layers}};  % outermost layer
    p_outer_boundary = coverlayer.shift( p_outer_layer, d_cover );
    particles_with_cover{{end+1}} = p_outer_boundary;  % new outer boundary
    particles_with_cover{{end+1}} = p_outer_layer;      % inner boundary (original shell)

    fprintf('  [OK] Particle %d: cover layer on outermost ({outermost_metal})\\n', p_idx);
end

particles = particles_with_cover;
fprintf('  Total boundaries after cover layers: %d\\n', length(particles));
"""
        return code
    
    def _apply_coverlayer_cube(self, outermost_idx=None, outermost_metal=None):
        """Apply cover layer to cube structures.

        IMPORTANT: Cover layer is applied ONLY to the outermost metal layer.
        Inner layers (core, inner shells) are NOT modified.
        """
        d = self.nonlocal_gen.cover_thickness
        structure = self.config.get('structure', '')
        materials = self.config.get('materials', [])
        n_layers = len(materials)

        if structure == 'advanced_dimer_cube':
            shell_layers = self.config.get('shell_layers', [])
            mesh = self.config.get('mesh_density', 12)

            # Get rounding for the outermost layer
            roundings = self.config.get('roundings', None)
            if roundings is None:
                rounding = self.config.get('rounding', 0.25)
                roundings = [rounding] * n_layers
            outermost_rounding = roundings[outermost_idx] if outermost_idx < len(roundings) else 0.25

            if n_layers == 1:
                # Single material (no shell) - apply cover to all cubes
                core_size = self.config.get('core_size', 30)
                code = f"""
% Apply nonlocal cover layers to advanced_dimer_cube (single material: {outermost_metal})
d_cover = {d};
particles_with_cover = {{}};

fprintf('  Applying cover layers to {outermost_metal}...\\n');

for i = 1:length(particles)
    p_outer = particles{{i}};  % Original cube (outer boundary)

    % Create inner boundary (smaller cube)
    size_inner = {core_size} - 2*d_cover;
    p_inner = tricube({mesh}, size_inner, 'e', {outermost_rounding});

    % Align centers
    center_outer = mean(p_outer.verts, 1);
    center_inner = mean(p_inner.verts, 1);
    p_inner = shift(p_inner, center_outer - center_inner);

    % Add: outer first, then inner
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;

    fprintf('    [OK] Particle %d: cover layer %.3f nm\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total boundaries: %d\\n', length(particles));
"""
                return code
            else:
                # Multi-layer (core-shell) - only apply cover to outermost layer
                code = f"""
% Apply nonlocal cover layer ONLY to outermost layer ({outermost_metal})
% Inner layers are NOT modified
d_cover = {d};
n_layers = {n_layers};  % layers per particle
mesh = {mesh};
outermost_rounding = {outermost_rounding};
particles_with_cover = {{}};

n_particles = length(particles) / n_layers;  % number of particles (2 for dimer)

for p_idx = 1:n_particles
    base_idx = (p_idx - 1) * n_layers;

    % Copy inner layers as-is (no cover layer)
    for layer_idx = 1:(n_layers-1)
        particles_with_cover{{end+1}} = particles{{base_idx + layer_idx}};
    end

    % Apply cover layer ONLY to outermost layer
    p_outer_layer = particles{{base_idx + n_layers}};  % outermost layer
    verts = p_outer_layer.verts;
    current_size = max(verts(:,1)) - min(verts(:,1));

    % Create inner boundary for the outermost layer
    size_inner = current_size - 2*d_cover;
    p_inner_boundary = tricube(mesh, size_inner, 'e', outermost_rounding);
    center_outer = mean(p_outer_layer.verts, 1);
    center_inner = mean(p_inner_boundary.verts, 1);
    p_inner_boundary = shift(p_inner_boundary, center_outer - center_inner);

    % Add: outer boundary first, then inner boundary
    particles_with_cover{{end+1}} = p_outer_layer;      % outer boundary
    particles_with_cover{{end+1}} = p_inner_boundary;   % inner boundary

    fprintf('    [OK] Particle %d: cover layer on {outermost_metal}\\n', p_idx);
end

particles = particles_with_cover;
fprintf('  Total boundaries: %d\\n', length(particles));
"""
                return code

        else:
            # Other cube structures (dimer_cube, cube, etc.)
            rounding = self.config.get('rounding', 0.25)
            mesh = self.config.get('mesh_density', 12)

            if n_layers == 1:
                # Single layer - apply cover to all cubes
                code = f"""
% Apply nonlocal cover layers to cubes ({outermost_metal})
d_cover = {d};
particles_with_cover = {{}};

for i = 1:length(particles)
    p_outer = particles{{i}};
    verts = p_outer.verts;
    size_current = max(verts(:,1)) - min(verts(:,1));

    % Create inner boundary
    size_inner = size_current - 2*d_cover;
    p_inner = tricube({mesh}, size_inner, 'e', {rounding});
    center_orig = mean(p_outer.verts, 1);
    center_new = mean(p_inner.verts, 1);
    p_inner = shift(p_inner, center_orig - center_new);

    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;

    fprintf('  [OK] Particle %d: added %.3f nm cover layer\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total particles after cover layers: %d\\n', length(particles));
"""
            else:
                # Core-shell - only apply cover to outermost layer
                code = f"""
% Apply nonlocal cover layer ONLY to outermost layer ({outermost_metal})
% Inner layers are NOT modified
d_cover = {d};
n_layers = {n_layers};
particles_with_cover = {{}};

n_particles = length(particles) / n_layers;

for p_idx = 1:n_particles
    base_idx = (p_idx - 1) * n_layers;

    % Copy inner layers as-is
    for layer_idx = 1:(n_layers-1)
        particles_with_cover{{end+1}} = particles{{base_idx + layer_idx}};
    end

    % Apply cover layer to outermost layer only
    p_outer_layer = particles{{base_idx + n_layers}};
    verts = p_outer_layer.verts;
    size_current = max(verts(:,1)) - min(verts(:,1));

    size_inner = size_current - 2*d_cover;
    p_inner = tricube({mesh}, size_inner, 'e', {rounding});
    center_orig = mean(p_outer_layer.verts, 1);
    center_new = mean(p_inner.verts, 1);
    p_inner = shift(p_inner, center_orig - center_new);

    particles_with_cover{{end+1}} = p_outer_layer;  % outer boundary
    particles_with_cover{{end+1}} = p_inner;        % inner boundary

    fprintf('  [OK] Particle %d: cover layer on outermost ({outermost_metal})\\n', p_idx);
end

particles = particles_with_cover;
fprintf('  Total boundaries: %d\\n', length(particles));
"""
            return code
    
    def _apply_coverlayer_manual(self):
        """Generic cover layer application."""
        d = self.nonlocal_gen.cover_thickness
        
        code = f"""
% Manual cover layer application
d_cover = {d};

fprintf('  [!] Manual cover layer mode\\n');
fprintf('  [!] Verify geometry visually before running full simulation!\\n');

if length(particles) > 0
    p1 = particles{{1}};
    p1_outer = coverlayer.shift( p1, d_cover );
    particles = {{p1_outer, p1}};
    fprintf('  [OK] Applied cover layer to first particle\\n');
end
"""
        return code
    
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
        """Generate code for rod/cylinder (horizontal).
        
        Mesh can be specified in two ways:
        1. mesh_density (auto-calculated)
        2. rod_mesh = [nphi, ntheta, nz] (manual)
        """
        diameter = self.config.get('diameter', 10)
        height = self.config.get('height', 50)
        
        # Check if user provided explicit mesh parameters
        if 'rod_mesh' in self.config:
            # User specifies [nphi, ntheta, nz] directly
            n = self.config['rod_mesh']
            nphi, ntheta, nz = n
            
            code = f"""
%% Geometry: Rod (horizontal along x-axis)
diameter = {diameter};
height = {height};

% User-specified mesh: [{nphi}, {ntheta}, {nz}]
p = trirod(diameter, height, [{nphi}, {ntheta}, {nz}]);
p = rot(p, 90, [0, 1, 0]);

particles = {{p}};
"""
        else:
            # Use mesh_density (existing behavior)
            mesh = self.config.get('mesh_density', 144)
            n = self._mesh_density_to_n_rod(mesh)
            
            code = f"""
%% Geometry: Rod (horizontal along x-axis)
diameter = {diameter};
height = {height};

% Auto-calculated mesh from mesh_density={mesh}: {n}
p = trirod(diameter, height, {n});
p = rot(p, 90, [0, 1, 0]);

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

    def _sphere_cluster_aggregate(self):
        """Generate compact sphere cluster (close-packed aggregate structure).

        Structures:
            N=1: Single sphere
            N=2: Dimer (horizontal)
            N=3: Triangle (2 bottom, 1 top)
            N=4: Center + 3 surrounding (hexagonal positions)
            N=5: Center + 4 surrounding
            N=6: Center + 5 surrounding
            N=7: Center + 6 surrounding (complete hexagon, close-packed)

        For N=4~7, spheres are arranged with one center sphere and surrounding
        spheres placed at 60° intervals (hexagonal pattern). At N=7, all 6
        surrounding positions are filled, creating a perfect close-packed structure.

        Gap parameter controls spacing between all contacting sphere pairs.
        """
        n_spheres = self.config.get('n_spheres', 1)
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', -0.1)
        mesh = self.config.get('mesh_density', 144)
        
        # Center-to-center spacing for contact
        spacing = diameter + gap

        # 60-degree triangle height
        dy_60deg = spacing * 0.866025404  # sin(60°) = sqrt(3)/2

        # Hexagonal surrounding positions (60° intervals, starting from +x direction)
        # Used for N=4~7: center + surrounding spheres
        hex_positions = []
        for i in range(6):
            angle = i * 60 * np.pi / 180  # 0°, 60°, 120°, 180°, 240°, 300°
            x = spacing * np.cos(angle)
            y = spacing * np.sin(angle)
            hex_positions.append((x, y))

        # Define xy positions for each cluster (z=0 for all, substrate contact handled separately)
        # Format: [(x, y), ...]
        # N=1,2,3: Original configurations
        # N=4~7: Center sphere + surrounding spheres in hexagonal positions
        cluster_positions = {
            1: [(0, 0)],

            2: [(-spacing/2, 0),
                (spacing/2, 0)],

            3: [(-spacing/2, 0),         # bottom-left
                (spacing/2, 0),          # bottom-right
                (0, dy_60deg)],          # top

            # N=4~7: Center (0,0) + hexagonal surrounding positions
            4: [(0, 0)] + hex_positions[0:3],  # center + 3 surrounding

            5: [(0, 0)] + hex_positions[0:4],  # center + 4 surrounding

            6: [(0, 0)] + hex_positions[0:5],  # center + 5 surrounding

            7: [(0, 0)] + hex_positions[0:6],  # center + 6 surrounding (complete hexagon)
        }
        
        if n_spheres not in cluster_positions:
            raise ValueError(f"n_spheres must be 1-7, got {n_spheres}")
        
        positions = cluster_positions[n_spheres]
        
        # Generate MATLAB code
        code = f"""
%% Geometry: Compact Sphere Cluster (Close-Packed Aggregate)
n_spheres = {n_spheres};
diameter = {diameter};
gap = {gap};  % negative = 0.1nm overlap (contact)
spacing = diameter + gap;  % {spacing:.3f} nm

fprintf('\\n=== Creating Compact Sphere Cluster ===\\n');
fprintf('  Number of spheres: %d\\n', n_spheres);
fprintf('  Diameter: %.2f nm\\n', diameter);
fprintf('  Gap: %.3f nm (%.1f nm overlap)\\n', gap, abs(gap));
fprintf('  Center-to-center spacing: %.3f nm\\n', spacing);
fprintf('  Structure type: ');

% Define positions for each sphere
positions = [
"""
        
        # Add position coordinates
        for i, (x, y) in enumerate(positions):
            if i < len(positions) - 1:
                code += f"    {x:.6f}, {y:.6f}, 0;  % Sphere {i+1}\n"
            else:
                code += f"    {x:.6f}, {y:.6f}, 0   % Sphere {i+1}\n"
        
        code += """];

% Determine structure name
switch n_spheres
    case 1
        fprintf('Single sphere\\n');
    case 2
        fprintf('Dimer\\n');
    case 3
        fprintf('Triangle\\n');
    case 4
        fprintf('Center + 3 surrounding\\n');
    case 5
        fprintf('Center + 4 surrounding\\n');
    case 6
        fprintf('Center + 5 surrounding\\n');
    case 7
        fprintf('Center + 6 surrounding (complete hexagon)\\n');
end

% Create particles
particles = {};
for i = 1:n_spheres
    % Create sphere
    p_sphere = trisphere(""" + f"{mesh}" + """, diameter);
    
    % Shift to position
    p_sphere = shift(p_sphere, positions(i, :));
    
    % Add to particle list
    particles{end+1} = p_sphere;
    
    fprintf('  Sphere %d: (%.2f, %.2f, 0) nm\\n', i, positions(i, 1), positions(i, 2));
end

% Calculate cluster bounds
x_coords = positions(:, 1);
y_coords = positions(:, 2);
x_min = min(x_coords) - diameter/2;
x_max = max(x_coords) + diameter/2;
y_min = min(y_coords) - diameter/2;
y_max = max(y_coords) + diameter/2;

fprintf('  Cluster bounds: x=[%.2f, %.2f], y=[%.2f, %.2f] nm\\n', ...
        x_min, x_max, y_min, y_max);
fprintf('  All spheres in XY plane (z=0)\\n');
fprintf('=================================\\n');
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

particles = {{p_core, p_shell}};
"""
        return code
    
    def _core_shell_cube(self):
        """Generate code for core-shell cube."""
        core_size = self.config.get('core_size')
        shell_thickness = self.config.get('shell_thickness')
        rounding = self.config.get('rounding')
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

particles = {{p_core, p_shell}};
"""
        return code

    def _core_shell_rod(self):
        """Generate code for core-shell rod with complete shell coverage.
        
        Mesh specification:
            Option 1: mesh_density (auto-calculated)
            Option 2: rod_mesh = [nphi, ntheta, nz] (manual override)
        """
        core_diameter = self.config.get('core_diameter', 15)
        shell_thickness = self.config.get('shell_thickness', 5)
        height = self.config.get('height', 80)
        
        # Check for manual mesh specification
        if 'rod_mesh' in self.config:
            n = self.config['rod_mesh']
            if len(n) != 3:
                raise ValueError(f"rod_mesh must have 3 values [nphi, ntheta, nz], got {n}")
        else:
            # Auto-calculate from mesh_density
            mesh = self.config.get('mesh_density', 144)
            n = self._mesh_density_to_n_rod(mesh)
        
        shell_diameter = core_diameter + 2 * shell_thickness
        shell_height = height
        core_height = height - 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Rod
core_diameter = {core_diameter};
shell_thickness = {shell_thickness};
shell_diameter = core_diameter + 2 * shell_thickness;
shell_height = {height};
core_height = shell_height - 2 * shell_thickness;

% Mesh: {n}
% Create rod particles (initially standing along z-axis)
p_core = trirod(core_diameter, core_height, {n}, 'triangles');
p_shell = trirod(shell_diameter, shell_height, {n}, 'triangles');

% Rotate 90 degrees to lie down along x-axis
p_core = rot(p_core, 90, [0, 1, 0]);
p_shell = rot(p_shell, 90, [0, 1, 0]);

particles = {{p_core, p_shell}};
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

% Particle 1 (Left)
core1 = tricube({mesh}, core_size, 'e', rounding_param);
core1 = shift(core1, [-shift_distance, 0, 0]);

shell1 = tricube({mesh}, shell_size, 'e', rounding_param);
shell1 = shift(shell1, [-shift_distance, 0, 0]);

% Particle 2 (Right)
core2 = tricube({mesh}, core_size, 'e', rounding_param);
core2 = shift(core2, [shift_distance, 0, 0]);

shell2 = tricube({mesh}, shell_size, 'e', rounding_param);
shell2 = shift(shell2, [shift_distance, 0, 0]);

particles = {{core1, shell1, core2, shell2}};
"""
        return code
    
    def _advanced_dimer_cube(self):
        """Generate advanced dimer cube with full control."""
        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        mesh = self.config.get('mesh_density', 12)

        if len(materials) != 1 + len(shell_layers):
            raise ValueError(
                f"materials length ({len(materials)}) must equal "
                f"1 (core) + {len(shell_layers)} (shells) = {1 + len(shell_layers)}"
            )

        if 'roundings' in self.config:
            roundings = self.config.get('roundings')
            if len(roundings) != len(materials):
                raise ValueError(
                    f"roundings length ({len(roundings)}) must equal "
                    f"materials length ({len(materials)})"
                )
        elif 'rounding' in self.config:
            single_rounding = self.config.get('rounding', 0.25)
            roundings = [single_rounding] * len(materials)
        else:
            roundings = [0.25] * len(materials)

        gap = self.config.get('gap', 10)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)

        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2 * thickness)

        total_size = sizes[-1]
        shift_distance = (total_size + gap) / 2

        # Check if adaptive mesh is enabled
        use_adaptive_mesh = self.config.get('use_adaptive_mesh', False)

        if use_adaptive_mesh:
            return self._advanced_dimer_cube_adaptive(
                sizes, materials, roundings, gap, offset,
                tilt_angle, tilt_axis, rotation_angle,
                mesh, shift_distance
            )

        # Standard uniform mesh (original behavior)
        code = f"""
%% Geometry: Advanced Dimer Cube
mesh_density = {mesh};
gap = {gap};
shift_distance = {shift_distance};

"""

        # Particle 1
        code += "\n%% === Particle 1 (Left) ===\n"
        particles_list = []

        for i, (size, material, rounding) in enumerate(zip(sizes, materials, roundings)):
            if i == 0:
                code += f"% Core: {material}\n"
                code += f"p1_core = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p1_core = shift(p1_core, [-shift_distance, 0, 0]);\n"
                particles_list.append("p1_core")
            else:
                shell_num = i
                code += f"\n% Shell {shell_num}: {material}\n"
                code += f"p1_shell{shell_num} = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p1_shell{shell_num} = shift(p1_shell{shell_num}, [-shift_distance, 0, 0]);\n"
                particles_list.append(f"p1_shell{shell_num}")

        # Particle 2
        code += "\n%% === Particle 2 (Right with transformations) ===\n"

        for i, (size, material, rounding) in enumerate(zip(sizes, materials, roundings)):
            if i == 0:
                code += f"% Core: {material}\n"
                code += f"p2_core = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p2_core = rot(p2_core, {rotation_angle}, [0, 0, 1]);\n"
                code += f"p2_core = rot(p2_core, {tilt_angle}, [{tilt_axis[0]}, {tilt_axis[1]}, {tilt_axis[2]}]);\n"
                code += f"p2_core = shift(p2_core, [shift_distance, 0, 0]);\n"
                code += f"p2_core = shift(p2_core, [{offset[0]}, {offset[1]}, {offset[2]}]);\n"
                particles_list.append("p2_core")
            else:
                shell_num = i
                code += f"\n% Shell {shell_num}: {material}\n"
                code += f"p2_shell{shell_num} = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p2_shell{shell_num} = rot(p2_shell{shell_num}, {rotation_angle}, [0, 0, 1]);\n"
                code += f"p2_shell{shell_num} = rot(p2_shell{shell_num}, {tilt_angle}, [{tilt_axis[0]}, {tilt_axis[1]}, {tilt_axis[2]}]);\n"
                code += f"p2_shell{shell_num} = shift(p2_shell{shell_num}, [shift_distance, 0, 0]);\n"
                code += f"p2_shell{shell_num} = shift(p2_shell{shell_num}, [{offset[0]}, {offset[1]}, {offset[2]}]);\n"
                particles_list.append(f"p2_shell{shell_num}")

        particles_str = ", ".join(particles_list)
        code += f"\n%% Combine all particles\nparticles = {{{particles_str}}};\n"

        return code

    def _advanced_dimer_cube_adaptive(self, sizes, materials, roundings, gap, offset,
                                       tilt_angle, tilt_axis, rotation_angle,
                                       base_mesh, shift_distance):
        """
        Generate advanced dimer cube with adaptive mesh (per-face density control).

        For dimer structures along x-axis:
        - Particle 1 (left): +x face is gap-facing (fine mesh)
        - Particle 2 (right): -x face is gap-facing (fine mesh)
        - Opposite faces: coarse mesh
        - Side faces (y, z): medium mesh
        """
        import scipy.io as sio

        # Get adaptive mesh settings
        adaptive_config = self.config.get('adaptive_mesh', {})
        gap_density = adaptive_config.get('gap_density', base_mesh)
        back_density = adaptive_config.get('back_density', max(6, base_mesh // 3))
        side_density = adaptive_config.get('side_density', max(8, base_mesh // 2))

        output_dir = Path(self.config.get('output_dir', './results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"  Adaptive mesh enabled:")
            print(f"    Gap faces: {gap_density}")
            print(f"    Back faces: {back_density}")
            print(f"    Side faces: {side_density}")

        code = f"""
%% Geometry: Advanced Dimer Cube (Adaptive Mesh)
%% Gap-facing faces have fine mesh, opposite faces have coarse mesh
gap = {gap};
shift_distance = {shift_distance};

fprintf('Loading adaptive mesh geometry...\\n');
"""

        particles_list = []
        particle_idx = 1

        # Generate mesh for each surface layer
        for layer_idx, (size, material, rounding) in enumerate(zip(sizes, materials, roundings)):

            # Particle 1 (Left): +x is gap-facing
            densities_p1 = {
                '+x': gap_density,   # Gap face - fine
                '-x': back_density,  # Back face - coarse
                '+y': side_density,  # Side faces - medium
                '-y': side_density,
                '+z': side_density,
                '-z': side_density,
            }

            mesh_gen = AdaptiveCubeMesh(size, rounding=rounding, verbose=self.verbose)
            # Use proper curved rounding (not shrink-based)
            if rounding > 0:
                verts1, faces1 = mesh_gen.generate_proper_rounded(densities_p1)
            else:
                verts1, faces1 = mesh_gen.generate(densities_p1)

            # Shift particle 1 to left position
            verts1[:, 0] -= shift_distance

            # Save to .mat file (MNPBEM expects Nx4, 4th column NaN for triangles)
            mat_file1 = f'adaptive_mesh_p1_layer{layer_idx}.mat'
            mat_path1 = output_dir / mat_file1
            sio.savemat(str(mat_path1), {'vertices': verts1, 'faces': faces1}, do_compression=True)

            if layer_idx == 0:
                p1_name = "p1_core"
            else:
                p1_name = f"p1_shell{layer_idx}"

            code += f"""
% Particle 1, Layer {layer_idx}: {material}
mesh_data = load('{mat_file1}');
{p1_name} = particle(mesh_data.vertices, mesh_data.faces, op, 'interp', 'flat');
fprintf('  {p1_name}: %d vertices, %d faces\\n', size(mesh_data.vertices, 1), size(mesh_data.faces, 1));
"""
            particles_list.append(p1_name)

            # Particle 2 (Right): -x is gap-facing
            densities_p2 = {
                '+x': back_density,  # Back face - coarse
                '-x': gap_density,   # Gap face - fine
                '+y': side_density,  # Side faces - medium
                '-y': side_density,
                '+z': side_density,
                '-z': side_density,
            }

            # Use proper curved rounding (not shrink-based)
            if rounding > 0:
                verts2, faces2 = mesh_gen.generate_proper_rounded(densities_p2)
            else:
                verts2, faces2 = mesh_gen.generate(densities_p2)

            # Apply transformations for particle 2
            # 1. Rotation around z-axis
            if rotation_angle != 0:
                rad = np.radians(rotation_angle)
                cos_r, sin_r = np.cos(rad), np.sin(rad)
                rot_z = np.array([
                    [cos_r, -sin_r, 0],
                    [sin_r, cos_r, 0],
                    [0, 0, 1]
                ])
                verts2 = verts2 @ rot_z.T

            # 2. Tilt rotation around custom axis
            if tilt_angle != 0:
                verts2 = self._rotate_vertices(verts2, tilt_angle, tilt_axis)

            # 3. Shift to right position
            verts2[:, 0] += shift_distance

            # 4. Apply offset
            verts2[:, 0] += offset[0]
            verts2[:, 1] += offset[1]
            verts2[:, 2] += offset[2]

            # Save to .mat file
            mat_file2 = f'adaptive_mesh_p2_layer{layer_idx}.mat'
            mat_path2 = output_dir / mat_file2
            sio.savemat(str(mat_path2), {'vertices': verts2, 'faces': faces2}, do_compression=True)

            if layer_idx == 0:
                p2_name = "p2_core"
            else:
                p2_name = f"p2_shell{layer_idx}"

            code += f"""
% Particle 2, Layer {layer_idx}: {material}
mesh_data = load('{mat_file2}');
{p2_name} = particle(mesh_data.vertices, mesh_data.faces, op, 'interp', 'flat');
fprintf('  {p2_name}: %d vertices, %d faces\\n', size(mesh_data.vertices, 1), size(mesh_data.faces, 1));
"""
            particles_list.append(p2_name)

        # Print summary using actual face counts
        # Total elements = sum of all faces from both particles
        actual_elements = sum(len(f) for f in [faces1, faces2] for _ in range(len(sizes)))
        # For comparison: uniform mesh at gap_density
        uniform_densities = {f: gap_density for f in ['+x', '-x', '+y', '-y', '+z', '-z']}
        mesh_uniform = AdaptiveCubeMesh(sizes[-1], rounding=roundings[-1], verbose=False)
        if roundings[-1] > 0:
            _, uniform_faces = mesh_uniform.generate_proper_rounded(uniform_densities)
        else:
            _, uniform_faces = mesh_uniform.generate(uniform_densities)
        total_elements_uniform = len(uniform_faces) * 2 * len(sizes)
        total_elements_adaptive = len(faces1) * 2 * len(sizes)
        reduction = (1 - total_elements_adaptive / total_elements_uniform) * 100

        if self.verbose:
            print(f"  Element reduction: {reduction:.1f}%")
            print(f"    Uniform (all at {gap_density}): {total_elements_uniform} elements")
            print(f"    Adaptive: {total_elements_adaptive} elements")

        code += f"""
%% Summary: Adaptive mesh reduces element count by ~{reduction:.0f}%
fprintf('Adaptive mesh loaded successfully.\\n');
"""

        particles_str = ", ".join(particles_list)
        code += f"\n%% Combine all particles\nparticles = {{{particles_str}}};\n"

        return code

    def _rotate_vertices(self, vertices, angle_deg, axis):
        """Rotate vertices around arbitrary axis using Rodrigues' rotation formula."""
        rad = np.radians(angle_deg)
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # Normalize

        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # Rodrigues' rotation formula
        rotated = np.zeros_like(vertices)
        for i, v in enumerate(vertices):
            rotated[i] = (v * cos_a +
                         np.cross(axis, v) * sin_a +
                         axis * np.dot(axis, v) * (1 - cos_a))
        return rotated
    
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
        
        output_dir = self.config.get('output_dir', './results')
        
        if self.verbose:
            print(f"Loading DDA shape file...")
        
        loader = ShapeFileLoader(shape_file, voxel_size=voxel_size, method=method, verbose=self.verbose)
        loader.load()
        
        code = loader.generate_matlab_code(materials, output_dir=output_dir)
        
        return code
