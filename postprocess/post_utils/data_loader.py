"""
Data Loading Utilities

Loads and parses MATLAB output files.
"""

import numpy as np
import scipy.io as sio
import os


class DataLoader:
    """Handles loading of MATLAB output files."""

    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose

        # Validate required config keys
        output_dir = config.get('output_dir')
        simulation_name = config.get('simulation_name')

        if output_dir is None:
            raise ValueError("Config missing required key: 'output_dir'")
        if simulation_name is None:
            raise ValueError("Config missing required key: 'simulation_name'")

        self.output_dir = os.path.join(output_dir, simulation_name)
    
    def _loadmat(self, mat_file):
        """Load .mat file, supporting both v5/v7 (scipy) and v7.3 HDF5 (h5py)."""
        try:
            return sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True), 'scipy'
        except NotImplementedError:
            # MATLAB v7.3 (HDF5 format)
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "This .mat file is MATLAB v7.3 (HDF5) format.\n"
                    "Install h5py to read it:  pip install h5py"
                )
            if self.verbose:
                print("  Detected MATLAB v7.3 (HDF5) format, using h5py...")
            return h5py.File(mat_file, 'r'), 'h5py'

    def load_simulation_results(self):
        """Load simulation results from MATLAB output."""
        mat_file = os.path.join(self.output_dir, 'simulation_results.mat')

        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Results file not found: {mat_file}")

        if self.verbose:
            print(f"Loading results from: {mat_file}")

        # Load MATLAB file (supports v5/v7 and v7.3 HDF5)
        mat_data, fmt = self._loadmat(mat_file)

        if fmt == 'h5py':
            return self._load_results_h5(mat_data)

        if 'results' not in mat_data:
            raise KeyError(f"MAT file missing 'results' structure: {mat_file}")
        results = mat_data['results']
        
        # Extract data
        data = {
            'wavelength': self._extract_array(results.wavelength),
            'scattering': self._extract_array(results.scattering),
            'extinction': self._extract_array(results.extinction),
            'absorption': self._extract_array(results.absorption),
            'polarizations': self._extract_array(results.polarizations),
            'propagation_dirs': self._extract_array(results.propagation_dirs),
        }
        
        # Add calculation time if available
        if hasattr(results, 'calculation_time'):
            data['calculation_time'] = float(results.calculation_time)
        
        # Ensure 2D arrays for cross sections
        for key in ['scattering', 'extinction', 'absorption']:
            if data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)
        
        # FIX: Add n_polarizations to data dictionary
        data['n_polarizations'] = data['scattering'].shape[1]
        
        if self.verbose:
            print(f"  Loaded {len(data['wavelength'])} wavelength points")
            print(f"  Polarizations: {data['n_polarizations']}")
        
        # Load field data if available
        if hasattr(results, 'fields'):
            data['fields'] = self._load_field_data(results.fields)
            if self.verbose and data['fields']:
                # Count unique wavelengths and polarizations
                unique_wls = set(f.get('wavelength_idx', f.get('wavelength')) for f in data['fields'])
                unique_pols = set(f.get('polarization_idx', 0) for f in data['fields'])
                print(f"  Field data loaded: {len(data['fields'])} entries "
                      f"({len(unique_wls)} wavelength(s), {len(unique_pols)} polarization(s))")

        # Load surface charge data if available
        if hasattr(results, 'surface_charge'):
            data['surface_charge'] = self._load_surface_charge_data(results.surface_charge)
            if self.verbose and data['surface_charge']:
                print(f"  Surface charge loaded: {len(data['surface_charge'])} entries")

        return data
    
    def _load_results_h5(self, f):
        """Load simulation results from MATLAB v7.3 HDF5 file."""
        import h5py

        if 'results' not in f:
            raise KeyError("MAT file missing 'results' structure")

        results = f['results']

        def _h5_get(group, key):
            """Extract numpy array from HDF5, handling refs and transposition."""
            if key not in group:
                return None
            ds = group[key]
            if ds.dtype == h5py.ref_dtype:
                # Array of references — dereference first element (scalar field)
                return np.array(f[ds.flat[0]]).squeeze()
            val = np.array(ds).squeeze()
            # MATLAB HDF5 stores 2D arrays transposed
            if val.ndim == 2:
                val = val.T
            return val

        data = {
            'wavelength': _h5_get(results, 'wavelength'),
            'scattering': _h5_get(results, 'scattering'),
            'extinction': _h5_get(results, 'extinction'),
            'absorption': _h5_get(results, 'absorption'),
            'polarizations': _h5_get(results, 'polarizations'),
            'propagation_dirs': _h5_get(results, 'propagation_dirs'),
        }

        if 'calculation_time' in results:
            data['calculation_time'] = float(_h5_get(results, 'calculation_time'))

        # Ensure 2D arrays for cross sections
        for key in ['scattering', 'extinction', 'absorption']:
            if data[key] is not None and data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)

        data['n_polarizations'] = data['scattering'].shape[1]

        if self.verbose:
            print(f"  Loaded {len(data['wavelength'])} wavelength points")
            print(f"  Polarizations: {data['n_polarizations']}")

        # Load field data if available
        if 'fields' in results:
            data['fields'] = self._load_field_data_h5(f, results['fields'])
            if self.verbose and data['fields']:
                unique_wls = set(fd.get('wavelength_idx', fd.get('wavelength')) for fd in data['fields'])
                unique_pols = set(fd.get('polarization_idx', 0) for fd in data['fields'])
                print(f"  Field data loaded: {len(data['fields'])} entries "
                      f"({len(unique_wls)} wavelength(s), {len(unique_pols)} polarization(s))")

        # Load surface charge data if available
        if 'surface_charge' in results:
            data['surface_charge'] = self._load_surface_charge_h5(f, results['surface_charge'])
            if self.verbose and data['surface_charge']:
                print(f"  Surface charge loaded: {len(data['surface_charge'])} entries")

        f.close()
        return data

    def _load_field_data_h5(self, f, fields_grp):
        """Load field data from HDF5 struct array."""
        import h5py

        field_names = list(fields_grp.keys())
        if not field_names:
            return []

        first_ds = fields_grp[field_names[0]]
        is_ref = (first_ds.dtype == h5py.ref_dtype)
        n_entries = first_ds.size if is_ref else 1

        field_data_list = []
        for i in range(n_entries):
            entry = {}
            for name in field_names:
                ds = fields_grp[name]
                try:
                    if is_ref:
                        val = np.array(f[ds.flat[i]]).squeeze()
                    else:
                        val = np.array(ds).squeeze()
                    if isinstance(val, np.ndarray) and val.ndim == 2:
                        val = val.T
                except Exception:
                    continue

                if name == 'wavelength':
                    entry[name] = float(val)
                elif name in ('wavelength_idx', 'polarization_idx'):
                    entry[name] = int(val)
                else:
                    arr = np.array(val)
                    if arr.size > 0:
                        entry[name] = arr

            field_data_list.append(entry)

        return field_data_list

    def _load_surface_charge_h5(self, f, sc_grp):
        """Load surface charge data from HDF5 struct array."""
        import h5py

        sc_fields = list(sc_grp.keys())
        if not sc_fields:
            return []

        first_ds = sc_grp[sc_fields[0]]
        is_ref = (first_ds.dtype == h5py.ref_dtype)
        n_entries = first_ds.size if is_ref else 1

        sc_list = []
        for i in range(n_entries):
            entry = {}
            for name in sc_fields:
                ds = sc_grp[name]
                try:
                    if is_ref:
                        val = np.array(f[ds.flat[i]]).squeeze()
                    else:
                        val = np.array(ds).squeeze()
                    if isinstance(val, np.ndarray) and val.ndim == 2:
                        val = val.T
                except Exception:
                    continue

                if name in ('wavelength',):
                    entry[name] = float(val)
                elif name in ('wavelength_idx', 'polarization_idx'):
                    entry[name] = int(val)
                else:
                    arr = np.array(val)
                    entry[name] = arr if arr.size > 0 else None

            sc_list.append(entry)

        return sc_list

    def _load_field_data(self, fields_struct):
        """Load electromagnetic field data from MATLAB structure."""
        if fields_struct is None:
            return []
        
        # Handle single polarization vs multiple
        if not isinstance(fields_struct, np.ndarray):
            fields_struct = [fields_struct]
        
        field_data_list = []
        
        for field_item in fields_struct:
            # Validate required fields
            required_fields = ['wavelength', 'polarization', 'x_grid', 'y_grid', 'z_grid']
            missing_fields = [f for f in required_fields if not hasattr(field_item, f)]
            if missing_fields:
                raise AttributeError(f"Field data missing required attributes: {missing_fields}")

            field_dict = {
                'wavelength': float(field_item.wavelength),
                'polarization': self._extract_array(field_item.polarization),
                'x_grid': self._extract_array(field_item.x_grid),
                'y_grid': self._extract_array(field_item.y_grid),
                'z_grid': self._extract_array(field_item.z_grid),
            }

            # New fields for per-polarization peak wavelength support
            if hasattr(field_item, 'wavelength_idx'):
                field_dict['wavelength_idx'] = int(field_item.wavelength_idx)

            if hasattr(field_item, 'polarization_idx'):
                field_dict['polarization_idx'] = int(field_item.polarization_idx)
            
            # Electric field components
            if hasattr(field_item, 'e_total'):
                field_dict['e_total'] = self._extract_array(field_item.e_total)
            
            if hasattr(field_item, 'e_induced'):
                field_dict['e_induced'] = self._extract_array(field_item.e_induced)
            
            if hasattr(field_item, 'e_incoming'):
                field_dict['e_incoming'] = self._extract_array(field_item.e_incoming)
            
            # Enhancement and intensity
            if hasattr(field_item, 'enhancement'):
                field_dict['enhancement'] = self._extract_array(field_item.enhancement)

            if hasattr(field_item, 'intensity'):
                field_dict['intensity'] = self._extract_array(field_item.intensity)

            # Raw intensity fields for energy ratio calculation: sum(|E|²)/sum(|E0|²)
            if hasattr(field_item, 'e_sq'):
                field_dict['e_sq'] = self._extract_array(field_item.e_sq)

            if hasattr(field_item, 'e0_sq'):
                field_dict['e0_sq'] = self._extract_array(field_item.e0_sq)

            if hasattr(field_item, 'e_sq_ext'):
                field_dict['e_sq_ext'] = self._extract_array(field_item.e_sq_ext)

            if hasattr(field_item, 'e_sq_int'):
                field_dict['e_sq_int'] = self._extract_array(field_item.e_sq_int)

            field_data_list.append(field_dict)
        
        return field_data_list
    
    def _load_surface_charge_data(self, sc_struct):
        """
        Load surface charge data from MATLAB structure.

        Returns:
            list: List of surface charge dictionaries, one per entry
        """
        if sc_struct is None:
            return []

        # Handle both single struct and array of structs
        if not isinstance(sc_struct, np.ndarray):
            sc_struct = [sc_struct]

        surface_charge_list = []

        for sc in sc_struct:
            sc_data = {
                'wavelength': float(sc.wavelength) if hasattr(sc, 'wavelength') else None,
                'wavelength_idx': int(sc.wavelength_idx) if hasattr(sc, 'wavelength_idx') else None,
                'polarization': self._extract_array(sc.polarization) if hasattr(sc, 'polarization') else None,
                'polarization_idx': int(sc.polarization_idx) if hasattr(sc, 'polarization_idx') else None,
                'vertices': self._extract_array(sc.vertices) if hasattr(sc, 'vertices') else None,
                'faces': self._extract_array(sc.faces) if hasattr(sc, 'faces') else None,
                'centroids': self._extract_array(sc.centroids) if hasattr(sc, 'centroids') else None,
                'normals': self._extract_array(sc.normals) if hasattr(sc, 'normals') else None,
                'areas': self._extract_array(sc.areas) if hasattr(sc, 'areas') else None,
                'charge': self._extract_array(sc.charge) if hasattr(sc, 'charge') else None,
            }

            surface_charge_list.append(sc_data)

        return surface_charge_list

    def _extract_array(self, matlab_array):
        """Extract numpy array from MATLAB data."""
        if matlab_array is None:
            return np.array([])

        # Convert to numpy array if not already
        arr = np.array(matlab_array)

        # Handle scalar
        if arr.ndim == 0:
            return arr.item()

        return arr
    
    def load_text_results(self):
        """Load results from text file (backup method)."""
        txt_file = os.path.join(self.output_dir, 'simulation_results.txt')
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Results file not found: {txt_file}")
        
        if self.verbose:
            print(f"Loading results from text file: {txt_file}")
        
        # Load data
        data_array = np.loadtxt(txt_file, skiprows=1)

        # Validate data format: should have (3*n_pol + 1) columns
        # Column layout: wavelength, sca_pol1..n, ext_pol1..n, abs_pol1..n
        n_cols = data_array.shape[1]
        if (n_cols - 1) % 3 != 0:
            raise ValueError(
                f"Invalid text file format: expected (3*n_pol + 1) columns, got {n_cols}. "
                f"File may be corrupted or have unexpected format."
            )
        n_pol = (n_cols - 1) // 3

        if n_pol < 1:
            raise ValueError(f"Invalid number of polarizations calculated: {n_pol}")

        data = {
            'wavelength': data_array[:, 0],
            'scattering': data_array[:, 1:n_pol+1],
            'extinction': data_array[:, n_pol+1:2*n_pol+1],
            'absorption': data_array[:, 2*n_pol+1:],
            'n_polarizations': n_pol,  # FIX: Add this here too
        }
        
        if self.verbose:
            print(f"  Loaded {len(data['wavelength'])} wavelength points")
            print(f"  Polarizations: {n_pol}")
        
        return data
