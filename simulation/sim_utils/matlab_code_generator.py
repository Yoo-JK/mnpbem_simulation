def _generate_wavelength_loop_with_chunking(self):
        """
        Wavelength loop with memory-efficient chunking AND field calculation support.
        
        FINAL VERSION - Supports:
        1. Direct solver + parfor
        2. Iterative solver + parfor (H-matrix compression ~14% memory)
        3. Serial execution with thread parallelism
        4. Proper memory management with bem = clear(bem)
        5. Internal + External field calculation
        
        FIXED BUGS:
        - pt_internal.pc → pt_internal (compoint returns point object directly)
        - 4D array handling for grid-based field calculations
        - NaN filtering after reshape (meshfield mindist filtering)
        - 3D array slicing condition: size(...,3) == n_polarizations (not > 3)
        - Enhancement calculation threshold (CRITICAL FIX!)
        - Changed to INTENSITY enhancement: |E|²/|E0|² (not field magnitude)
        
        Strategy: 
        1. Calculate ALL cross sections first (no field in loop)
        2. Find peak AFTER all chunks complete
        3. Calculate field separately for peak wavelength (internal + external)
        """
        
        wavelength_range = self.config['wavelength_range']
        chunk_size = self.config.get('wavelength_chunk_size', 20)
        use_parallel = self.config.get('use_parallel', False)
        use_iterative = self.config.get('use_iterative_solver', False)
        excitation_type = self.config['excitation_type']
        calculate_fields = self.config.get('calculate_fields', False)
        
        code = f"""
%% Wavelength Loop with Chunking (Memory-Efficient!)
if ~exist('enei', 'var')
    enei = linspace({wavelength_range[0]}, {wavelength_range[1]}, {wavelength_range[2]});
end

n_wavelengths = length(enei);
n_polarizations = size(pol, 1);

% Chunking setup
chunk_size = {chunk_size};
n_chunks = ceil(n_wavelengths / chunk_size);

fprintf('\\n');
fprintf('================================================================\\n');
fprintf('     Starting BEM Calculation with Wavelength Chunking         \\n');
fprintf('================================================================\\n');
fprintf('Total wavelengths: %d\\n', n_wavelengths);
fprintf('Chunk size: %d wavelengths\\n', chunk_size);
fprintf('Number of chunks: %d\\n', n_chunks);
"""

        if use_iterative:
            code += """fprintf('Solver mode: ITERATIVE (H-matrix compression, ~14%% memory)\\n');
"""
        else:
            code += """fprintf('Solver mode: DIRECT (full matrix)\\n');
"""
        
        code += """fprintf('----------------------------------------------------------------\\n');

% Initialize result arrays
sca = zeros(n_wavelengths, n_polarizations);
ext = zeros(n_wavelengths, n_polarizations);
abs_cross = zeros(n_wavelengths, n_polarizations);
"""

        # Parallel setup
        if use_parallel:
            code += self._generate_parallel_setup()
        
        # Excitation initialization
        code += """
%% Initialize Excitation (once, outside all loops!)
fprintf('\\nInitializing excitation object...\\n');
"""

        if excitation_type == 'planewave':
            code += """exc = planewave(pol, dir, op);
fprintf('  [OK] Plane wave excitation initialized\\n');
"""
        elif excitation_type == 'dipole':
            code += """pt = compoint(p, dip_pos, op);
exc = dipole(pt, dip_mom, op);
fprintf('  [OK] Dipole excitation initialized\\n');
"""
        elif excitation_type == 'eels':
            code += """exc = eelsret(p, impact, beam_energy, 'width', beam_width, op);
fprintf('  [OK] EELS excitation initialized\\n');
"""

        code += """
% Start overall timer
total_start = tic;

%% ========================================
%% CHUNK LOOP: Calculate cross sections ONLY (NO field calculation)
%% ========================================
for ichunk = 1:n_chunks
    % Calculate wavelength indices for this chunk
    idx_start = (ichunk-1) * chunk_size + 1;
    idx_end = min(ichunk * chunk_size, n_wavelengths);
    chunk_indices = idx_start:idx_end;
    n_chunk = length(chunk_indices);
    
    fprintf('\\n');
    fprintf('================================================================\\n');
    fprintf('  Processing Chunk %d/%d: wavelengths %d-%d (%d points)\\n', ...
            ichunk, n_chunks, idx_start, idx_end, n_chunk);
    fprintf('  lambda range: %.1f - %.1f nm\\n', ...
            enei(idx_start), enei(idx_end));
    fprintf('================================================================\\n');
    
    chunk_start = tic;
    
    % CRITICAL: Clear BEM solver between chunks for memory efficiency
    % Use MNPBEM's clear method to release H-matrices while keeping structure
    if ichunk > 1
        fprintf('  -> Clearing BEM auxiliary matrices...\\n');
        bem = clear(bem);
        fprintf('  [OK] Memory released\\n');
    end
    
"""

        if use_parallel:
            # Parallel execution (works with both Direct and Iterative)
            code += """    % ========================================
    % PARALLEL EXECUTION (parfor)
    % ========================================
    if exist('parallel_enabled', 'var') && parallel_enabled
        fprintf('  Using parallel execution (parfor)\\n\\n');

        % Pre-allocate chunk-local arrays for parfor slicing
        sca_chunk = zeros(n_chunk, n_polarizations);
        ext_chunk = zeros(n_chunk, n_polarizations);
        abs_chunk = zeros(n_chunk, n_polarizations);
        
        % Local copies for parfor
        enei_local = enei;
        chunk_idx_local = chunk_indices;

        parfor i_local = 1:n_chunk
            ien = chunk_idx_local(i_local);

            try
                % Progress (sparse output in parfor)
                if mod(i_local-1, max(1, floor(n_chunk/5))) == 0
                    fprintf('    [Worker] lambda %d/%d (%.1f nm)\\n', ...
                            i_local, n_chunk, enei_local(ien));
                end

                % BEM calculation
                sig = bem \\ exc(p, enei_local(ien));

                % Store results
                sca_chunk(i_local, :) = exc.sca(sig);
                ext_chunk(i_local, :) = exc.ext(sig);
                abs_chunk(i_local, :) = ext_chunk(i_local, :) - sca_chunk(i_local, :);

            catch ME
                fprintf('    [ERROR] lambda %d (%.1f nm): %s\\n', ...
                        ien, enei_local(ien), ME.message);
                sca_chunk(i_local, :) = zeros(1, n_polarizations);
                ext_chunk(i_local, :) = zeros(1, n_polarizations);
                abs_chunk(i_local, :) = zeros(1, n_polarizations);
            end
        end

        % Copy chunk results to main arrays
        sca(chunk_indices, :) = sca_chunk;
        ext(chunk_indices, :) = ext_chunk;
        abs_cross(chunk_indices, :) = abs_chunk;

    else
        % Fallback to serial
        fprintf('  Using serial execution\\n\\n');
        
        for i_local = 1:n_chunk
            ien = chunk_indices(i_local);
            
            if mod(i_local-1, max(1, floor(n_chunk/10))) == 0
                fprintf('    Progress: %d/%d (lambda = %.1f nm)\\n', ...
                        i_local, n_chunk, enei(ien));
            end
            
            try
                sig = bem \\ exc(p, enei(ien));
                sca(ien, :) = exc.sca(sig);
                ext(ien, :) = exc.ext(sig);
                abs_cross(ien, :) = ext(ien, :) - sca(ien, :);
            catch ME
                fprintf('    [ERROR] lambda %d (%.1f nm): %s\\n', ...
                        ien, enei(ien), ME.message);
                sca(ien, :) = zeros(1, n_polarizations);
                ext(ien, :) = zeros(1, n_polarizations);
                abs_cross(ien, :) = zeros(1, n_polarizations);
            end
        end
    end
"""
        else:
            # Serial only
            code += """    % ========================================
    % SERIAL EXECUTION
    % ========================================
    fprintf('  Using serial execution\\n\\n');
    
    for i_local = 1:n_chunk
        ien = chunk_indices(i_local);
        
        % Progress
        if mod(i_local-1, max(1, floor(n_chunk/10))) == 0
            fprintf('    Progress: %d/%d (lambda = %.1f nm)\\n', ...
                    i_local, n_chunk, enei(ien));
        end
        
        try
            % BEM calculation
            sig = bem \\ exc(p, enei(ien));
            
            % Store results
            sca(ien, :) = exc.sca(sig);
            ext(ien, :) = exc.ext(sig);
            abs_cross(ien, :) = ext(ien, :) - sca(ien, :);
            
        catch ME
            fprintf('    [ERROR] lambda %d (%.1f nm): %s\\n', ...
                    ien, enei(ien), ME.message);
            sca(ien, :) = zeros(1, n_polarizations);
            ext(ien, :) = zeros(1, n_polarizations);
            abs_cross(ien, :) = zeros(1, n_polarizations);
        end
    end
"""
        
        # Chunk timing
        code += """    
    chunk_time = toc(chunk_start);
    fprintf('\\n  [OK] Chunk %d completed in %.1f seconds (%.1f min)\\n', ...
            ichunk, chunk_time, chunk_time/60);
    fprintf('  Average: %.2f sec/wavelength\\n', chunk_time/n_chunk);
    
    % Brief pause for garbage collection
    pause(0.5);
end

% Total timing for cross sections
total_time = toc(total_start);
calculation_time = total_time;
fprintf('\\n');
fprintf('================================================================\\n');
fprintf('ALL CHUNKS COMPLETED\\n');
fprintf('Total time: %.1f seconds (%.1f minutes)\\n', total_time, total_time/60);
fprintf('Average: %.2f seconds/wavelength\\n', total_time/n_wavelengths);
fprintf('================================================================\\n');
"""

        # ================================================================
        # FIELD CALCULATION (After all chunks complete) - COMPLETE FIX
        # ================================================================
        if calculate_fields:
            code += """
%% ========================================
%% FIELD CALCULATION (After all chunks complete)
%% Internal + External Field Support (Complete 3D/4D handling)
%% ========================================
fprintf('\\n');
fprintf('================================================================\\n');
fprintf('    Field Calculation at Peak Wavelength (Internal + External)  \\n');
fprintf('================================================================\\n');
"""
            
            # Determine peak wavelength
            field_wl_idx = self.config.get('field_wavelength_idx', 'middle')
            
            if field_wl_idx == 'middle':
                code += """
% Use middle wavelength
field_wavelength_idx = round(n_wavelengths / 2);
fprintf('Using middle wavelength: lambda = %.1f nm (index %d)\\n', ...
        enei(field_wavelength_idx), field_wavelength_idx);
"""
            elif field_wl_idx == 'peak':
                code += """
% Find absorption peak wavelength
fprintf('Finding absorption peak...\\n');
abs_avg = mean(abs_cross, 2);
[max_abs, field_wavelength_idx] = max(abs_avg);
fprintf('  [OK] Peak absorption: %.2e nm^2 at lambda = %.1f nm (index %d)\\n', ...
        max_abs, enei(field_wavelength_idx), field_wavelength_idx);
"""
            elif field_wl_idx == 'peak_ext':
                code += """
% Find extinction peak wavelength
fprintf('Finding extinction peak...\\n');
ext_avg = mean(ext, 2);
[max_ext, field_wavelength_idx] = max(ext_avg);
fprintf('  [OK] Peak extinction: %.2e nm^2 at lambda = %.1f nm (index %d)\\n', ...
        max_ext, enei(field_wavelength_idx), field_wavelength_idx);
"""
            elif isinstance(field_wl_idx, int):
                code += f"""
% Use specified wavelength index
field_wavelength_idx = {field_wl_idx};
fprintf('Using specified wavelength: lambda = %.1f nm (index %d)\\n', ...
        enei(field_wavelength_idx), field_wavelength_idx);
"""
            else:
                code += """
% Default: use middle wavelength
field_wavelength_idx = round(n_wavelengths / 2);
fprintf('Using middle wavelength: lambda = %.1f nm (index %d)\\n', ...
        enei(field_wavelength_idx), field_wavelength_idx);
"""
            
            # Create field grid
            use_substrate = self.config.get('use_substrate', False)
            
            if not use_substrate:
                field_region = self.config.get('field_region', {})
                x_range = field_region.get('x_range', [-50, 50, 101])
                y_range = field_region.get('y_range', [0, 0, 1])
                z_range = field_region.get('z_range', [0, 0, 1])
                
                code += """
% Create field grid
fprintf('\\nCreating field grid...\\n');
"""
                
                if y_range[2] == 1:  # xz-plane
                    code += f"""x_field = linspace({x_range[0]}, {x_range[1]}, {x_range[2]});
z_field = linspace({z_range[0]}, {z_range[1]}, {z_range[2]});
[x_grid, z_grid] = meshgrid(x_field, z_field);
y_grid = {y_range[0]} * ones(size(x_grid));
"""
                elif z_range[2] == 1:  # xy-plane
                    code += f"""x_field = linspace({x_range[0]}, {x_range[1]}, {x_range[2]});
y_field = linspace({y_range[0]}, {y_range[1]}, {y_range[2]});
[x_grid, y_grid] = meshgrid(x_field, y_field);
z_grid = {z_range[0]} * ones(size(x_grid));
"""
            else:
                code += """
% Substrate mode: grid already created
x_grid = reshape(x_grid, grid_shape);
y_grid = reshape(y_grid, grid_shape);
z_grid = reshape(z_grid, grid_shape);
"""
            
            # External + Internal field setup
            mindist_external = self.config.get('field_mindist', 0.2 if not use_substrate else 0.5)
            mindist_internal = self.config.get('field_mindist_internal', 0.0)
            nmax = self.config.get('field_nmax', 2000)
            
            code += f"""
% Store grid shape
grid_shape = size(x_grid);
n_grid_points = numel(x_grid);

%% ========================================
%% STEP 1: External Field Setup (meshfield)
%% ========================================
fprintf('\\n[1/2] Setting up EXTERNAL field calculation...\\n');
field_mindist_external = {mindist_external};
emesh_external = meshfield(p, x_grid, y_grid, z_grid, op, ...
                           'mindist', field_mindist_external, 'nmax', {nmax});
fprintf('  → External meshfield: %d points\\n', emesh_external.pt.n);

%% ========================================
%% STEP 2: Internal Field Setup (compoint + greenfunction)
%% ========================================
fprintf('\\n[2/2] Setting up INTERNAL field calculation...\\n');

% Auto-detect internal medium index
if size(inout, 1) == 1
    internal_medium_idx = inout(1, 1);
    fprintf('  → Single particle: medium %d\\n', internal_medium_idx);
else
    internal_medium_idx = inout(1, 1);
    fprintf('  → Multi-particle: medium %d (first particle interior)\\n', internal_medium_idx);
end

fprintf('  Creating compoint (medium=%d, mindist={mindist_internal})...\\n', internal_medium_idx);

try
    % FIX 1: compoint returns point object directly (not .pc)
    pt_internal = compoint(p, [x_grid(:), y_grid(:), z_grid(:)], op, ...
                           'medium', internal_medium_idx, ...
                           'mindist', {mindist_internal});
    
    fprintf('  Creating Green function...\\n');
    g_internal = greenfunction(pt_internal, p, op);
    
    % FIX 1: Use pt_internal.n (not pt_internal.pc.n)
    fprintf('  → Internal field setup: %d points\\n', pt_internal.n);
    has_internal_field = true;
    
catch ME
    fprintf('  [!] Internal field setup failed: %s\\n', ME.message);
    fprintf('  [!] Proceeding with EXTERNAL field only\\n');
    has_internal_field = false;
end

fprintf('\\n[OK] Field calculation setup complete\\n');
fprintf('  Grid: %dx%d = %d points\\n', grid_shape(1), grid_shape(2), n_grid_points);
if has_internal_field
    fprintf('  External: %d points, Internal: %d points\\n', ...
            emesh_external.pt.n, pt_internal.n);
else
    fprintf('  External: %d points (Internal: disabled)\\n', emesh_external.pt.n);
end

%% ========================================
%% STEP 3: Calculate BEM Solution at Peak Wavelength
%% ========================================
fprintf('\\nCalculating BEM solution at peak wavelength...\\n');
field_calc_start = tic;

% Clear BEM and recalculate
bem = clear(bem);
sig_peak = bem \\ exc(p, enei(field_wavelength_idx));
fprintf('  [OK] BEM solution ready\\n');

%% ========================================
%% STEP 4: Calculate Fields for Each Polarization
%% ========================================
field_data = struct();

for ipol = 1:n_polarizations
    fprintf('\\n  Processing polarization %d/%d...\\n', ipol, n_polarizations);
    
"""
            
            # Create single-polarization excitation
            excitation_type = self.config['excitation_type']
            
            if excitation_type == 'planewave':
                code += """    % Create single-polarization plane wave
    exc_single = planewave(pol(ipol, :), dir(ipol, :), op);
"""
            elif excitation_type == 'dipole':
                code += """    % Create dipole excitation
    pt_single = compoint(p, dip_pos, op);
    exc_single = dipole(pt_single, dip_mom, op);
"""
            else:
                code += """    % Use original excitation
    exc_single = exc;
"""
            
            # FIXED: Complete 3D/4D array handling
            code += """
    %% EXTERNAL FIELD
    fprintf('    → External field...\\n');
    
    % Induced field from BEM solution
    e_induced_ext_all = emesh_external(sig_peak);
    
    % FIX 2: Handle 4D arrays (grid-based) and 3D arrays + NaN filtering
    fprintf('      Raw external field size: [%s]\\n', num2str(size(e_induced_ext_all)));
    
    if ndims(e_induced_ext_all) == 4
        % 4D array: [nz, nx, 3, n_pol] - typical for grid-based calculations
        e_induced_ext = e_induced_ext_all(:, :, :, ipol);  % [nz, nx, 3]
        e_induced_ext = reshape(e_induced_ext, [], 3);  % [15251, 3] with NaNs
        
        % CRITICAL FIX: Remove NaN points (meshfield filtering)
        valid_mask = all(isfinite(e_induced_ext), 2);
        e_induced_ext = e_induced_ext(valid_mask, :);  % [15148, 3]
        fprintf('      After filtering: [%d, 3] (removed %d NaN points)\\n', ...
                size(e_induced_ext, 1), sum(~valid_mask));
        
    elseif ndims(e_induced_ext_all) == 3
        % 3D array: Two cases
        if size(e_induced_ext_all, 3) == n_polarizations
            % [n_points, 3, n_pol] → slice
            e_induced_ext = e_induced_ext_all(:, :, ipol);
        else
            % [n_points, 3] or [nz, nx, 3] → keep as is
            e_induced_ext = e_induced_ext_all;
        end
    else
        % 2D array: [n_points, 3] - single polarization
        e_induced_ext = e_induced_ext_all;
    end
    
    % Incoming field
    exc_field_ext = exc_single.field(emesh_external.pt, enei(field_wavelength_idx));
    e_incoming_ext_all = emesh_external(exc_field_ext);
    
    fprintf('      Raw incoming field size: [%s]\\n', num2str(size(e_incoming_ext_all)));
    
    % Extract incoming field for this polarization
    if ndims(e_incoming_ext_all) == 4
        e_incoming_ext = e_incoming_ext_all(:, :, :, ipol);  % [nz, nx, 3]
        e_incoming_ext = reshape(e_incoming_ext, [], 3);  % [15251, 3]
        
        % CRITICAL FIX: Remove NaN points (same as induced)
        valid_mask = all(isfinite(e_incoming_ext), 2);
        e_incoming_ext = e_incoming_ext(valid_mask, :);  % [15148, 3]
        
    elseif ndims(e_incoming_ext_all) == 3
        % 3D array: Two cases
        if size(e_incoming_ext_all, 3) == n_polarizations
            % [n_points, 3, n_pol] → slice
            e_incoming_ext = e_incoming_ext_all(:, :, ipol);
        else
            % [nz, nx, 3] → reshape and filter
            e_incoming_ext = reshape(e_incoming_ext_all, [], 3);
            valid_mask = all(isfinite(e_incoming_ext), 2);
            e_incoming_ext = e_incoming_ext(valid_mask, :);
        end
    else
        e_incoming_ext = e_incoming_ext_all;
    end
    
    % Ensure 2D: [n_points, 3]
    if ndims(e_induced_ext) == 3
        e_induced_ext = squeeze(e_induced_ext);
    end
    if ndims(e_incoming_ext) == 3
        e_incoming_ext = squeeze(e_incoming_ext);
    end
    
    % Verify sizes match
    if size(e_induced_ext, 1) ~= size(e_incoming_ext, 1)
        error('Size mismatch: induced [%d, 3] vs incoming [%d, 3]', ...
              size(e_induced_ext, 1), size(e_incoming_ext, 1));
    end
    
    % Total external field
    e_total_ext = e_induced_ext + e_incoming_ext;
    
    fprintf('      Final external field size: [%s]\\n', num2str(size(e_total_ext)));
    
    %% INTERNAL FIELD
    if has_internal_field
        fprintf('    → Internal field...\\n');
        
        % Induced field using Green function
        f_induced_int = field(g_internal, sig_peak);
        e_induced_int_all = f_induced_int.e;
        
        fprintf('      Raw internal induced field size: [%s]\\n', num2str(size(e_induced_int_all)));
        
        % FIX 3: Complete 3D/4D handling for internal fields
        if ndims(e_induced_int_all) == 4
            % 4D: [nz, nx, 3, n_pol]
            e_induced_int = e_induced_int_all(:, :, :, ipol);
            e_induced_int = reshape(e_induced_int, [], 3);
            
            % Remove NaN points
            valid_mask = all(isfinite(e_induced_int), 2);
            e_induced_int = e_induced_int(valid_mask, :);
            
        elseif ndims(e_induced_int_all) == 3
            % 3D: Check if polarization dimension exists
            if size(e_induced_int_all, 3) == n_polarizations
                % [n_points, 3, n_pol] → slice for this polarization
                e_induced_int = e_induced_int_all(:, :, ipol);
                fprintf('      Sliced 3D array: [%s]\\n', num2str(size(e_induced_int)));
            else
                % [n_points, 3] → keep as is
                e_induced_int = e_induced_int_all;
            end
        else
            % 2D: [n_points, 3]
            e_induced_int = e_induced_int_all;
        end
        
        % Incoming field at internal points
        exc_field_int = exc_single.field(pt_internal, enei(field_wavelength_idx));
        e_incoming_int_all = exc_field_int.e;
        
        fprintf('      Raw internal incoming field size: [%s]\\n', num2str(size(e_incoming_int_all)));
        
        % Extract for this polarization
        if ndims(e_incoming_int_all) == 4
            % 4D: [nz, nx, 3, n_pol]
            e_incoming_int = e_incoming_int_all(:, :, :, ipol);
            e_incoming_int = reshape(e_incoming_int, [], 3);
            
            % Remove NaN points
            valid_mask = all(isfinite(e_incoming_int), 2);
            e_incoming_int = e_incoming_int(valid_mask, :);
            
        elseif ndims(e_incoming_int_all) == 3
            % 3D: Check if polarization dimension exists
            if size(e_incoming_int_all, 3) == n_polarizations
                % [n_points, 3, n_pol] → slice for this polarization
                e_incoming_int = e_incoming_int_all(:, :, ipol);
                fprintf('      Sliced 3D array: [%s]\\n', num2str(size(e_incoming_int)));
            else
                % [n_points, 3] → keep as is
                e_incoming_int = e_incoming_int_all;
            end
        else
            % 2D: [n_points, 3]
            e_incoming_int = e_incoming_int_all;
        end
        
        % Ensure 2D
        if ndims(e_induced_int) == 3
            e_induced_int = squeeze(e_induced_int);
        end
        if ndims(e_incoming_int) == 3
            e_incoming_int = squeeze(e_incoming_int);
        end
        
        % Verify sizes
        fprintf('      After processing: induced [%s], incoming [%s]\\n', ...
                num2str(size(e_induced_int)), num2str(size(e_incoming_int)));
        
        if size(e_induced_int, 1) ~= size(e_incoming_int, 1) || size(e_induced_int, 2) ~= 3
            error('Size mismatch: internal induced [%s] vs incoming [%s]', ...
                  num2str(size(e_induced_int)), num2str(size(e_incoming_int)));
        end
        
        % Total internal field
        e_total_int = e_induced_int + e_incoming_int;
        
        fprintf('      Final internal field size: [%s]\\n', num2str(size(e_total_int)));
    else
        e_total_int = [];
    end
    
    %% MERGE EXTERNAL + INTERNAL (IMPROVED!)
    fprintf('    → Merging fields...\\n');
    
    % Initialize full grid with NaN
    e_total_full = nan(n_grid_points, 3);
    e_incoming_full = nan(n_grid_points, 3);
    
    % NEW: Separate storage for visualization
    e_total_ext_grid = nan(n_grid_points, 3);
    e_total_int_grid = nan(n_grid_points, 3);
    
    % Flatten grid coordinates
    x_flat = x_grid(:);
    y_flat = y_grid(:);
    z_flat = z_grid(:);
    
    % Fill external points
    fprintf('      Filling %d external points...\\n', emesh_external.pt.n);
    for ii = 1:emesh_external.pt.n
        dx = abs(x_flat - emesh_external.pt.pos(ii,1));
        dy = abs(y_flat - emesh_external.pt.pos(ii,2));
        dz = abs(z_flat - emesh_external.pt.pos(ii,3));
        [~, idx] = min(dx + dy + dz);
        
        e_total_full(idx, :) = e_total_ext(ii, :);
        e_incoming_full(idx, :) = e_incoming_ext(ii, :);
        
        % Store external separately
        e_total_ext_grid(idx, :) = e_total_ext(ii, :);
    end
    
    % Fill internal points (IMPROVED: Exact grid matching)
    if has_internal_field
        fprintf('      Filling %d internal points (exact matching)...\\n', pt_internal.n);
        
        % Extract grid vectors
        if numel(unique(y_grid)) == 1
            % xz-plane: y is constant
            x_vec = unique(x_grid);
            z_vec = unique(z_grid);
            
            fprintf('      Grid type: xz-plane (y=%.1f)\\n', unique(y_grid));
            fprintf('      x range: [%.1f, %.1f], %d points\\n', min(x_vec), max(x_vec), numel(x_vec));
            fprintf('      z range: [%.1f, %.1f], %d points\\n', min(z_vec), max(z_vec), numel(z_vec));
            
            % Find nearest indices for each internal point
            ix = zeros(pt_internal.n, 1);
            iz = zeros(pt_internal.n, 1);
            
            for ii = 1:pt_internal.n
                [~, ix(ii)] = min(abs(pt_internal.pos(ii,1) - x_vec));
                [~, iz(ii)] = min(abs(pt_internal.pos(ii,3) - z_vec));
            end
            
            % Convert to linear index
            linear_idx = sub2ind(grid_shape, iz, ix);
            
        elseif numel(unique(z_grid)) == 1
            % xy-plane: z is constant
            x_vec = unique(x_grid);
            y_vec = unique(y_grid);
            
            fprintf('      Grid type: xy-plane (z=%.1f)\\n', unique(z_grid));
            fprintf('      x range: [%.1f, %.1f], %d points\\n', min(x_vec), max(x_vec), numel(x_vec));
            fprintf('      y range: [%.1f, %.1f], %d points\\n', min(y_vec), max(y_vec), numel(y_vec));
            
            % Find nearest indices
            ix = zeros(pt_internal.n, 1);
            iy = zeros(pt_internal.n, 1);
            
            for ii = 1:pt_internal.n
                [~, ix(ii)] = min(abs(pt_internal.pos(ii,1) - x_vec));
                [~, iy(ii)] = min(abs(pt_internal.pos(ii,2) - y_vec));
            end
            
            % Convert to linear index
            linear_idx = sub2ind(grid_shape, iy, ix);
        else
            error('Unsupported grid configuration');
        end
        
        % Check for duplicates
        [unique_idx, ~, ic] = unique(linear_idx);
        n_unique = numel(unique_idx);
        n_duplicates = pt_internal.n - n_unique;
        
        if n_duplicates > 0
            fprintf('      Warning: %d duplicate grid points detected\\n', n_duplicates);
            fprintf('      Multiple internal points map to same grid location\\n');
        end
        
        % Assign internal field to exact grid locations
        e_total_full(linear_idx, :) = e_total_int;
        e_incoming_full(linear_idx, :) = e_incoming_int;
        
        % Store internal separately
        e_total_int_grid(linear_idx, :) = e_total_int;
        
        fprintf('      Internal field mapped to %d unique grid points\\n', n_unique);
    end
    
    % Count filled points
    n_filled = sum(all(isfinite(e_total_full), 2));
    n_ext_only = sum(all(isfinite(e_total_ext_grid), 2) & all(isnan(e_total_int_grid), 2));
    n_int_only = sum(all(isfinite(e_total_int_grid), 2) & all(isnan(e_total_ext_grid), 2));
    n_overlap = sum(all(isfinite(e_total_ext_grid), 2) & all(isfinite(e_total_int_grid), 2));
    
    fprintf('      Merge complete: %d total points filled\\n', n_filled);
    fprintf('        External only: %d points\\n', n_ext_only);
    fprintf('        Internal only: %d points\\n', n_int_only);
    fprintf('        Overlap: %d points\\n', n_overlap);
    
    %% CALCULATE INTENSITY ENHANCEMENT (IMPROVED - THRESHOLD!)
    % Compute field intensities
    e_intensity = sum(e_total_full .* conj(e_total_full), 2);     % |E|²
    e0_intensity = sum(e_incoming_full .* conj(e_incoming_full), 2);  % |E0|²
    
    % CRITICAL FIX: Threshold to prevent division by very small numbers
    % This is especially important for internal fields in metals where
    % incoming field can be exponentially small (e.g., 1e-50, 1e-100)
    e0_threshold = 1e-10;
    
    % Calculate INTENSITY enhancement: |E|² / |E0|² (NOT field magnitude!)
    intensity_enhancement = e_intensity ./ max(e0_intensity, e0_threshold);
    
    % Mark unreliable points (very weak incoming field) as NaN
    intensity_enhancement(e0_intensity < e0_threshold) = NaN;
    
    fprintf('      Intensity enhancement calculated (merged): min=%.3f, max=%.3f, valid=%d\\n', ...
            min(intensity_enhancement(isfinite(intensity_enhancement))), ...
            max(intensity_enhancement(isfinite(intensity_enhancement))), ...
            sum(isfinite(intensity_enhancement)));
    
    % Separate intensity enhancement for internal/external (IMPROVED!)
    e_intensity_ext = sum(e_total_ext_grid .* conj(e_total_ext_grid), 2);  % |E_ext|²
    e_intensity_int = sum(e_total_int_grid .* conj(e_total_int_grid), 2);  % |E_int|²
    
    e0_intensity_ext = e0_intensity;  % Same incoming field
    e0_intensity_int = e0_intensity;
    
    % CRITICAL FIX: Apply threshold for both external and internal
    e0_threshold = 1e-10;
    
    % Intensity enhancement (NOT field magnitude!)
    intensity_enhancement_ext = e_intensity_ext ./ max(e0_intensity_ext, e0_threshold);
    intensity_enhancement_int = e_intensity_int ./ max(e0_intensity_int, e0_threshold);
    
    intensity_enhancement_ext(e0_intensity_ext < e0_threshold) = NaN;
    intensity_enhancement_int(e0_intensity_int < e0_threshold) = NaN;
    
    fprintf('      Intensity_enh_ext valid: %d/%d\\n', ...
            sum(isfinite(intensity_enhancement_ext)), numel(intensity_enhancement_ext));
    fprintf('      Intensity_enh_int valid: %d/%d\\n', ...
            sum(isfinite(intensity_enhancement_int)), numel(intensity_enhancement_int));
    
    % Reshape to grid
    intensity_enhancement = reshape(intensity_enhancement, grid_shape);
    e_intensity = reshape(e_intensity, grid_shape);
    e_total_grid = reshape(e_total_full, [grid_shape, 3]);
    
    % Reshape separate fields
    intensity_enhancement_ext = reshape(intensity_enhancement_ext, grid_shape);
    intensity_enhancement_int = reshape(intensity_enhancement_int, grid_shape);
    e_intensity_ext = reshape(e_intensity_ext, grid_shape);
    e_intensity_int = reshape(e_intensity_int, grid_shape);
    e_total_ext_grid = reshape(e_total_ext_grid, [grid_shape, 3]);
    e_total_int_grid = reshape(e_total_int_grid, [grid_shape, 3]);
    
    %% STORE FIELD DATA (with separate internal/external)
    field_data(ipol).wavelength = enei(field_wavelength_idx);
    field_data(ipol).wavelength_idx = field_wavelength_idx;
    field_data(ipol).polarization = pol(ipol, :);
    field_data(ipol).polarization_idx = ipol;
    
    % Combined (merged) - using intensity enhancement
    field_data(ipol).e_total = e_total_grid;
    field_data(ipol).enhancement = intensity_enhancement;  % |E|²/|E0|²
    field_data(ipol).intensity = e_intensity;              % |E|²
    
    % Separate fields
    field_data(ipol).e_total_ext = e_total_ext_grid;
    field_data(ipol).e_total_int = e_total_int_grid;
    field_data(ipol).enhancement_ext = intensity_enhancement_ext;  % |E|²/|E0|²
    field_data(ipol).enhancement_int = intensity_enhancement_int;  % |E|²/|E0|²
    field_data(ipol).intensity_ext = e_intensity_ext;              % |E|²
    field_data(ipol).intensity_int = e_intensity_int;              % |E|²
    
    % Grid coordinates
    field_data(ipol).x_grid = x_grid;
    field_data(ipol).y_grid = y_grid;
    field_data(ipol).z_grid = z_grid;
    
    fprintf('    → Stored field_data(%d)\\n', ipol);
    fprintf('      Valid points (merged): %d/%d\\n', sum(isfinite(intensity_enhancement(:))), numel(intensity_enhancement));
    fprintf('      Valid points (external): %d/%d\\n', sum(isfinite(intensity_enhancement_ext(:))), numel(intensity_enhancement_ext));
    fprintf('      Valid points (internal): %d/%d\\n', sum(isfinite(intensity_enhancement_int(:))), numel(intensity_enhancement_int));
end

field_calc_time = toc(field_calc_start);
fprintf('\\n[OK] Field calculation completed in %.1f seconds\\n', field_calc_time);
fprintf('================================================================\\n');
"""

        return code
