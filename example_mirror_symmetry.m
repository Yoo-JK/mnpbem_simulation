%% EXAMPLE_MIRROR_SYMMETRY
%  AuAg core-shell nanocube homodimer with xy-mirror symmetry
%
%  Structure: Two core-shell nanocubes aligned along z-axis
%  Symmetry: x=0, y=0 planes (xy-symmetry)
%  Excitation: x-polarization and y-polarization, z-propagation
%
%  Note: z=0 plane symmetry is NOT supported by MNPBEM mirror functions

clear; close all;

%% ========================================
%  STEP 1: Define simulation options
%  ========================================

%  IMPORTANT: Add 'sym', 'xy' for mirror symmetry
op = bemoptions( 'sim', 'ret', 'interp', 'curv', 'sym', 'xy' );

%% ========================================
%  STEP 2: Define dielectric functions
%  ========================================

%  1: surrounding medium (air/vacuum)
%  2: Au (shell material)
%  3: Ag (core material)
epstab = { epsconst( 1 ), epstable( 'gold.dat' ), epstable( 'silver.dat' ) };

%% ========================================
%  STEP 3: Define geometry (1/4 of full structure)
%  ========================================

%  Nanocube parameters
cube_size = 50;           % nm, outer cube size
shell_thickness = 5;      % nm
core_size = cube_size - 2*shell_thickness;  % inner core size
gap = 10;                 % nm, gap between two cubes
z_offset = (cube_size + gap) / 2;  % z-position of cube centers

%  Mesh parameters
n_face = 8;  % grid points per face edge

%  ------------------------------------------
%  Create 1/4 of the nanocube (x>0, y>0 region only)
%  ------------------------------------------

%  For xy-symmetry, we only define the x>0, y>0 quadrant
%  The full particle will be created by mirror operations

%  Method 1: Using polygon extrusion with symmetry
%  Create polygon for 1/4 of square (only x>0, y>0 corner)

%  --- Shell (outer cube) ---
%  Quarter square polygon for shell
poly_shell = polygon( 4, 'size', [cube_size, cube_size] );
%  Select only x>0, y>0 part
poly_shell = select( poly_shell, 'index', 1 );  % first quadrant

%  Edge profile (z-direction extrusion)
edge_shell = edgeprofile( cube_size, n_face );

%  Extrude with symmetry - this creates 1/4 of the cube
p_shell_quarter = tripolygon( poly_shell, edge_shell, op, 'sym', 'xy' );

%  --- Core (inner cube) ---
poly_core = polygon( 4, 'size', [core_size, core_size] );
poly_core = select( poly_core, 'index', 1 );

edge_core = edgeprofile( core_size, n_face );
p_core_quarter = tripolygon( poly_core, edge_core, op, 'sym', 'xy' );

%% Alternative Method: Manual creation of 1/4 nanocube
%  This gives more control over the mesh

%  Function to create 1/4 of a cube (x>0, y>0)
%  Only the faces visible in x>0, y>0 quadrant

%  Cube 1 (upper, z > 0)
[p_shell1, p_core1] = create_quarter_coreshell_cube( ...
    cube_size, core_size, n_face, [0, 0, z_offset] );

%  Cube 2 (lower, z < 0)
[p_shell2, p_core2] = create_quarter_coreshell_cube( ...
    cube_size, core_size, n_face, [0, 0, -z_offset] );

%% ========================================
%  STEP 4: Create comparticlemirror object
%  ========================================

%  Particle table (1/4 of each surface)
ptab = { p_shell1, p_core1, p_shell2, p_core2 };

%  Index to dielectric media: [inside, outside]
%  Shell 1: inside=Au(2), outside=air(1)
%  Core 1:  inside=Ag(3), outside=Au(2)
%  Shell 2: inside=Au(2), outside=air(1)
%  Core 2:  inside=Ag(3), outside=Au(2)
inout = [ 2, 1;    % shell 1
          3, 2;    % core 1
          2, 1;    % shell 2
          3, 2 ];  % core 2

%  Closed surfaces (all are closed)
closed = [ 1, 1, 1, 1 ];

%  Create comparticlemirror object
p = comparticlemirror( epstab, ptab, inout, closed, op );

%  Get full particle for visualization
pfull = full( p );

%% ========================================
%  STEP 5: Visualize the structure
%  ========================================

figure;
subplot(1,2,1);
plot( p, 'EdgeColor', 'b' );
title( '1/4 Structure (for BEM)' );
axis equal; view( 30, 30 );

subplot(1,2,2);
plot( pfull, 'EdgeColor', 'b' );
title( 'Full Structure (visualization)' );
axis equal; view( 30, 30 );

%% ========================================
%  STEP 6: Set up BEM solver and excitation
%  ========================================

%  BEM solver with mirror symmetry
bem = bemsolver( p, op );

%  Plane wave excitation
%  pol: [1,0,0] = x-polarization, [0,1,0] = y-polarization
%  dir: [0,0,1] = +z propagation direction
%
%  IMPORTANT: For mirror symmetry, only x and y polarizations are supported
%             z-polarization is NOT allowed
exc = planewave( [1, 0, 0; 0, 1, 0], [0, 0, 1; 0, 0, 1], op );

%% ========================================
%  STEP 7: Run simulation
%  ========================================

%  Wavelength range
enei = linspace( 400, 800, 50 );  % nm

%  Allocate results
n_wav = length( enei );
sca = zeros( n_wav, 2 );  % scattering: [x-pol, y-pol]
ext = zeros( n_wav, 2 );  % extinction: [x-pol, y-pol]
abs_ = zeros( n_wav, 2 ); % absorption: [x-pol, y-pol]

%  Main simulation loop
fprintf( 'Starting simulation with mirror symmetry...\n' );
tic;

for ien = 1 : n_wav
    %  Solve BEM equations
    sig = bem \ exc( p, enei(ien) );

    %  Calculate cross sections
    sca( ien, : ) = exc.sca( sig );
    ext( ien, : ) = exc.ext( sig );
    abs_( ien, : ) = exc.abs( sig );

    %  Progress
    if mod( ien, 10 ) == 0
        fprintf( '  Wavelength %d/%d (%.1f nm)\n', ien, n_wav, enei(ien) );
    end
end

elapsed = toc;
fprintf( 'Simulation completed in %.1f seconds\n', elapsed );

%% ========================================
%  STEP 8: Plot results
%  ========================================

figure;

subplot(2,2,1);
plot( enei, ext(:,1), 'b-', enei, ext(:,2), 'r-', 'LineWidth', 1.5 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Extinction (nm^2)' );
legend( 'x-pol', 'y-pol' );
title( 'Extinction Cross Section' );

subplot(2,2,2);
plot( enei, sca(:,1), 'b-', enei, sca(:,2), 'r-', 'LineWidth', 1.5 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering (nm^2)' );
legend( 'x-pol', 'y-pol' );
title( 'Scattering Cross Section' );

subplot(2,2,3);
plot( enei, abs_(:,1), 'b-', enei, abs_(:,2), 'r-', 'LineWidth', 1.5 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Absorption (nm^2)' );
legend( 'x-pol', 'y-pol' );
title( 'Absorption Cross Section' );

subplot(2,2,4);
%  Due to xy-symmetry, x-pol and y-pol should give same results
%  (for symmetric structures)
plot( enei, ext(:,1) - ext(:,2), 'k-', 'LineWidth', 1.5 );
xlabel( 'Wavelength (nm)' );
ylabel( '\Delta Extinction (nm^2)' );
title( 'x-pol minus y-pol (should be ~0 for symmetric structure)' );

%% ========================================
%  STEP 9: Get surface charges for full particle (optional)
%  ========================================

%  If you need surface charges on the full particle:
sig_at_peak = bem \ exc( p, 550 );  % at 550 nm

%  Expand to full particle
sig_full = full( sig_at_peak );

%  Now sig_full contains surface charges for the complete structure
%  sig_full.sig1, sig_full.sig2 for x-pol and y-pol

%% ========================================
%  Helper function: Create 1/4 of core-shell cube
%  ========================================

function [p_shell, p_core] = create_quarter_coreshell_cube( ...
    outer_size, inner_size, n_grid, center )
%  CREATE_QUARTER_CORESHELL_CUBE
%  Creates 1/4 of a core-shell cube (x>0, y>0 quadrant) for mirror symmetry
%
%  For xy-mirror symmetry, we need faces in the x>0, y>0 region:
%  - Top face (z+): quarter
%  - Bottom face (z-): quarter
%  - Front face (x+): half (y>0 part)
%  - Right face (y+): half (x>0 part)
%
%  The x=0 and y=0 faces are NOT included (they're mirror planes)

%  Outer shell (quarter)
p_shell = create_quarter_cube( outer_size, n_grid, center );

%  Inner core (quarter)
p_core = create_quarter_cube( inner_size, n_grid, center );

end

function p = create_quarter_cube( size, n, center )
%  CREATE_QUARTER_CUBE - Create 1/4 of a cube for xy-mirror symmetry

    half = size / 2;

    %  Vertices for x>0, y>0 quadrant
    %  We need: top quarter, bottom quarter, +x half, +y half

    %  Use particle class to create faces
    %  Top face (z = +half), quarter (x>0, y>0)
    [x, y] = meshgrid( linspace(0, half, n), linspace(0, half, n) );
    z = half * ones( size(x) );
    p_top = particle( x + center(1), y + center(2), z + center(3) );

    %  Bottom face (z = -half), quarter (x>0, y>0)
    z = -half * ones( size(x) );
    p_bot = particle( x + center(1), y + center(2), z + center(3) );
    p_bot = flip( p_bot, 3 );  % flip normal direction

    %  Front face (x = +half), half (y>0)
    [y, z] = meshgrid( linspace(0, half, n), linspace(-half, half, 2*n-1) );
    x = half * ones( size(y) );
    p_front = particle( x + center(1), y + center(2), z + center(3) );

    %  Right face (y = +half), half (x>0)
    [x, z] = meshgrid( linspace(0, half, n), linspace(-half, half, 2*n-1) );
    y = half * ones( size(x) );
    p_right = particle( x + center(1), y + center(2), z + center(3) );

    %  Combine all faces
    p = [ p_top, p_bot, p_front, p_right ];
end
