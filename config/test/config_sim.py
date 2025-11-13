"""
20nm Gold Nanosphere in Air - Simulation Configuration
"""

import os
from pathlib import Path

args = {}

# ============================================================================
# MNPBEM TOOLBOX PATH (REQUIRED)
# ============================================================================
args['mnpbem_path'] = '/home/yoojk20/workspace/MNPBEM'

# ============================================================================
# SIMULATION NAME
# ============================================================================
args['simulation_name'] = '30nm_au_sphere_air_spectrum'

# ============================================================================
# SIMULATION TYPE
# ============================================================================
# 20nm는 작은 입자이므로 'stat' (quasistatic) 사용 권장
args['simulation_type'] = 'stat'  # 빠르고 충분히 정확

# 더 정확한 계산을 원하면:
# args['simulation_type'] = 'ret'  # 전체 Maxwell 방정식 사용

args['interp'] = 'curv'  # 곡면 경계 요소 (더 정확)
args['waitbar'] = 0

# ============================================================================
# EXCITATION TYPE - Plane Wave
# ============================================================================
args['excitation_type'] = 'planewave'

# Polarization directions
args['polarizations'] = [
    [1, 0, 0],  # x-편광
    [0, 1, 0],  # y-편광
    [0, 0, 1],  # z-편광
]

# Propagation directions
args['propagation_dirs'] = [
    [0, 0, 1],  # z 방향으로 전파
    [0, 0, 1],
    [0, 1, 0],  # y 방향으로 전파 (z-편광용)
]

# ============================================================================
# WAVELENGTH RANGE
# ============================================================================
# Au nanosphere의 일반적인 플라즈몬 공명 영역
args['wavelength_range'] = [400, 800, 100]  # 400-800 nm, 100 포인트

# 더 넓은 범위:
# args['wavelength_range'] = [300, 1000, 140]

# ============================================================================
# OUTPUT TYPE
# ============================================================================
args['output_types'] = ['extinction', 'scattering', 'absorption']

# ============================================================================
# NUMERICAL SETTINGS
# ============================================================================
args['refine'] = 2  # 적분 정밀도
args['relcutoff'] = 2  # 상대 cutoff

# ============================================================================
# FIELD CALCULATION (optional)
# ============================================================================
args['calculate_fields'] = False  # 스펙트럼만 필요하면 False

# 전기장 분포가 필요하면:
# args['calculate_fields'] = True
# args['field_wavelengths'] = [520]  # 공명 파장에서 계산
# args['field_region'] = {
#     'x_range': [-50, 50, 101],
#     'y_range': [-50, 50, 101],
#     'z_range': [0, 0, 1]  # z=0 평면
# }

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
args['output_dir'] = os.path.join(Path.home(), 'research/mnpbem/20nm_au_sphere')
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi'] = 300
args['spectrum_xaxis'] = 'wavelength'  # 또는 'energy'

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = False
args['use_nonlocality'] = False

args['matlab_executable'] = 'matlab'
args['matlab_options'] = '-nodisplay -nosplash -nodesktop'
