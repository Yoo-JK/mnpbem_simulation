"""
20nm Gold Nanosphere in Air - Structure Configuration
"""

args = {}

# ============================================================================
# STRUCTURE NAME
# ============================================================================
args['structure_name'] = '20nm_au_sphere_air'

# ============================================================================
# STRUCTURE TYPE
# ============================================================================
args['structure'] = 'sphere'
args['mesh_density'] = 144  # 구 형태에 권장되는 메시 밀도
args['diameter'] = 30  # nm

# ============================================================================
# MATERIALS
# ============================================================================
# Medium (surrounding environment)
args['medium'] = 'vacuum'

# Particle material
args['materials'] = ['gold']  # Au nanosphere

# ============================================================================
# OPTIONAL: Custom Refractive Index
# ============================================================================
# 기본 gold.dat 사용. 필요시 커스텀 파일 경로 지정 가능
# args['refractive_index_paths'] = {
#     'gold': './materials/gold_palik.dat',
# }

# ============================================================================
# OPTIONAL: Substrate
# ============================================================================
args['use_substrate'] = False

# 기판 위에 올리려면:
# args['use_substrate'] = True
# args['substrate'] = {
#     'material': 'glass',
#     'position': -10,  # sphere 중심이 z=0이므로, z=-10에 기판 위치
# }
