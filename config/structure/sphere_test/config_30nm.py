from pathlib import Path

args = {}

# Refractive index paths (선택사항 - 기본 MNPBEM 파일 사용)
args['refractive_index_paths'] = {
    'gold': '/home/yoojk20/workspace/MNPBEM/Material/@epstable/gold.dat',  # 필요시 커스텀 경로
}

args['structure_name'] = 'simple_au_sphere'
args['structure'] = 'sphere'
args['diameter'] = 30
args['mesh_density'] = 256
args['medium'] = 'vacuum'
args['materials'] = ['gold']
args['use_substrate'] = False
args['custom_materials'] = {}
