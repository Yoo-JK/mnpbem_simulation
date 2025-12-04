# Python Backend Guide for MNPBEM Simulation

## 개요

이제 MNPBEM 시뮬레이션을 **MATLAB 없이** 순수 Python으로 실행할 수 있습니다!
[pyMNPBEM](https://github.com/Yoo-JK/pyMNPBEM)을 사용하여 기존 설정 파일을 그대로 사용하면서 Python에서 직접 BEM 계산을 수행합니다.

## 주요 특징

✅ **선택 가능한 Backend**: 설정에서 `backend = 'python'` 또는 `'matlab'` 선택
✅ **기존 설정 파일 호환**: 구조/시뮬레이션 설정 파일 그대로 사용
✅ **MATLAB 불필요**: Python backend 사용 시 MATLAB 설치 불필요
✅ **동일한 출력 형식**: 기존 MATLAB 출력과 호환되는 결과 파일
✅ **빠른 실행**: 코드 생성 없이 직접 계산 수행

## 설치

### 1. 필수 패키지 설치

```bash
pip install numpy scipy matplotlib tqdm
```

### 2. pyMNPBEM 설치

```bash
# pyMNPBEM 클론
git clone https://github.com/Yoo-JK/pyMNPBEM.git

# pyMNPBEM을 Python path에 추가
# 방법 1: Symbolic link 생성 (추천)
ln -s /path/to/pyMNPBEM /path/to/mnpbem

# 방법 2: 환경 변수 설정
export PYTHONPATH=/path/to/pyMNPBEM:$PYTHONPATH

# 방법 3: pip editable 설치 (optional)
# cd pyMNPBEM && pip install -e .
```

## 사용 방법

### Backend 선택

`config/simulation/config_simulation.py`에서 backend 설정:

```python
# Python backend 사용 (MATLAB 불필요)
args['backend'] = 'python'
args['pymnpbem_path'] = None  # PYTHONPATH 사용

# 또는 직접 경로 지정
# args['pymnpbem_path'] = '/path/to/pyMNPBEM'

# MATLAB backend 사용 (기존 방식)
# args['backend'] = 'matlab'
# args['mnpbem_path'] = '/path/to/MNPBEM'
```

### 시뮬레이션 실행

#### Python Backend 사용

```bash
# pyMNPBEM이 PYTHONPATH에 있는 경우
python run_simulation.py \
    --str-conf config/structure/config_structure.py \
    --sim-conf config/simulation/config_simulation.py

# 또는 PYTHONPATH 직접 지정
PYTHONPATH=/path/to/pyMNPBEM python run_simulation.py \
    --str-conf config/structure/config_structure.py \
    --sim-conf config/simulation/config_simulation.py
```

#### MATLAB Backend 사용 (기존 방식)

```bash
# 설정에서 backend = 'matlab' 설정 후
python run_simulation.py \
    --str-conf config/structure/config_structure.py \
    --sim-conf config/simulation/config_simulation.py

# 그 다음 master.sh로 MATLAB 실행
./master.sh
```

## 지원되는 구조

Python backend는 다음 구조들을 지원합니다:

### ✅ 완전 지원
- **Single particles**: sphere, cube, rod, ellipsoid
- **Core-shell**: core_shell_sphere, core_shell_cube
- **Dimers**: dimer_sphere, dimer_cube

### 🚧 부분 지원
- **advanced_dimer_cube**: 단순화된 버전으로 동작 (전체 transformation 기능은 추후 추가 예정)

### 📝 향후 추가 예정
- sphere_cluster_aggregate
- from_shape (DDA 파일)
- substrate 지원
- nonlocal 효과

## 출력 파일

Python backend는 다음 파일들을 생성합니다:

```
output_dir/simulation_name/
├── config_snapshot.py          # 사용된 설정 스냅샷
├── cross_sections.txt          # 스펙트럼 데이터 (MATLAB 호환 형식)
├── results.json                # JSON 형식 결과
├── results.npz                 # NumPy 압축 형식
└── logs/                       # 로그 디렉토리
```

### 결과 파일 형식

#### cross_sections.txt
```
# Wavelength(nm) Scattering(nm^2) Absorption(nm^2) Extinction(nm^2)
400.000000 4.014584e+04 0.000000e+00 4.014584e+04
404.040404 3.866119e+04 0.000000e+00 3.866119e+04
...
```

#### results.json
```json
{
  "wavelengths": [400, 404.04, ...],
  "scattering": [[...], [...]],
  "absorption": [[...], [...]],
  "extinction": [[...], [...]],
  "config": {...}
}
```

## 예시: Gold Sphere 시뮬레이션

### 1. 구조 설정 (config/structure/config_structure.py)

```python
args = {}

args['structure_name'] = 'gold_sphere_50nm'
args['structure'] = 'sphere'
args['diameter'] = 50  # nm
args['mesh_density'] = 144
args['materials'] = ['gold']
args['medium'] = 'vacuum'
```

### 2. 시뮬레이션 설정 (config/simulation/config_simulation.py)

```python
args = {}

# Backend 선택
args['backend'] = 'python'
args['pymnpbem_path'] = None  # PYTHONPATH 사용

# 시뮬레이션 설정
args['simulation_name'] = 'gold_sphere_spectrum'
args['simulation_type'] = 'stat'
args['excitation_type'] = 'planewave'
args['wavelength_range'] = [400, 800, 100]
args['polarizations'] = [[1, 0, 0]]
args['propagation_dirs'] = [[0, 0, 1]]

# 출력 설정
args['output_dir'] = '/path/to/output'
```

### 3. 실행

```bash
PYTHONPATH=/path/to/pyMNPBEM python run_simulation.py \
    --str-conf config/structure/config_structure.py \
    --sim-conf config/simulation/config_simulation.py
```

### 4. 결과

```
============================================================
Python BEM Simulation Complete
============================================================
Structure:        gold_sphere_50nm
Structure type:   sphere
Simulation:       gold_sphere_spectrum
Simulation type:  stat
Excitation:       planewave
Wavelength range: 400-800 nm (100 points)
Run folder:       /path/to/output/gold_sphere_spectrum

Backend:          Python (pyMNPBEM)
============================================================
```

## 결과 분석

### Python으로 결과 읽기

```python
import numpy as np
import matplotlib.pyplot as plt

# NumPy 형식 읽기
data = np.load('results.npz')
wavelengths = data['wavelengths']
scattering = data['scattering']
extinction = data['extinction']

# 플롯
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, scattering[:, 0], label='Scattering')
plt.plot(wavelengths, extinction[:, 0], label='Extinction')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Cross Section (nm²)')
plt.legend()
plt.grid(True)
plt.show()
```

### JSON으로 읽기

```python
import json

with open('results.json', 'r') as f:
    results = json.load(f)

wavelengths = results['wavelengths']
scattering = results['scattering']
```

## Backend 비교

| 특징 | Python Backend | MATLAB Backend |
|------|----------------|----------------|
| **속도** | ⚡ 빠름 (직접 실행) | 🐌 느림 (코드 생성 + 실행) |
| **설치** | ✅ Python 패키지만 | ❌ MATLAB 라이센스 필요 |
| **호환성** | 🔄 기존 설정 파일 사용 | ✅ 완전 호환 |
| **기능** | 📊 기본 기능 지원 | 🎯 전체 기능 지원 |
| **디버깅** | 🐛 Python 디버거 사용 | 📝 MATLAB 디버거 필요 |
| **확장성** | 🔧 Python 코드로 확장 | 📜 MATLAB 코드 생성 |

## 문제 해결

### ImportError: No module named 'mnpbem'

```bash
# pyMNPBEM 경로 확인
ls /path/to/pyMNPBEM/__init__.py

# PYTHONPATH 설정
export PYTHONPATH=/path/to/pyMNPBEM:$PYTHONPATH

# 또는 symbolic link 생성
ln -s /path/to/pyMNPBEM /usr/local/lib/python3.x/site-packages/mnpbem
```

### ModuleNotFoundError: No module named 'tqdm'

```bash
pip install tqdm
```

### ComplexWarning during simulation

이것은 pyMNPBEM의 정상적인 경고입니다. 결과에는 영향을 주지 않습니다.

## 향후 개발 계획

- [ ] Substrate 지원
- [ ] Nonlocal 효과 구현
- [ ] Field 계산 추가
- [ ] Advanced dimer 전체 기능 구현
- [ ] Sphere cluster aggregate 지원
- [ ] DDA shape 파일 import
- [ ] 병렬 계산 최적화
- [ ] 진행 상황 표시 개선

## 기여

버그 리포트나 기능 제안은 GitHub Issue로 제출해주세요.

## 라이센스

이 프로젝트는 원본 MNPBEM과 pyMNPBEM의 라이센스를 따릅니다.

---

**주의**: Python backend는 현재 기본 기능만 지원합니다. 복잡한 시뮬레이션이나 고급 기능이 필요한 경우 MATLAB backend를 사용하세요.
