# CTP (Charge Transfer Plasmon) Simulation Guide

## 자동 Conductive Junction 기능

이제 `gap = 0.0` (또는 `gap <= 0`)으로 설정하면 **자동으로 conductive junction**이 활성화되어 CTP 시뮬레이션이 가능합니다.

---

## 물리적 배경

### Capacitive Coupling (gap > 0)
- **물리**: 입자들이 분리되어 정전기적으로만 결합
- **전하**: 각 입자가 독립적인 전위를 가짐
- **MNPBEM**: `comparticle(..., 1, 2, ..., op)` - 각각 별도 인자
- **관측**: Bonding/antibonding plasmon 모드

### Conductive Junction (gap ≤ 0)
- **물리**: 입자들이 접촉하여 금속 연결 (ohmic contact)
- **전하**: 모든 입자가 하나의 equipotential surface
- **MNPBEM**: `comparticle(..., [1, 2, ...], op)` - 배열로 전달
- **관측**: **Charge Transfer Plasmon (CTP)** 모드
  - 적색편이된 Bonding Dipole Plasmon (BDP)
  - Gap이 작을수록 강한 효과

---

## 사용 방법

### 1. Dimer 구조 (가장 간단)

```python
# config/structure/config_structure.py
args = {}
args['structure'] = 'dimer_sphere'  # 또는 'dimer_cube'
args['diameter'] = 50  # nm
args['gap'] = 0.0  # ⭐ 접촉 → 자동으로 conductive junction 활성화
args['materials'] = ['gold']
args['medium'] = 'water'
```

**생성되는 MATLAB 코드:**
```matlab
% Gap = 0.0 → Conductive junction (자동 감지)
closed = [1, 2];
p = comparticle(epstab, particles, inout, [1, 2], op);
```

### 2. Sphere Cluster (N개 구)

```python
# config/structure/config_structure.py
args['structure'] = 'sphere_cluster_aggregate'
args['n_spheres'] = 7
args['diameter'] = 50
args['gap'] = 0.0  # ⭐ 모든 구가 하나의 conductive surface로 연결
args['materials'] = ['gold']
```

**생성되는 MATLAB 코드:**
```matlab
% Gap = 0.0 → 7개 구가 모두 연결 (CTP mode)
closed = [1, 2, 3, 4, 5, 6, 7];
p = comparticle(epstab, particles, inout, [1, 2, 3, 4, 5, 6, 7], op);
```

### 3. Advanced Dimer (복잡한 구조)

```python
# config/structure/config_structure.py
args['structure'] = 'advanced_dimer_cube'
args['core_size'] = 30
args['shell_layers'] = [5, 3]
args['materials'] = ['gold', 'silver', 'agcl']
args['gap'] = 0.0  # ⭐ Conductive junction
```

---

## Gap 값에 따른 동작

| Gap (nm) | 모드 | Comparticle 호출 | 설명 |
|----------|------|------------------|------|
| **5.0** | Capacitive | `(..., 1, 2, op)` | 분리된 입자들, 정전기적 결합 |
| **1.0** | Capacitive | `(..., 1, 2, op)` | 강한 정전기적 결합 |
| **0.5** | Capacitive | `(..., 1, 2, op)` | 매우 강한 결합, 양자 효과 시작 |
| **0.0** | **Conductive** | `(..., [1, 2], op)` | ⭐ **CTP 모드** - 금속 접촉 |
| **-0.1** | **Conductive** | `(..., [1, 2], op)` | ⭐ **CTP 모드** - 약간 겹침 |

---

## 비교 실험 예제

### Capacitive vs Conductive 스펙트럼 비교

```python
# Test 1: Capacitive (gap > 0)
args['structure_name'] = 'au_dimer_capacitive'
args['gap'] = 1.0
# → comparticle(..., 1, 2, op)

# Test 2: Conductive (gap = 0)
args['structure_name'] = 'au_dimer_conductive'
args['gap'] = 0.0
# → comparticle(..., [1, 2], op)
```

**예상 결과:**
- Capacitive → Conductive 전환 시:
  - 주 공명 피크 **적색편이**
  - Field enhancement 패턴 변화
  - BDP (Bonding Dipole Plasmon) 모드 출현

---

## Quantum Correction과의 차이

### Classical CTP (현재 구현)
```python
args['gap'] = 0.0
# → Conductive junction만 사용
```
- **물리**: Ohmic contact를 통한 전하 이동
- **범위**: 모든 gap size (특히 gap = 0)
- **관측**: Classical BDP 모드
- **계산**: 표준 BEM, 빠름

### Quantum CTP (Nonlocal QCM)
```python
args['gap'] = 0.0
args['use_nonlocality'] = True
```
- **물리**: Electron tunneling (양자 효과)
- **범위**: Gap < 1 nm에서 중요
- **관측**: 진정한 quantum CTP (적외선 영역)
- **계산**: Nonlocal correction, 느림

### 조합 (최고 정밀도)
```python
args['gap'] = 0.0  # → Conductive junction (자동)
args['use_nonlocality'] = True  # → Quantum tunneling
```
- Classical conductive + Quantum tunneling
- Sub-nanometer gap에서 가장 정확

---

## 확인 방법

### 1. Verbose 모드로 확인

```bash
python run_simulation.py -v
```

**출력 예시:**
```
=== Generating Closed Surfaces ===
  Auto-detected gap <= 0: Using conductive junction (CTP mode)
  Closed surfaces: [1, 2]
```

### 2. 생성된 MATLAB 코드 확인

```bash
cat matlab_output/[structure_name]_geometry.m
```

**Capacitive (gap > 0):**
```matlab
closed = [1, 2];
p = comparticle(epstab, particles, inout, 1, 2, op);
```

**Conductive (gap ≤ 0):**
```matlab
closed = [1, 2];
p = comparticle(epstab, particles, inout, [1, 2], op);
```

### 3. 스펙트럼 확인

- **적색편이**: Capacitive → Conductive 전환 시 관측
- **피크 강도**: Field enhancement 패턴 변화
- **새로운 모드**: BDP 모드 출현 (낮은 에너지)

---

## 지원되는 구조체

자동 conductive junction을 지원하는 구조체:

✅ `dimer_sphere` - 구 dimer
✅ `dimer_cube` - 큐브 dimer
✅ `dimer_core_shell_cube` - Core-shell 큐브 dimer
✅ `sphere_cluster_aggregate` - N개 구 cluster
✅ `advanced_dimer_cube` - Multi-shell dimer

❌ 지원 안 됨 (gap 파라미터 없음):
- Single particle 구조체 (`sphere`, `cube`, `rod`, etc.)
- Core-shell single particle
- DDA shape file

---

## 주의사항

### 1. 물리적 의미
- Gap = 0: **접촉**하지만 여전히 두 개의 mesh
- Gap < 0: **겹침** (metallic junction을 모사)
- **Nonlocal 없이도 CTP 효과 관측 가능** (classical limit)

### 2. Mesh 요구사항
- Gap = 0에서는 **높은 mesh density** 권장
  - Sphere: `mesh_density ≥ 144`
  - Cube: `mesh_density ≥ 20`
- Contact point 주변의 정밀한 mesh 필요

### 3. 계산 시간
- Conductive junction: 계산 시간은 capacitive와 동일
- Matrix size는 입자 개수와 mesh density에 의존

---

## 예제 시뮬레이션

### 완전한 예제: Gold Dimer Contact CTP

**Structure Config:**
```python
# config/structure/config_structure.py
args = {}
args['structure_name'] = 'au_dimer_contact_ctp'
args['structure'] = 'dimer_sphere'
args['diameter'] = 50
args['gap'] = 0.0  # ⭐ Conductive junction 자동 활성화
args['materials'] = ['gold']
args['medium'] = 'water'
args['mesh_density'] = 144
```

**Simulation Config:**
```python
# config/simulation/config_simulation.py
args['simulation_name'] = 'au_dimer_ctp_test'
args['simulation_type'] = 'stat'
args['wavelengths'] = [400, 1000, 120]  # BDP는 적색편이

# Excitation
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0]]  # Dimer axis
args['propagation_dirs'] = [[0, 0, 1]]

# Options
args['npol'] = 15
args['calculate_cross_sections'] = True
```

**실행:**
```bash
python run_simulation.py
```

**결과:**
- `extinction_spectrum.png`: Capacitive vs Conductive 비교 시 적색편이 관측
- `field_distribution.png`: Gap 주변의 field 패턴 변화

---

## 참고문헌

- **Classical CTP**: Atay et al., Nano Lett. 4, 1627 (2004)
- **Quantum CTP**: Zuloaga et al., Nano Lett. 9, 887 (2009)
- **MNPBEM**: Hohenester & Trügler, Comput. Phys. Commun. 183, 370 (2012)

---

## 문제 해결

### Q: Gap = 0인데 여전히 capacitive처럼 동작합니다
**A:** Verbose 모드로 확인하세요:
```bash
python run_simulation.py -v
```
"Using conductive junction" 메시지가 나오는지 확인

### Q: Conductive junction을 명시적으로 끄고 싶습니다
**A:** 현재는 gap <= 0이면 자동 활성화됩니다. Gap > 0 (예: 0.1 nm)로 설정하세요.

### Q: Nonlocal과 conductive를 같이 써야 하나요?
**A:** 아니오.
- Gap = 0만: Classical CTP (빠름)
- Gap = 0 + Nonlocal: Quantum CTP (정밀, 느림)

### Q: 어떤 gap size에서 CTP가 나타나나요?
**A:**
- Classical CTP: Gap = 0에서 최대
- Quantum CTP: Gap < 0.5 nm에서 중요
- 일반적으로 gap이 작을수록 강한 CTP 효과
