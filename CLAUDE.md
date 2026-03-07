# CLAUDE.md

## Project Overview

MNPBEM Simulation is a Python framework that generates MATLAB code for simulating electromagnetic properties of metallic nanoparticles using the MNPBEM toolbox. It follows a three-stage pipeline: **Configuration** (Python dicts) → **MATLAB Code Generation** (Python generates .m scripts) → **Post-processing** (Python analyzes MATLAB results).

## Repository Structure

```
mnpbem_simulation/
├── run_simulation.py              # Entry point: generates MATLAB code
├── run_postprocess.py             # Entry point: analyzes results
├── master.sh                      # Bash orchestrator (generate → run MATLAB → postprocess)
├── config/
│   ├── structure/
│   │   ├── config_structure.py    # Nanoparticle geometry config (args dict)
│   │   └── guide_structure.txt    # Comprehensive structure parameter guide
│   └── simulation/
│       └── config_simulation.py   # Simulation parameters config (args dict)
├── simulation/
│   ├── calculate.py               # SimulationManager orchestrator
│   └── sim_utils/
│       ├── geometry_generator.py  # MATLAB geometry code generation
│       ├── material_manager.py    # Material/dielectric function definitions
│       ├── matlab_code_generator.py # Full MATLAB script assembly
│       ├── nonlocal_generator.py  # Quantum correction code
│       └── refractive_index_loader.py # Material data file loading
└── postprocess/
    ├── postprocess.py             # PostprocessManager orchestrator
    ├── plot_field_maps.py         # Standalone field map plotter
    └── post_utils/
        ├── data_loader.py         # Load MATLAB .mat results
        ├── spectrum_analyzer.py   # Peak detection, FWHM, unpolarized spectra
        ├── visualizer.py          # Plots (spectra, field maps, hotspots)
        ├── field_analyzer.py      # EM field analysis, hotspot detection
        ├── field_exporter.py      # Export field data to JSON
        ├── data_exporter.py       # Export spectra to TXT/CSV/JSON
        ├── edge_filter.py         # Filter numerical edge artifacts
        └── geometry_cross_section.py # Geometric cross-section calculations
```

## How to Run

### Environment Setup
```bash
conda create -n mnpbem python=3.11
conda activate mnpbem
conda install numpy scipy matplotlib
```

### Dependencies
- **Python 3.8+** with NumPy, SciPy, Matplotlib
- **MATLAB** with the MNPBEM toolbox (required for simulation execution)

### Running the Full Pipeline
```bash
bash master.sh --str-conf config/structure/config_structure.py \
               --sim-conf config/simulation/config_simulation.py \
               --verbose
```

### Running Individual Steps
```bash
# Step 1: Generate MATLAB code
python run_simulation.py --str-conf <structure_config> --sim-conf <simulation_config> --verbose

# Step 2: Run MATLAB (handled by master.sh)

# Step 3: Post-process results
python run_postprocess.py --str-conf <structure_config> --sim-conf <simulation_config> --verbose
```

## Architecture

### Data Flow
```
Config files (Python dicts with `args = {}`)
  → SimulationManager (validates, generates)
    → MATLAB script (.m file in run folder)
      → MATLAB execution (MNPBEM toolbox)
        → Results (.mat file)
          → PostprocessManager (loads, analyzes, plots, exports)
            → Plots (PNG/PDF) + Data (TXT/CSV/JSON)
```

### Key Design Patterns

1. **Configuration-as-Code**: Configs are Python files containing an `args = {}` dict, loaded via `exec()`. This allows computed values and imports in configs.

2. **Manager/Orchestrator**: `SimulationManager` and `PostprocessManager` each coordinate multiple specialized utility classes.

3. **Code Generation**: Python generates complete MATLAB scripts as strings rather than using direct Python-MATLAB bindings. Each sub-module (`GeometryGenerator`, `MaterialManager`, `MatlabCodeGenerator`) produces MATLAB code fragments that are assembled into a full script.

### Configuration System

Both config files define an `args` dictionary. Structure and simulation configs are merged at runtime (simulation overrides on key collision).

**Structure config** (`config_structure.py`): Defines geometry type, materials, dimensions, mesh density, substrate settings.

**Simulation config** (`config_simulation.py`): Defines simulation type (`stat`/`ret`), excitation type (`planewave`/`dipole`/`eels`), wavelength range, field calculation settings, output formats, parallel computing options.

See `config/structure/guide_structure.txt` for the full parameter reference.

### Supported Structures
Single particles (sphere, cube, rod, ellipsoid, triangle), core-shell variants, dimers, advanced multi-shell dimers, sphere clusters, and DDA shape file imports (`from_shape`).

## Code Conventions

- **Classes**: CamelCase (`SimulationManager`, `GeometryGenerator`)
- **Functions/variables**: snake_case (`generate_geometry`, `load_config`)
- **Docstrings**: Google-style with Args/Returns sections
- **Error handling**: `FileNotFoundError` for missing files, `ValueError` for invalid params; traceback printing in verbose mode
- **No formal test suite**: No pytest/unittest tests exist
- **No linting config**: No .flake8, pylintrc, or pyproject.toml

## Key Files for Common Tasks

| Task | Key Files |
|------|-----------|
| Add new geometry type | `simulation/sim_utils/geometry_generator.py`, `config/structure/guide_structure.txt` |
| Add new material | `simulation/sim_utils/material_manager.py` |
| Modify MATLAB output | `simulation/sim_utils/matlab_code_generator.py` |
| Add new analysis | `postprocess/post_utils/spectrum_analyzer.py` or `field_analyzer.py` |
| Add new plot type | `postprocess/post_utils/visualizer.py` |
| Add new export format | `postprocess/post_utils/data_exporter.py` or `field_exporter.py` |
| Change pipeline flow | `master.sh`, `run_simulation.py`, `run_postprocess.py` |
| Add config parameters | `config/structure/config_structure.py` or `config/simulation/config_simulation.py` |

## Important Notes

- The `exec()` pattern for config loading means config files execute as Python code - be cautious with untrusted configs.
- Run folders are auto-generated with unique names under the configured `output_dir`.
- `master.sh` extracts the run folder path from `run_simulation.py` stdout (`RUN_FOLDER=...` line).
- The postprocess `run()` method returns a tuple of 2 or 3 values: `(data, analysis[, field_analysis])`.
- Large modules exist: `matlab_code_generator.py` (~4,000 lines), `geometry_generator.py` (~3,900 lines), `visualizer.py` (~2,400 lines).
