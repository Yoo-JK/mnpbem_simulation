#!/bin/bash
#SBATCH --job-name=ag_mono
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: Ag monomer gap ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/dimer/monomer/str/r0.2/config_str_ag_r0.2.py --sim-conf ./config/dimer/monomer/sim/r0.2/config_sim_ag_r0.2.py --verbose

echo "Job finished on $(date)"

