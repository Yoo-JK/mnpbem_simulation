#!/bin/bash
#SBATCH --job-name=nl
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --mem 64G

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: nonlocal ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./slurm_scripts/dimer/test/config_str.py --sim-conf ./slurm_scripts/dimer/test/config_sim_nl.py --verbose

echo "Job finished on $(date)"

