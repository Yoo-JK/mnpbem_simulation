#!/bin/bash
#SBATCH --job-name=nosub_auctab
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: Au-CTAB ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/rod_ctab/rod_nosub/config_str_auctab.py --sim-conf ./config/rod_ctab/rod_nosub/config_sim_auctab.py --verbose

echo "Job finished on $(date)"

