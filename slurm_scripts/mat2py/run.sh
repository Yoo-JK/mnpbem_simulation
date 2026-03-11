#!/bin/bash
#SBATCH --job-name=mat2py
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: (STAT) AuNR 22x47_sub ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/mat2py/rod/config_str_aunr_22x47_sub.py --sim-conf ./config/mat2py/rod/config_sim_stat_aunr_22x47_sub.py --verbose

echo "---------- Start simulation: (RET) AuNR 22x47_sub ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/mat2py/rod/config_str_aunr_22x47_sub.py --sim-conf ./config/mat2py/rod/config_sim_ret_aunr_22x47_sub.py --verbose

echo "---------- Start simulation: (STAT) AuNS 30_sub ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/mat2py/sphere/config_str_au_30_sub.py --sim-conf ./config/mat2py/sphere/config_sim_stat_au_30_sub.py --verbose

echo "---------- Start simulation: (RET) AuNS 30_sub ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/mat2py/sphere/config_str_au_30_sub.py --sim-conf ./config/mat2py/sphere/config_sim_ret_au_30_sub.py --verbose

echo "Job finished on $(date)"

