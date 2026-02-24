#!/bin/bash
#SBATCH --job-name=22x47
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

echo "---------- Start simulation: 22x47_00 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_00.py --sim-conf ./config/ho_rod/config_sim_22x47_00.py --verbose

echo "---------- Start simulation: 22x47_01 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_01.py --sim-conf ./config/ho_rod/config_sim_22x47_01.py --verbose

echo "---------- Start simulation: 22x47_03 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_03.py --sim-conf ./config/ho_rod/config_sim_22x47_03.py --verbose

echo "---------- Start simulation: 22x47_05 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_05.py --sim-conf ./config/ho_rod/config_sim_22x47_05.py --verbose

echo "---------- Start simulation: 22x47_07 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_07.py --sim-conf ./config/ho_rod/config_sim_22x47_07.py --verbose

echo "---------- Start simulation: 22x47_10 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_10.py --sim-conf ./config/ho_rod/config_sim_22x47_10.py --verbose

echo "---------- Start simulation: 22x47_15 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_22x47_15.py --sim-conf ./config/ho_rod/config_sim_22x47_15.py --verbose

echo "---------- Start simulation: 22x47_20 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/ho_rod/config_str_22x47_20.py --sim-conf ./config/ho_rod/config_sim_22x47_20.py --verbose

echo "Job finished on $(date)"

