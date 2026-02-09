#!/bin/bash
#SBATCH --job-name=40x79
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: 40x79_00 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_00.py --sim-conf ./config/ho_rod/config_sim_40x79_00.py --verbose

echo "---------- Start simulation: 40x79_01 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_01.py --sim-conf ./config/ho_rod/config_sim_40x79_01.py --verbose

echo "---------- Start simulation: 40x79_03 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_03.py --sim-conf ./config/ho_rod/config_sim_40x79_03.py --verbose

echo "---------- Start simulation: 40x79_05 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_05.py --sim-conf ./config/ho_rod/config_sim_40x79_05.py --verbose

echo "---------- Start simulation: 40x79_07 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_07.py --sim-conf ./config/ho_rod/config_sim_40x79_07.py --verbose

echo "---------- Start simulation: 40x79_10 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_10.py --sim-conf ./config/ho_rod/config_sim_40x79_10.py --verbose

echo "---------- Start simulation: 40x79_15 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
#./master.sh --str-conf ./config/ho_rod/config_str_40x79_15.py --sim-conf ./config/ho_rod/config_sim_40x79_15.py --verbose

echo "---------- Start simulation: 40x79_20 ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/ho_rod/config_str_40x79_20.py --sim-conf ./config/ho_rod/config_sim_40x79_20.py --verbose

echo "Job finished on $(date)"

