
#!/bin/bash

echo "Job started on $(date)"

echo "---------- Postprocess: Aggregation 1 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_1_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_1_agg.py --verbose

echo "---------- Postprocess: Aggregation 2 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_2_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_2_agg.py --verbose

echo "---------- Postprocess: Aggregation 3 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_3_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_3_agg.py --verbose

echo "---------- Postprocess: Aggregation 4 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_4_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_4_agg.py --verbose

echo "---------- Postprocess: Aggregation 5 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_5_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_5_agg.py --verbose

echo "---------- Postprocess: Aggregation 6 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_6_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_6_agg.py --verbose

echo "---------- Postprocess: Aggregation 7 Sphere(s) without substrate ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
python run_postprocess.py --str-conf ./config/agg_sph_t2/wo_sub/config_str_7_agg.py --sim-conf ./config/agg_sph_t2/wo_sub/config_sim_7_agg.py --verbose

echo "Job finished on $(date)"

