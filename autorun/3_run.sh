#!/bin/bash

cd /home/yoojk20/workspace/mnpbem_simulation

echo "******************** Au Dimer / Rounding 0.2 / Gap 0.6 nm ********************"
./master.sh --str-conf ./config/dimer/au/str/r0.2/config_str_au_r0.2_g0.6.py --sim-conf ./config/dimer/au/sim/r0.2/config_sim_au_r0.2_g0.6.py --verbose ;

echo "******************** AuAg Dimer / Rounding 0.2 / Gap 0.6 nm ********************"
./master.sh --str-conf ./config/dimer/auag/str/r0.2/config_str_auag_r0.2_g0.6.py --sim-conf ./config/dimer/auag/sim/r0.2/config_sim_auag_r0.2_g0.6.py --verbose ;

echo "******************** Au Dimer / Rounding 0.2 / Gap 5.0 nm ********************"
./master.sh --str-conf ./config/dimer/au/str/r0.2/config_str_au_r0.2_g5.0.py --sim-conf ./config/dimer/au/sim/r0.2/config_sim_au_r0.2_g5.0.py --verbose 


echo "JOB DONE"
