#!/bin/bash

cd /home/yoojk20/workspace/mnpbem_simulation


echo "***************************************** Au rod (Small and Big) with table refractive index *****************************************"
./master.sh --str-conf ./config/rod/au/au100cu0/config_str_au.py --sim-conf ./config/rod/au/au100cu0/config_sim_au.py --verbose
./master.sh --str-conf ./config/rod/au/au100cu0/config_str_auau.py --sim-conf ./config/rod/au/au100cu0/config_sim_auau.py --verbose
./master.sh --str-conf ./config/rod/au/au100cu0/config_str_auau_cs.py --sim-conf ./config/rod/au/au100cu0/config_sim_auau_cs.py --verbose

echo "***************************************** Au rod (Small and Big) with johnson refractive index *****************************************"
#./master.sh --str-conf ./config/rod/au/johnson/config_str_au.py --sim-conf ./config/rod/au/johnson/config_sim_au.py --verbose
#./master.sh --str-conf ./config/rod/au/johnson/config_str_auau.py --sim-conf ./config/rod/au/johnson/config_sim_auau.py --verbose

echo "***************************************** Au@AuCu alloy rod with table refractive index *****************************************"
./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au90cu10.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au90cu10.py --verbose
./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au99cu01.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au99cu01.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au98cu02.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au98cu02.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au97cu03.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au97cu03.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au96cu04.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au96cu04.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au95cu05.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au95cu05.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au94cu06.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au94cu06.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au93cu07.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au93cu07.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au92cu08.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au92cu08.py --verbose
#./master.sh --str-conf ./config/rod/aucu/au100cu0/config_str_au91cu09.py --sim-conf ./config/rod/aucu/au100cu0/config_sim_au91cu09.py --verbose

echo "***************************************** Au@AuCu alloy rod with table refractive index *****************************************"
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au90cu10.py --sim-conf ./config/rod/aucu/johnson/config_sim_au90cu10.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au99cu01.py --sim-conf ./config/rod/aucu/johnson/config_sim_au99cu01.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au98cu02.py --sim-conf ./config/rod/aucu/johnson/config_sim_au98cu02.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au97cu03.py --sim-conf ./config/rod/aucu/johnson/config_sim_au97cu03.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au96cu04.py --sim-conf ./config/rod/aucu/johnson/config_sim_au96cu04.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au95cu05.py --sim-conf ./config/rod/aucu/johnson/config_sim_au95cu05.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au94cu06.py --sim-conf ./config/rod/aucu/johnson/config_sim_au94cu06.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au93cu07.py --sim-conf ./config/rod/aucu/johnson/config_sim_au93cu07.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au92cu08.py --sim-conf ./config/rod/aucu/johnson/config_sim_au92cu08.py --verbose
#./master.sh --str-conf ./config/rod/aucu/johnson/config_str_au91cu09.py --sim-conf ./config/rod/aucu/johnson/config_sim_au91cu09.py --verbose

echo "JOB DONE"
