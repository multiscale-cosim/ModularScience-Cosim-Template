killall -9 python3
killall -9 mpirun

export CO_SIM_ROOT_PATH="/home/vagrant/multiscale-cosim-repos/"
export CO_SIM_MODULES_ROOT_PATH="${CO_SIM_ROOT_PATH}/TVB-NEST-usecase1"
export CO_SIM_USE_CASE_ROOT_PATH="${CO_SIM_MODULES_ROOT_PATH}"
export PYTHONPATH=/home/vagrant/nest_installed/lib/python3.8/site-packages:/home/vagrant/multiscale-cosim-repos/TVB-NEST-usecase1

rm -r /home/vagrant/multiscale-cosim-repos/my_forks/TVB-NEST-usecase1/result_sim

python3 ../../main.py --global-settings ../../EBRAINS_WorkflowConfigurations/general/global_settings.xml --action-plan ../../EBRAINS_WorkflowConfigurations/usecase/local/plans/cosim_alpha_brunel_local.xml
