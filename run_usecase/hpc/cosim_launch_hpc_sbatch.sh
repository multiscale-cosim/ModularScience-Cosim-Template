#!/bin/bash -x
#SBATCH --account=slns
#SBATCH --partition=devel
#SBATCH --output=slurm_logs/cosim_sbatch-out.%j
#SBATCH --error=slurm_logs/cosim_sbatch-err.%j
#SBATCH --nodes=2
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00

module load Stages/2022 GCCcore/.11.2.0 GCC/11.2.0 ParaStationMPI/5.5.0-1 Python/3.9.6 mpi4py/3.1.3 CMake/3.21.1 ZeroMQ/4.3.4

export CO_SIM_ROOT_PATH="${PROJECT_cslns}/${LOGNAME}/multiscale-cosim"
export CO_SIM_MODULES_ROOT_PATH="${CO_SIM_ROOT_PATH}/TVB-NEST-usecase1"
export CO_SIM_USE_CASE_ROOT_PATH="${CO_SIM_ROOT_PATH}/TVB-NEST-usecase1"
export CO_SIM_NEST=${CO_SIM_ROOT_PATH}/nest

export PATH=${CO_SIM_NEST}/bin:${PATH}
export PYTHONPATH=${PYTHONPATH}:${CO_SIM_NEST}/lib64/python3.9/site-packages
export PYTHONPATH=${PYTHONPATH}:${CO_SIM_ROOT_PATH}/site-packages
export PYTHONPATH=${PYTHONPATH}:${CO_SIM_USE_CASE_ROOT_PATH}

srun -n 1 --exact python3 ${CO_SIM_USE_CASE_ROOT_PATH}/main.py --global-settings ${CO_SIM_USE_CASE_ROOT_PATH}/EBRAINS_WorkflowConfigurations/general/global_settings.xml --action-plan ${CO_SIM_USE_CASE_ROOT_PATH}/EBRAINS_WorkflowConfigurations/usecase/hpc/plans/cosim_alpha_brunel_hpc.xml
