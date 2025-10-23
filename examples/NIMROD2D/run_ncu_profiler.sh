#!/bin/bash -l

#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:59:00
#SBATCH --job-name=fno-profile
#SBATCH --mail-user=nkdiamond@miners.utep.edu
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'

module load python
module load pytorch

export PYTHONPATH=$PWD/../../:$PYTHONPATH

NCU_REPORT_FILE="fno_profile.ncu-rep"

ncu --set full --nvtx -o "${NCU_REPORT_FILE}" --force-overwrite python main.py | tee run_output_profile.txt