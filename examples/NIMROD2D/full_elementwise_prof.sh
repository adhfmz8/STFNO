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

# Load necessary modules
module load python
module load pytorch

# Set Python path
export PYTHONPATH=$PWD/../../:$PYTHONPATH

# Define the output file for the Nsight Compute report
NCU_REPORT_FILE="elementwise_kernel_profile.ncu-rep"

echo "Pausing DCGM..."
dcgmi profile --pause

echo "Running Nsight Compute on Rank 1 Kernel..."

srun ncu --set full \
    --kernel-name-base function \
    --kernel-name "elementwise_kernel" \
    --launch-skip 0 --launch-count 1 \
    -o elementwise_full.ncu-rep --force-overwrite \
    python main.py

echo "Resuming DCGM..."
dcgmi profile --resume

echo "Profiling job complete. Report saved to ${NCU_REPORT_FILE}"