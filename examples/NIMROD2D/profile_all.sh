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

# Pause DCGM profiling
echo "Pausing DCGM..."
dcgmi profile --pause

# Profile the first kernel
echo "Running Nsight Compute on ampere_sgemm_32x32_sliced1x4_nt Kernel..."
srun ncu --kernel-name "ampere_sgemm_32x32_sliced1x4_nt" \
    --launch-skip 10 --launch-count 1 \
    --set roofline \
    -o ampere_sgemm_profile.ncu-rep --force-overwrite \
    python main.py

# Profile the second kernel
echo "Running Nsight Compute on reduce_kernel Kernel..."
srun ncu --kernel-name "reduce_kernel" \
    --launch-skip 10 --launch-count 1 \
    --set roofline \
    -o reduce_profile.ncu-rep --force-overwrite \
    python main.py

# Profile the third kernel
echo "Running Nsight Compute on vectorized_elementwise_kernel Kernel..."
srun ncu --kernel-name "vectorized_elementwise_kernel" \
    --launch-skip 10 --launch-count 1 \
    --set roofline \
    -o vectorized_elementwise_profile.ncu-rep --force-overwrite \
    python main.py

# Resume DCGM profiling
echo "Resuming DCGM..."
dcgmi profile --resume

echo "Profiling job complete. Reports saved to ampere_sgemm_profile.ncu-rep, reduce_profile.ncu-rep, and vectorized_elementwise_profile.ncu-rep"