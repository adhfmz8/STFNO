#!/bin/bash -l

#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --job-name=fno_compile_log
#SBATCH --mail-user=nkdiamond@miners.utep.edu
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'

module load python
module load pytorch

cd $SCRATCH

export PYTHONPATH=$SLURM_SUBMIT_DIR/../../:$PYTHONPATH

# --- LOGGING SETUP START ---

# 1. Enable ALL logging channels (Graph breaks, inductor code, dynamo tracing, etc.)
export TORCH_LOGS="all"

# 2. Enable Debug Mode (dumps generated Triton kernels and graphs to a local folder)
export TORCH_COMPILE_DEBUG=1

# --- LOGGING SETUP END ---

echo "Starting run with Torch Compile logs enabled..."

# Run the script
# We redirect stdout and stderr to a file because TORCH_LOGS="all" produces massive output.
srun python $SLURM_SUBMIT_DIR/main.py > torch_compile.log 2>&1

echo "Job finished."
echo "Logs saved to: ${SCRATCH}/torch_compile.log"
echo "Debug artifacts saved to: ${SCRATCH}/torch_compile_debug/"