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
#SBATCH --job-name=fno-batch-sweep
#SBATCH --mail-user=nkdiamond@miners.utep.edu
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'

#SBATCH --array=0-5

#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err

# The index corresponds to the SLURM_ARRAY_TASK_ID
BATCH_SIZES=(8 16 32 64 128 256)

CURRENT_BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

mkdir -p slurm_out
mkdir -p logs

module load python
module load pytorch

export PYTHONPATH=$PWD/../../:$PYTHONPATH

echo "========================================================"
echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID}"
echo "Batch Size for this run: ${CURRENT_BATCH_SIZE}"
echo "========================================================"

python main.py --batch_size $CURRENT_BATCH_SIZE | tee logs/run_output_bs_${CURRENT_BATCH_SIZE}.txt