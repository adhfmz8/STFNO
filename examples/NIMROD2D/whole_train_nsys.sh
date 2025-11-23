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
#SBATCH --job-name=fno_nsys_profile
#SBATCH --mail-user=nkdiamond@miners.utep.edu
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'

module load python
module load pytorch

cd $SCRATCH

NSYS_REPORT_FILE="fno_model_profile_compile.nsys-rep"

export PYTHONPATH=$SLURM_SUBMIT_DIR/../../:$PYTHONPATH

echo "Pausing DCGM..."
dcgmi profile --pause

echo "Starting Nsight Systems profiling..."

srun nsys profile --stats=true -t nvtx,cuda -o ${NSYS_REPORT_FILE} --force-overwrite=true \
    python $SLURM_SUBMIT_DIR/main.py

echo "Nsight Systems profiling complete."
echo "Resuming DCGM..."
dcgmi profile --resume

echo "Profiling job finished. Report saved to ${SCRATCH}/${NSYS_REPORT_FILE}"