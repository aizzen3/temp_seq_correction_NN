#!/bin/bash
#SBATCH -p fat                # high-RAM nodes
#SBATCH --job-name=qc_run
#SBATCH --output=qc_%j.log    # log file (%j = job id)
#SBATCH --time=3-00:00:00       # max runtime
#SBATCH --mem=512G            # request RAM
#SBATCH --cpus-per-task=8  

set -euo pipefail

cd ~/temp_seq_nn
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "RAM: $SLURM_MEM_PER_NODE"
echo "Python path:"
which python
python --version


python testing_model.py

