#!/bin/bash
#SBATCH -p standard
#SBATCH --job-name=temp_seq_nn
#SBATCH --output=temp_seq_nn_%j.log
#SBATCH --error=temp_seq_nn_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
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

python run_satis_conv_search.py