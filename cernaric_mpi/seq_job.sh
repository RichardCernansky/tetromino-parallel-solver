#!/bin/bash
# ============================================================================
# Slurm script for sequential / OpenMP-only baseline
#
# Usage:
#   sbatch -p arm_fast -c 48 seq_job.sh          # quick test
#   sbatch -p arm_long -c 48 seq_job.sh          # medium run
#   sbatch -p arm_serial -c 48 seq_job.sh        # long run
# ============================================================================

#SBATCH --job-name=cernaric_seq_mpi
#SBATCH --output="logs/%x-%J.out"
#SBATCH --error=logs/"%x-%J.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /etc/profile.d/zz-cray-pe.sh
module load cray-mvapich2_pmix_nogpu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export MV2_ENABLE_AFFINITY=0

echo "=== Baseline run ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"
echo "===================="

# nprocs=1 triggers the pure-OpenMP fallback path, cutoff_depth:=4
srun --ntasks=1 ./cernaric_mpi mapa5_11.txt $OMP_NUM_THREADS 4

exit 0