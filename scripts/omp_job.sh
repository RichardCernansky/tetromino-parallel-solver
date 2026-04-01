#!/bin/bash
# ============================================================================
# Slurm script for OpenMP-only runs
#
# Usage:
#   sbatch -p arm_fast -c 48 omp_job.sh
#   sbatch -p arm_long -c 48 omp_job.sh
#   sbatch -p arm_serial -c 48 omp_job.sh
# ============================================================================

#SBATCH --job-name=cernaric_omp
#SBATCH --output="log_test/%x/%x-%J.out"
#SBATCH --error="log_test/%x/%x-%J.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /etc/profile.d/zz-cray-pe.sh

# ---- Change these to configure the run ----
BINARY=./${SLURM_JOB_NAME:-data}                       # seq/task/data 
MAP="${1:?Missing map file path as first argument}"
CUTOFF=4

# Stack size for deep DFS recursion
ulimit -s unlimited
export OMP_STACKSIZE=64M

# OpenMP settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OMP_WAIT_POLICY=PASSIVE
export CRAY_OMP_CHECK_AFFINITY=FALSE

echo "=== Job info ==="
echo "Binary:      $BINARY"
echo "Map:         $MAP"
echo "Cutoff:      $CUTOFF"
echo "CPUs/task:   $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"
echo "================"

srun $BINARY $MAP $OMP_NUM_THREADS $CUTOFF

exit 0