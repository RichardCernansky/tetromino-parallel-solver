#!/bin/bash
# ============================================================================
# Slurm script for MPI + OpenMP runs
#
# Usage:
#   sbatch -p arm_fast -N 1 -n 4 -c 12 mpi_job.sh          # quick test
#   sbatch -p arm_long -N 2 -n 8 -c 12 mpi_job.sh          # 2 nodes
#   sbatch -p arm_long -N 4 -n 16 -c 12 mpi_job.sh         # 4 nodes
# ============================================================================

#SBATCH --job-name=cernaric_mpi
#SBATCH --output="../log/%x-%J.out"
#SBATCH --error="../log/%x-%J.err"

# Activate Cray programming environment
source /etc/profile.d/zz-cray-pe.sh
module load cray-mvapich2_pmix_nogpu

# Stack size (deep DFS recursion needs more stack per thread) 
ulimit -s unlimited
export OMP_STACKSIZE=64M

# MVAPICH2 settings 
export MV2_ENABLE_AFFINITY=0
export MV2_USE_THREAD_WARNING=0
export MV2_SUPPRESS_JOB_STARTUP_PERFORMANCE_WARNING=1
export MV2_HOMOGENEOUS_CLUSTER=1

# communicate purerly with network stack
export MV2_SMP_USE_CMA=0
export MV2_USE_SHARED_MEM=0

# OpenMP settings 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OMP_WAIT_POLICY=PASSIVE
export CRAY_OMP_CHECK_AFFINITY=FALSE

echo "=== Job info ==="
echo "Nodes:       $SLURM_JOB_NUM_NODES"
echo "Tasks:       $SLURM_NTASKS"
echo "CPUs/task:   $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"
echo "================"

# --export=ALL forces srun to propagate all environment variables to MPI processes
srun --export=ALL ./cernaric_mpi ../mapa/mapa5_11.txt $OMP_NUM_THREADS 4
# srun --export=ALL,MV2_ENABLE_AFFINITY=0,MV2_USE_THREAD_WARNING=0,OMP_NUM_THREADS=$OMP_NUM_THREADS,OMP_STACKSIZE=64M,OMP_WAIT_POLICY=PASSIVE ./cernaric_mpi ./mapa/mapa5_11.txt $OMP_NUM_THREADS 4

exit 0