#!/bin/bash

mkdir -p log_test

for map in ./mapa_test/*.txt; do
    name=$(basename "$map" .txt)

    #run serial
    sbatch -p arm_serial -c 1 -J "seq_1c_${name}" ./scripts/omp_job.sh "$map"

    # run data/task on multiple cores
    for c in 2 4 8 12 24 48; do
        sbatch -p arm_long -c $c -J "data_${c}c_${name}" ./scripts/omp_job.sh "$map"
        sbatch -p arm_long -c $c -J "task_${c}c_${name}" ./scripts/omp_job.sh "$map"
    done

    # run mpi on multiple nodes
    sbatch -p arm_long -N 1 -n 6  -c 8 -J "mpi_48c_${name}"  ./scripts/mpi_job.sh "$map"
    sbatch -p arm_long -N 2 -n 12 -c 8 -J "mpi_96c_${name}"  ./scripts/mpi_job.sh "$map"
    sbatch -p arm_long -N 4 -n 24 -c 8 -J "mpi_192c_${name}" ./scripts/mpi_job.sh "$map"
    sbatch -p arm_long -N 6 -n 36 -c 8 -J "mpi_288c_${name}" ./scripts/mpi_job.sh "$map"

    echo "Submitted ${name}: 17 jobs"
done