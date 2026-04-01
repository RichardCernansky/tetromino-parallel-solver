#!/bin/bash

for map in ./mapa_test/*.txt; do
    sbatch -p arm_serial -J seq ./scripts/omp_job.sh "$map"
done