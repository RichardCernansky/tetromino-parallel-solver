#!/bin/bash
# extract.sh — extract wall times from logs into CSV
# Usage: bash extract.sh > results.csv

echo "variant,cores,map,wall_time"

for f in log_test/*.out; do
    name=$(basename "$f" .out)
    # name: data_12c_3m-12345
    variant=$(echo "$name" | cut -d_ -f1)
    cores=$(echo "$name" | cut -d_ -f2 | tr -dc '0-9')
    map=$(echo "$name" | cut -d_ -f3- | sed 's/-[0-9]*$//')

    # try both "Wall time:" and "Total wall time:" formats
    time=$(grep -oP '(Wall time:|Total wall time:)\s*\K[\d.]+' "$f" | tail -1)

    if [ -n "$time" ]; then
        echo "$variant,$cores,$map,$time"
    fi
done