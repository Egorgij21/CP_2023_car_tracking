#!/bin/bash

paths=(
    "path/to/another/source"
    "path/to/yet/another/source"
)

for path in "${paths[@]}"
do
    echo "Predicting for source: $path"
    python predict.py source="$path"
done
