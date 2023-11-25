#!/bin/bash

path_pairs=(
    "path/to/first/source1 path/to/first/source2"
    "path/to/second/source1 path/to/second/source2"
)

for pair in "${path_pairs[@]}"
do
    read -r path1 path2 <<< "$pair"

    echo "Predicting for sources: $path1 and $path2"
    PYTHONPATH='../../../..' python predict.py source="$path1" json_path="$path2"
done