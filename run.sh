#!/usr/bin/env bash
for algo in edge region; do
    for file in `ls images`; do
        python count.py ${algo} images/${file} outputs/$(basename ${file} .tif)_${algo}.png
    done
done