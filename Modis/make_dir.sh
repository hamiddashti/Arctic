#!/bin/bash

FOLDERS=(`seq 2002 2019`)
n=${#FOLDERS[@]}
for i in `seq 0 $n`; do
    mkdir Tifs/${FOLDERS[$i]}
done
