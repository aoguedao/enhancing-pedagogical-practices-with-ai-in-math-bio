#!/bin/bash

for i in {1..2}
do
    for j in {1..3}
    do
        python sir_n_layers.py $i $j
    done
done