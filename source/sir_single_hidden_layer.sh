#!/bin/bash

for i in {1..30}
do
    for j in {1..5}
    do
        python sir_single_hidden_layer.py $i $j
    done
done