#!/bin/bash

# Activate python virtual env

# Processor count
num_nodes=$( cat /proc/cpuinfo | grep processor -c )


# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net

for (( i = 2; i <= $num_nodes; i++ ))
do
    echo "feedforward distributed training with $i threads"
    python3 src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=ff; 
    python3 src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=ff;
    echo "Successfully executed with $i threads."
done

# Convolutional Neural Net

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python3 src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv; 
    python3 src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv; 
    echo "Successfully executed with $i threads."
done