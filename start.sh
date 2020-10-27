#!/bin/bash

# Activate python virtual env



# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net

for (( i = 2; i < 20; i++ ))
do
    for (( j = 2; j <= 1000; ))
    do  
        for ((k=1;k<20;k++))
        do
            python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=ff --order=y
        done 
        for ((k=1;k<20;k++))
        do
           python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=ff --order=n
        done 
        echo ""
        echo ""
        echo ""
        if [ $j -lt 10 ]
        then 
            let j=j+1
        else
            if [ $j -lt 100 ]
            then 
                let j=j+10
            else 
                let j=j+100
            fi
        fi
    done
done

# Convolutional Neural Net
for (( i = 2; i < 20; i++ ))
do
    for (( j = 2; j <= 1000; ))
    do  
        for ((k=1;k<20;k++))
        do
            python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=y
        done 
        for ((k=1;k<20;k++))
        do
           python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=n
        done 
        echo ""
        echo ""
        echo ""
        if [ $j -lt 10 ]
        then 
            let j=j+1
        else
            if [ $j -lt 100 ]
            then 
                let j=j+10
            else 
                let j=j+100
            fi
        fi
    done
done