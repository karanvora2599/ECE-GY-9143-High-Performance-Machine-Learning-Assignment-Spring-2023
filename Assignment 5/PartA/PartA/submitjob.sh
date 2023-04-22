#!/bin/bash -e
#SBATCH --output=%x_%j.txt --time=1:30:00 --wrap "sleep infinity"
module purge
module load intel/19.1.2
echo "Hostname: $(hostname)"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $4}')"
cd /scratch/kv2154/Assignment5/PartA/PartA
make clean
make
./vecadd01 $1 >> vecadd01.txt
./matmult01 $2 >> matmult01.txt