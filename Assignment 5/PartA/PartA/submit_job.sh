#!/bin/bash -e
#SBATCH --output=%x_%j.txt --time=1:30:00 --wrap "sleep infinity"
module purge
module load intel/19.1.2
module load cuda/11.6.2
echo "Hostname: $(hostname)"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $4}')"
echo "GPU: $(lspci | grep -i nvidia)"
echo "$(nvidia-smi)"
g++ -std=c++11 -o PartB_Q2 PartB_Q2.cu
./PartB_Q2 >> PartB_Q2_Output.txt