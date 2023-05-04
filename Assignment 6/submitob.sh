#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --output="/home/vb2184/KaranC5_CPU.out"
#SBATCH --job-name=trainer
#SBATCH --mem=70GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
cd /scratch/vb2184
module load intel/19.1.2
module load anaconda3/2020.07
module load python/intel/3.8.6
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')"
echo "GPU: $(nvidia-smi -q | grep 'Product Name')"

python3 Test_Network.py >> Output.txt