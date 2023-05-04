#!/bin/bash
#SBATCH --time=00:90:00
#SBATCH --output="/scratch/kv2154/Assignment6/OutputQ2.txt"
#SBATCH --partition=rtx8000
#SBATCH --job-name=trainer
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

cd /scratch/kv2154/Assignment6

module load intel/19.1.2
module load anaconda3/2020.07
module load python/intel/3.8.6
module load cuda/11.6.2

echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')"
echo "GPU: $(nvidia-smi -q | grep 'Product Name')"

pip install typing-extensions
pip install sympy
pip install torch torchvision
python3 Q2.py