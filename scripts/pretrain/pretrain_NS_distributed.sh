#!/bin/bash
#SBATCH --job-name=pretrain_NS_distributed   
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00               
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:4

cd $HOME/Xinhan/github/Meta-PINO
module load conda
conda init bash
source ~/.bashrc
conda activate pino

# Run your training script with the desired number of GPUs
srun python train_PINO3d.py --config_path configs/pretrain-pino/Re500-pretrain-400.yaml --num_gpus 4