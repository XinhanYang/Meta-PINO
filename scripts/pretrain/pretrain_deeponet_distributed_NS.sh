#!/bin/bash
#SBATCH --job-name=pretrain_deeponet_NS_distributed_400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G 
#SBATCH --time=96:00:00               
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:4

cd $HOME/Xinhan/github/Meta-PINO
module load conda
conda activate pino

# Run your training script with the desired number of GPUs
#srun python train_PINO3d.py --config_path configs/pretrain-pino/Re500-pretrain-1000.yaml --num_gpus 4

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         deeponet_ddp.py --config_path configs/pretrain-deeponet/Re500-pretrain-deeponet-400.yaml --num_gpus 4