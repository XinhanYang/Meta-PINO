#!/bin/bash
#SBATCH --job-name=iMAML_deeponet_NS_distributed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G 
#SBATCH --time=48:00:00
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:4

cd $HOME/Xinhan/github/Meta-PINO
module load conda
conda activate pino

torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         iMAML_deeponet_NS_distributed.py --config_path configs/meta-deeponet/iMAML-deeponet-NS-05s.yaml --num_gpus 4