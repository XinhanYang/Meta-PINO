#!/bin/bash
#SBATCH --job-name=eval_pino
#SBATCH --nodes=1
#SBATCH --ntasks=1                    
#SBATCH --mem-per-cpu=40G              
#SBATCH --time=24:00:00               
#SBATCH -p a100-4     
#SBATCH --gres=gpu:a100:1          

cd $HOME/Xinhan/github/Meta-PINO
module load conda
conda init bash
source ~/.bashrc
conda activate pino

python eval_operator.py --config_path configs/pretrain-pino/evaluation.yaml