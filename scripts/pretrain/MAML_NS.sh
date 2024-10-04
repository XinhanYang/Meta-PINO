#!/bin/bash
#SBATCH --job-name=pretain_pino
#SBATCH --nodes=1
#SBATCH --ntasks=1                    
#SBATCH --mem-per-cpu=40G              
#SBATCH --time=24:00:00               
#SBATCH -p a100-4     
#SBATCH --gres=gpu:a100:1          

cd $HOME/Xinhan/github/Meta-PINO
module load python
source activate pino
python train_PINO3d.py --config_path configs/pretrain-pino/Re500-pretrain-400.yaml