#!/bin/bash
#SBATCH --job-name=iMAML_NS   
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

python iMAML_PINO_NS.py --config_path configs/meta/Re500-MAML-NS-05s.yaml