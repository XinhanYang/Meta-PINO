#!/bin/bash
#SBATCH --job-name=iMAML_NS   
#SBATCH --nodes=1
#SBATCH --ntasks=1                    
#SBATCH --mem-per-cpu=40G              
#SBATCH --time=24:00:00               
#SBATCH -p a100-4     
#SBATCH --gres=gpu:a100:1          

cd $HOME/Xinhan/github/Meta-PINO
module load python
source activate pino
python iMAML_PINO_NS.py --config_path /users/4/mou00006/Xinhan/github/Meta-PINO/configs/meta/Re500-iMAML-NS-05s.yaml