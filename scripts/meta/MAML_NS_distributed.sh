#!/bin/bash
#SBATCH --job-name=MAML_NS_distributed   
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=2:00:00               
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:4

cd $HOME/Xinhan/github/Meta-PINO
module load python
source activate pino

export MASTER_ADDR=$(hostname)            # Master node address
export MASTER_PORT=12355                  # Master node port (ensure it's free and not blocked)

# Print environment variables for debugging
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

# Run your training script with the desired number of GPUs
srun python MAML_PINO_NS_distributed.py --config_path configs/meta/Re500-MAML-NS-05s.yaml --num_gpus 4