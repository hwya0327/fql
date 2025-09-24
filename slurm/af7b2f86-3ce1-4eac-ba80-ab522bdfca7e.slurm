#!/bin/bash
#SBATCH --job-name=low-priority
#SBATCH --partition=production-cluster
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --array=0-24%18
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=ip-26-0-146-[33,100,122-123,149,183,212,249],ip-26-0-147-[6,94,120,141],ip-26-0-152-[71,101,119,178,186,207,211],ip-26-0-153-[6,62,112,132,166,251],ip-26-0-154-[38,65],ip-26-0-155-[164,174,187,217],ip-26-0-156-[13,40],ip-26-0-157-27
##SBATCH --nodelist=ip-26-0-147-204


env_ids=(HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4)
seeds=(1 2 3 4 5)
env_id=${env_ids[$SLURM_ARRAY_TASK_ID / 5]}
seed=${seeds[$SLURM_ARRAY_TASK_ID % 5]}

echo "Running task $SLURM_ARRAY_TASK_ID with env_id: $env_id and seed: $seed"

srun poetry run python cleanrl/sac_continuous_action.py --track --env-id $env_id --seed $seed # 
