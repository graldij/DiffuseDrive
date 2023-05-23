#!/bin/bash


# sbatch --ntasks=1 --gres=gpu:1 --constraint='geforce_rtx_2080_ti|titan_xp' --mem-per-cpu=16G --cpus-per-task=4 --output=log/%j.out submit_train_carla_j.sh


echo "start submit_train_carla_j.sh"


#SBATCH --job-name=tmp              # Job name
#SBATCH --time=00:24:00             # 4h / 24h / 120h
#SBATCH --tmp=5G                    # Local scratch

cd /home/rl_course_10/DiffuseDrive

source /scratch_net/biwidl211/rl_course_10/.pyenv/versions/playground_diffmodel/bin/activate

PYTHONPATH=${PYTHONPATH}:~/DiffuseDrive python scripts/train.py
