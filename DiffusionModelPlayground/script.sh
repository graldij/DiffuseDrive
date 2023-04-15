#!/bin/bash

# sbatch --ntasks=2 --gres=gpu:2 --constraint='geforce_rtx_2080_ti|titan_xp' --mem-per-cpu=4G --cpus-per-task=4 --output=DiffusionModelPlayground/log/%j.out DiffusionModelPlayground/script.sh

echo "start submit.sh"

#SBATCH --job-name=tmp              # Job name
#SBATCH --time=00:04:00             # 4h / 24h / 120h
#SBATCH --tmp=5G                    # Local scratch

# DATA=${1}
# CFG=${2}

# DATA="/srv/beegfs02/scratch/rl_course/data/nyu2mini_depth.tar.gz"
# CONFIG="nyu/nyu_cls.json"

MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
# MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# source /scratch_net/biwidl211/rl_course_10/.pyenv/versions/3.9.13/envs/playground_diffmodel/bin/activate
source /scratch_net/biwidl211/rl_course_10/.pyenv/versions/playground_diffmodel/bin/activate


echo "Start script"
# cd rl-course-assignment-2/
srun --gpus=1 --cpus-per-task=4 python -u DiffusionModelPlayground/DiffusionModelPlayground.py
