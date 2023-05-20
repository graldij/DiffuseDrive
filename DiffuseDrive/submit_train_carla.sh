#!/bin/bash
#
#SBATCH -t 24:00:00
#SBATCH --job-name=dec-diff_carla
#SBATCH --output=log/initial%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --constraint='geforce_rtx_2080_ti|titan_xp'
#SBATCH --cpus-per-task=1

cd /scratch_net/biwidl216/rl_course_14/project/our_approach/decision-diffuser/code/analysis
source /scratch_net/biwidl216/rl_course_14/conda/etc/profile.d/conda.sh
conda activate decdiff
export PYTHONPATH=/scratch_net/biwidl216/rl_course_14/project/our_approach/decision-diffuser/code
pwd
python train.py
##bash data_collection/bashs/weather-1/routes_town01_long.sh &