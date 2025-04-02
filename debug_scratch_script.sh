#!/bin/bash
#
#SBATCH --job-name=thinking_in_street_grpo
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/localization_grpo_test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/video-r1/;
export PATH=/scratch/zl3466/env/video-r1/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
python ./debug_scratch.py