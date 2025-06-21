#!/bin/bash
#
#SBATCH --job-name=nusc
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/gen_examples_nusc.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street/generate_dataset;

export DATASET_DIR="/vast/zl3466/dataset";

python ./NuScenes/nusc2example.py \
    --out_dir "/scratch/zl3466/github/thinking_in_street/dataset/examples"


# sft: 500-700