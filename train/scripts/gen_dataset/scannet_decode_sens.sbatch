#!/bin/bash
#
#SBATCH --job-name=decode_scannet
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/decode_sens.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street/generate_dataset/ScanNet;


python ./decode_sens.py \
    --dataset_path "/scratch/zl3466/dataset/ScanNet/scans" \
    --output_path "/vast/zl3466/dataset/ScanNet/decoded" \
    --scene_start 600 \
    --scene_end 700 \
    --scene_idx -1 \
    --frame_skip 6 \
    --export_color_images \
    --export_poses \
    --export_intrinsics