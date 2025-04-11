#!/bin/bash
#
#SBATCH --job-name=data_gen_3q
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/localization_qwen_data_gen_3q.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
export DATASET_DIR="/scratch/zl3466/dataset/NuScenes/train_test";
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python ./generate_dataset/NuScenes/generate_thought_process_nusc_qwen_vllm_3q.py \
    --num_cam 1 \
    --train_scene_start 0 \
    --train_scene_end 100 \
    --test_scene_start 0 \
    --test_scene_end 20 \
    --batch_size 4 \
    --data_root_path "/scratch/zl3466/dataset/NuScenes" \
    --model "Qwen/Qwen2.5-VL-72B-Instruct" \
    --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"