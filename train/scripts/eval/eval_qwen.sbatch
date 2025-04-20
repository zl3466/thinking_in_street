#!/bin/bash
#
#SBATCH --job-name=eval_qwen
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/eval_result/eval_qwen.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export DATASET_DIR="/vast/zl3466/dataset";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./eval/eval_qwen_immediate_4q_plot.py \
  --num_cam 1 \
  --eval_scene_start 110 \
  --eval_scene_end 115 \
  --batch_size 4 \
  --base_model "Qwen2.5-VL-7B-Instruct" \
  --finetuned_model "checkpoint-1800" \
  --model_path "/scratch/zl3466/github/thinking_in_street/log/Qwen2.5-VL-7B-GRPO-100-a100-sbatch-4frame"

