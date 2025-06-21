#!/bin/bash
#
#SBATCH --job-name=sft2
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/sft2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
module load openmpi/intel/4.0.5;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export DATASET_DIR="/vast/zl3466/dataset"
export EXAMPLE_DIR="/scratch/zl3466/github/thinking_in_street/dataset/examples"
export MAX_VIDEO_LENGTH=16


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    ./train/localization/sft_video_new.py \
    --output_dir "./log/sft/7B-16frames-3dataset-sft" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "all" \
    --deepspeed "./train/local_scripts/zero2.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model false
