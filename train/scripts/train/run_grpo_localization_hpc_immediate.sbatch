#!/bin/bash
#
#SBATCH --job-name=rtx8000_immediate
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/grpo_immediate_rtx8000.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
module load openmpi/intel/4.0.5;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
export DATASET_DIR="/vast/zl3466/dataset"
cd /scratch/zl3466/github/thinking_in_street;

export NUM_TRAIN_SCENE=100
export VIDEO_LENGTH=4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    ./train/localization/grpo_new_immediate_4q.py \
    --output_dir "./log/Qwen2.5-VL-3B-GRPO-100-rtx8000-sbatch" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/vast/zl3466/dataset/NuScenes" \
    --deepspeed "./train/local_scripts/zero3.json" \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation sdpa \
    --max_pixels 200704 \
    --num_train_epochs 1 \
    --run_name grpo_immediate_rtx8000-100scene \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
