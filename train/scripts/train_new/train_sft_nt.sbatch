#!/bin/bash
#SBATCH --job-name=gr+sft
#SBATCH --output=/scratch/zl3466/log/volm/script_log/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

# Generate timestamp for unique folder names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="7B+sft+grpo_${TIMESTAMP}"
mkdir -p /scratch/zl3466/log/volm/script_log
mkdir -p /scratch/zl3466/log/volm/${RUN_NAME}/ckpt

module purge;
module load anaconda3/2020.07;
module load openmpi/intel/4.0.5;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export DATASET_DIR="/vast/zl3466/dataset"
export EXAMPLE_DIR="/scratch/zl3466/github/thinking_in_street/dataset/examples"
# export NUM_TRAIN_SCENE=50
# export TRAIN_SCENE_START=0
# export TRAIN_SCENE_END=400
export VIDEO_LENGTH=16

# --resume_from_checkpoint "/scratch/zl3466/github/thinking_in_street/log/Qwen2.5-VL-7B-GRPO-a100-16frames_0-200_nt/checkpoint-1200" \
 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
     --nnodes="1" \
     --node_rank="0" \
     --master_addr="127.0.0.1" \
     --master_port="12365" \
     ./train/localization/grpo_new_4q_all.py \
     --output_dir "/scratch/zl3466/log/volm/${RUN_NAME}/ckpt" \
     --model_name_or_path "/scratch/zl3466/log/volm/ckpt/sft/7B-16frames-3dataset-sft/checkpoint-4802" \
     --dataset_name "all" \
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
     --temporal false \
     --len_control true \
     --attn_implementation flash_attention_2 \
     --max_pixels 401408 \
     --num_train_epochs 1 \
     --run_name 7b-16frames-sft-nt \
     --save_steps 1000 \
     --beta 0.04 \
     --max_grad_norm 5 \
     --save_only_model false \
     --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance

#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
#    --nnodes="1" \
#    --node_rank="0" \
#    --master_addr="127.0.0.1" \
#    --master_port="12365" \
#    ./train/localization/grpo_new_4q_all.py \
#    --output_dir "/scratch/zl3466/log/volm/${RUN_NAME}/ckpt" \
#    --model_name_or_path "/scratch/zl3466/log/volm/ckpt/sft/7B-16frames-3dataset-sft/checkpoint-4802" \
#    --dataset_name "all" \
#    --deepspeed "./train/local_scripts/zero3.json" \
#    --max_prompt_length 16384 \
#    --max_completion_length 768 \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate 1e-6 \
#    --lr_scheduler_type "cosine" \
#    --weight_decay 0.01 \
#    --bf16 \
#    --logging_steps 1 \
#    --gradient_checkpointing true \
#    --temporal false \
#    --len_control true \
#    --attn_implementation flash_attention_2 \
#    --max_pixels 401408 \
#    --num_train_epochs 1 \
#    --run_name ${RUN_NAME} \
#    --save_steps 1000 \
#    --beta 0.04 \
#    --max_grad_norm 5 \
#    --save_only_model false \
#    --resume_from_checkpoint "/scratch/zl3466/github/thinking_in_street/log/3_dataset/7B+sft+grpo/checkpoint-9000" \
#    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance