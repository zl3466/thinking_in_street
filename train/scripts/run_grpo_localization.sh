export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/scratch/zl3466/github/thinking_in_street/debug_log_3b.txt"


# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# Qwen/Qwen2.5-VL-7B-Instruct

module purge;
module load anaconda3/2020.07;
module load openmpi/intel/4.0.5;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export NUM_TRAIN_SCENE=40

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    ./train/localization/grpo.py \
    --output_dir "./log/Qwen2.5-VL-3B-GRPO" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/scratch/zl3466/dataset/NuScenes" \
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
    --max_pixels 100352 \
    --num_train_epochs 1 \
    --run_name localization-3b-debug \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
