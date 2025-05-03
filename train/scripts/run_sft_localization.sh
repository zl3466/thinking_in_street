
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
export TRAIN_SCENE_START=0
export TRAIN_SCENE_END=100
export VIDEO_LENGTH=4


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    ./train/localization/sft_video_new_3b.py \
    --output_dir "./log/Qwen2.5-VL-3B-sft" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "./dataset/examples/sft" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B-sft \
    --save_steps 300 \
    --max_grad_norm 5 \
    --save_only_model true