#!/bin/bash
#
#SBATCH --job-name=thinking_in_street_grpo
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/localization_grpo_test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
python ./train/localization/grpo.py \
    --output_dir "./log/Qwen2.5-VL-7B-GRPO" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
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
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Video-R1 \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
