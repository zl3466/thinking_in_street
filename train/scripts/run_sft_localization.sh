#cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


python ../localization/sft_video.py \
    --output_dir "./log/Qwen2.5-VL-7B-Video-7B-cot-sft" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "./Video-R1-data/Video-R1-COT-165k.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-Video-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true