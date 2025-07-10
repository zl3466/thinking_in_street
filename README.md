Q/A datasets: `./dataset/examples`, json file

train sft: `./train/localization/sft_video_new.py`
train grpo: `./train/localization/grpo_new_4q_all.py`

checkpoints: `./log/3_dataset`

temporal loss (reverse-consistent performance): `./train/localization/trainer/grpo_trainer.py` - `Qwen2VLGRPOTrainer.compute_loss()`

generate Q/A json dataset: `./train/scripts/gen_dataset/gen_examples_final.sbatch`
generate thought process for sft json dataset: `./train/scripts/gen_dataset/gen_thought_process.sbatch`
training scripts: `./train/scripts/train`, the ones with `_new`
