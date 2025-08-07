### assets
- Q/A json datasets: `./dataset/examples`, json file
- checkpoints: `./log/3_dataset`
---
### important codes
- train sft code: `./train/localization/sft_video_new.py`
- train grpo code: `./train/localization/grpo_new_4q_all.py`
- trainer code, temporal loss implementation (forward-backward-consistency): `./train/localization/trainer/grpo_trainer.py` - `Qwen2VLGRPOTrainer.compute_loss()`
---
### scripts
- generate Q/A json dataset: `./train/scripts/gen_dataset/gen_examples_final.sbatch`
- generate thought process for sft json dataset: `./train/scripts/gen_dataset/gen_thought_process.sbatch`
- training scripts: `./train/scripts/train`, the ones with `_new`
