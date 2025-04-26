module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export DATASET_DIR="/vast/zl3466/dataset";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./eval/eval_qwen_immediate_4q_plot_prepared.py \
  --num_cam 1 \
  --eval_scene_start 400 \
  --eval_scene_end 500 \
  --batch_size 4 \
  --base_model "Qwen2.5-VL-7B-Instruct" \
  --finetuned_model "checkpoint-600" \
  --model_path "/scratch/zl3466/github/thinking_in_street/log/Qwen2.5-VL-7B-GRPO-200-a100-16frames-sbatch" \
  --example_dir "/scratch/zl3466/github/thinking_in_street/dataset/examples"
  

   
#  cd /home/zl3466/Documents/github/thinking_in_street;
#  export DATASET_DIR="/home/zl3466/Documents/dataset";
#  export VLLM_WORKER_MULTIPROC_METHOD=spawn
#  python ./eval/eval_qwen_immediate_4q_plot.py \
#      --num_cam 1 \
#      --eval_scene_start 47 \
#      --eval_scene_end 48 \
#      --batch_size 4 \
#      --base_model "Qwen/Qwen2.5-VL-3B-Instruct" \
#      --finetuned_model "checkpoint-300" \
#      --model_path "/home/zl3466/Documents/Qwen_models"