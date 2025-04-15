module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;

export DATASET_DIR="/vast/zl3466/dataset";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./eval/eval_qwen_immediate_4q.py \
   --num_cam 1 \
   --eval_scene_start 50 \
   --eval_scene_end 60 \
   --batch_size 4 \
   --model "Qwen/Qwen2.5-VL-3B-Instruct" \
   --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"
   
# cd /home/zl3466/Documents/github/thinking_in_street;
# export DATASET_DIR="/home/zl3466/Documents/dataset";
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# python ./eval/eval_qwen_immediate_4q.py \
#     --num_cam 1 \
#     --eval_scene_start 0 \
#     --eval_scene_end 1 \
#     --batch_size 4 \
#     --model "Qwen/Qwen2.5-VL-3B-Instruct" \
#     --model_path "/home/zl3466/Documents/Qwen_models"