module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
export DATASET_DIR="/vast/zl3466/dataset/ScanNet/decoded";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./generate_dataset/ScanNet/generate_thought_process_scannet_qwen_vllm_3q.py \
    --num_cam 1 \
    --train_num_scene 1 \
    --test_num_scene 1 \
    --step_size 2 \
    --batch_size 4 \
    --data_root_path "/vast/zl3466/dataset/ScanNet" \
    --model "Qwen/Qwen2.5-VL-32B-Instruct" \
    --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"