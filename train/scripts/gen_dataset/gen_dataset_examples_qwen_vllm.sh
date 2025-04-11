module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
export DATASET_DIR="/scratch/zl3466/dataset/NuScenes/train_test";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./generate_dataset/NuScenes/generate_thought_process_nusc_qwen_vllm.py \
    --num_cam 1 \
    --train_num_scene 2 \
    --test_num_scene 1 \
    --step_size 2 \
    --batch_size 4 \
    --data_root_path "/scratch/zl3466/dataset/NuScenes" \
    --model "Qwen/Qwen2.5-VL-32B-Instruct" \
    --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"