#module purge;
#module load anaconda3/2020.07;
#source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
#conda activate /scratch/zl3466/env/thinking-in-street/;
#export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /home/zl3466/Documents/github/thinking_in_street;
export DATASET_DIR="/home/zl3466/Downloads/ScanNet/decoded";
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python ./generate_dataset/ScanNet/generate_thought_process_scannet_qwen_vllm_3q.py \
    --num_cam 1 \
    --train_scene_start 0 \
    --train_scene_end 1 \
    --test_scene_start 0 \
    --test_scene_end 1 \
    --batch_size 4 \
    --data_root_path "/home/zl3466/Downloads/ScanNet" \
    --model "Qwen/Qwen2.5-VL-32B-Instruct" \
    --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"