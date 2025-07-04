module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street;
export DATASET_DIR="/vast/zl3466/dataset";
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python ./generate_dataset/_2_generate_thought_process.py \
    --example_dir "/scratch/zl3466/github/thinking_in_street/dataset/examples" \
    --out_dir "/scratch/zl3466/github/thinking_in_street/dataset/examples/cold_start" \
    --scene_start 500 \
    --scene_end 600 \
    --model_name "Qwen/Qwen2.5-VL-72B-Instruct" \
    --model_path "/scratch/zl3466/github/thinking_in_street/model/Qwen"
