
module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
export GEMINI_API_KEY="AIzaSyAiLHN0SsxnsYjj3ycy8jv12JUUbIPkBkw"
export DATASET_DIR="/scratch/zl3466/dataset/NuScenes/train_test"

cd /scratch/zl3466/github/thinking_in_street;

python ./generate_dataset/NuScenes/generate_thought_process_nusc_gemini.py \
    --num_cam 1 \
    --train_num_scene 4 \
    --test_num_scene 1 \
    --step_size 2 \
    --batch_size 4 \
    --data_root_path "/scratch/zl3466/dataset/NuScenes"