module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street/generate_dataset;

export DATASET_DIR="/vast/zl3466/dataset";

python ./_1_gen_examples.py \
    --num_cam 1 \
    --scene_start 0 \
    --scene_end 400 \
    --out_dir "/scratch/zl3466/github/thinking_in_street/dataset/examples/train"