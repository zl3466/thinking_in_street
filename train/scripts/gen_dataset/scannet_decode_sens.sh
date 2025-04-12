module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street/generate_dataset/ScanNet;


python ./decode_sens.py \
    --dataset_path "/scratch/zl3466/dataset/ScanNet/scans" \
    --output_path "/vast/zl3466/dataset/ScanNet/decoded" \
    --scene_idx -1 \
    --frame_skip 5 \
    --export_color_images \
    --export_poses \
    --export_intrinsics