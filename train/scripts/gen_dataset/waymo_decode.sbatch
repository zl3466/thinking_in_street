#!/bin/bash
#
#SBATCH --job-name=decode_scannet
#SBATCH --output=/scratch/zl3466/github/thinking_in_street/train_result/waymo_decode.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl3466@nyu.edu
#SBATCH --account=pr_116_tandon_priority

module purge;
module load anaconda3/2020.07;
module load google-cloud-sdk/379.0.0
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3466/env/thinking-in-street/;
export PATH=/scratch/zl3466/env/thinking-in-street/bin:$PATH;
cd /scratch/zl3466/github/thinking_in_street/generate_dataset/Waymo;


python decode_tf.py \
    --data_root /scratch/zl3466/dataset/Waymo/perception_tf/ \
    --target_dir /vast/zl3466/dataset/Waymo/ \
    --split training \
    --process_keys images calib pose \
    --num_scenes 798 \
    --workers 16



# CommandException: 5 files/objects could not be transferred.s ETA 00:00:35       
# Failed to download training: Command '['gsutil', '-m', 'cp', '-r', 'gs://waymo_open_dataset_v_1_4_3/individual_files/training', '/scratch/zl3466/dataset/Waymo/perception_tf']' returned non-zero exit status 1.
# Downloading test split to /scratch/zl3466/dataset/Waymo/perception_tf...
# CommandException: No URLs matched: gs://waymo_open_dataset_v_1_4_3/individual_files/test
# CommandException: 1 file/object could not be transferred.
# Failed to download test: Command '['gsutil', '-m', 'cp', '-r', 'gs://waymo_open_dataset_v_1_4_3/individual_files/test', '/scratch/zl3466/dataset/Waymo/perception_tf']' returned non-zero exit status 1.
# Downloading validation split to /scratch/zl3466/dataset/Waymo/perception_tf...
# Copying gs://waymo_open_dataset_v_1_4_3/individual_files/validation/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord...
# ==> NOTE: You are downloading one or more large file(s), which would            
# run significantly faster if you enabled sliced object downloads. This
# feature is enabled by default but requires that compiled crcmod be
# installed (see "gsutil help crcmod").