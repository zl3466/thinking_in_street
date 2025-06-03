import subprocess
import os
import sys
from pathlib import Path
import argparse

class WaymoDownloader:
    def __init__(self, output_dir="./waymo_data", version="v_1_4_3"):
        self.output_dir = Path(output_dir)
        self.version = version
        self.bucket_base = f"gs://waymo_open_dataset_{version}"

    def check_gsutil(self):
        """Check if gsutil is installed and authenticated"""
        try:
            result = subprocess.run(["gsutil", "ls"],
                                    capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def download_split(self, split_name):
        """Download a specific data split"""
        split_path = f"{self.bucket_base}/individual_files/{split_name}"
        # output_path = f"{self.output_dir}/{split_name}"
        # os.makedirs(output_path, exist_ok=True)
        cmd = ["gsutil", "-m", "cp", "-r", split_path, str(self.output_dir)]

        print(f"Downloading {split_name} split to {self.output_dir}...")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully downloaded {split_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {split_name}: {e}")
            return False

    def download_perception_dataset(self, splits=["training", "testing", "validation"]):
        """Download the full perception dataset"""

        # Check prerequisites
        if not self.check_gsutil():
            print("Error: gsutil not found or not authenticated")
            print("Please install Google Cloud SDK and run 'gcloud auth login'")
            return False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Download each split
        success = True
        for split in splits:
            if not self.download_split(split):
                success = False

        return success


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True, help='path to output folder')
    args = parser.parse_args()
    downloader = WaymoDownloader(
        output_dir=args.output_path,
        version="v_1_4_3"  # Latest version
    )

    # Download training and validation splits
    success = downloader.download_perception_dataset(["training", "testing", "validation"])

    if success:
        print("Dataset download completed successfully!")
    else:
        print("Some downloads failed. Check your authentication and try again.")