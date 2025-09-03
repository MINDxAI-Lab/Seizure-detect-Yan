#!/bin/bash
#SBATCH --job-name=Standarization_TUSZ
#SBATCH --error=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.err
#SBATCH --output=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.out

#SBATCH --time=18:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --partition=compute

echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
echo "Loading Conda"
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate eeg_gnn
echo "Start running"
python generate_standardization_files.py \
  --file_markers_dir ./data/file_markers_detection \
  --resampled_dir ./TUSZ_resampled \
  --output_dir ./data/file_markers_detection \
  --clip_len 12

  echo "Ending on DATE: $(date)"