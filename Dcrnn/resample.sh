#!/bin/bash
#SBATCH --job-name=resample_TUSZ
#SBATCH --error=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.err
#SBATCH --output=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.out

#SBATCH --time=12:00:00
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
python ./data/resample_signals.py \
  --raw_edf_dir "/home/y0chen55/seizure_detect/TUSZ/edf" \
  --save_dir "/home/y0chen55/seizure_detect/eeg-gnn-ssl/TUSZ_resampled"

echo "Ending on DATE: $(date)"