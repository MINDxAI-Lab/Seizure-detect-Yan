#!/bin/bash
#SBATCH --job-name=resample_TUSZ
#SBATCH --error=./%x.%j.err
#SBATCH --output=./%x.%j.out

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
conda activate seizure_detect
echo "Start running"
python resample.py
echo "Preprocessing data completed"

echo "Ending on DATE: $(date)"