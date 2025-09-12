#!/bin/sh
#SBATCH --cpus-per-task=4 # Number of cores
#SBATCH --mem=20gb  # Total RAM in GB
#SBATCH --time=36:00:00  # Time limit hrs:min:sec; for using days use --time= days-hrs:min:sec

#SBATCH --job-name=rest_data_preprocess # Job name
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=y0chen55@louisville.edu # Where to send mail
#SBATCH --output=R_%x_%j.out # Standard output and error log

echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
module load conda
conda activate seizure_detect
echo "Start running"
python data_utils.py
echo "Preprocessing data completed"

echo "Ending on DATE: $(date)"