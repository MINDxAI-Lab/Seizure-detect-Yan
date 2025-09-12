#!/bin/sh
#SBATCH --cpus-per-task=6 # Number of cores
#SBATCH --mem=24gb  # Total RAM in GB
#SBATCH --time=12:00:00  # Time limit hrs:min:sec; for using days use --time= days-hrs:min:sec
#SBATCH --gpus=1  # Request one GPU

#SBATCH --job-name=train_timenet # Job name
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=y0chen55@louisville.edu # Where to send mail
#SBATCH --output=R_%x_%j.out # Standard output and error log

echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
echo "Loading Conda"
module load conda
conda activate seizure_detect
echo "1st round Start running"
python test_train.py
echo "1st round training completed"

echo "2nd round Start running"
python test_train.py
echo "2nd round training completed"

echo "3rd round Start running"
python test_train.py
echo "3rd round training completed"

echo "Ending on DATE: $(date)"