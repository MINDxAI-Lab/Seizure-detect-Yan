#!/bin/bash
#SBATCH --job-name=train_dcrnn
#SBATCH --error=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.err
#SBATCH --output=/home/y0chen55/seizure_detect/eeg-gnn-ssl/%x.%j.out

#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1


echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
echo "Loading Conda"
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate eeg_gnn
python train.py \
  --task detection \
  --do_train \
  --model_name dcrnn \
  --graph_type individual \
  --max_seq_len 12 \
  --time_step_size 1 \
  --input_dir ./TUSZ_resampled \
  --raw_data_dir /home/y0chen55/seizure_detect/TUSZ \
  --use_fft \
  --train_batch_size 64 \
  --test_batch_size 64 \
  --num_workers 8 \
  --save_dir ./models_v203_dcrnn_12s_individual_finetune \
  --num_epochs 20 \
  --fine_tune \
  --load_model_path ./pretrained/pretrained_correlation_graph_12s.pth.tar \
  --pretrained_num_rnn_layers 3 \
  --lr_init 3e-5
echo "1st round training completed"

echo "Ending on DATE: $(date)"