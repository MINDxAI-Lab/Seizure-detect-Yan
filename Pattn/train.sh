#!/bin/bash
#SBATCH --job-name=train_EEG_PAttn
#SBATCH --error=./%x.%j.err
#SBATCH --output=./%x.%j.out

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
echo "Loading Conda"
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate seizure_detect

# 定义数据路径变量ls
MODEL_ID="PAttn_EEG_DecompAug_$(date +%Y%m%d_%H%M%S)"
ROOT_PATH="/home/y0chen55/seizure_detect/LLMsForTimeSeries/resampled_12s"
DATA_PATH="seizure_data"

echo "Training configuration:"
echo "MODEL_ID: $MODEL_ID"
echo "ROOT_PATH: $ROOT_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "Start training EEG seizure detection model"

cd PAttn

# 1. baseline训练（不使用增强）
echo "=== 1. Baseline训练（无增强）==="
python main_cls.py \
    --model_id "${MODEL_ID}_baseline" \
    --model PAttn \
    --data eeg_seizure \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --auto_pos_weight \
    --seq_len 3072 \
    --enc_in 19 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --train_epochs 50 \
    --patience 25 \
    --percent 100 \
    --e_layers 3 \
    --d_model 512 \
    --n_heads 16 \
    --dropout 0.2 \
    --patch_size 16 \
    --stride 8 \
    --num_workers 4 \
    --itr 1 \
    --gpu_loc 0 \
    --save_file_name "baseline_results.txt"

echo "Baseline训练完成"

# 2. 使用分解式增强训练（保守参数）
echo "=== 2. 分解式增强训练（保守参数）==="
python main_cls.py \
    --model_id "${MODEL_ID}_decomp_conservative" \
    --model PAttn \
    --data eeg_seizure \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --seq_len 3072 \
    --enc_in 19 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --train_epochs 50 \
    --patience 25 \
    --percent 100 \
    --e_layers 3 \
    --d_model 768 \
    --n_heads 16 \
    --dropout 0.2 \
    --patch_size 16 \
    --stride 8 \
    --num_workers 4 \
    --itr 1 \
    --gpu_loc 0 \
    --aug_method decomp \
    --aug_p 0.5 \
    --aug_win 129 \
    --aug_scale_low 0.9 \
    --aug_scale_high 1.1 \
    --aug_noise 0.02 \
    --save_file_name "decomp_conservative_results.txt"

echo "保守增强训练完成"

# 3. 使用分解式增强训练（积极参数）
echo "=== 3. 分解式增强训练（积极参数）==="
python main_cls.py \
    --model_id "${MODEL_ID}_decomp_aggressive" \
    --model PAttn \
    --data eeg_seizure \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --seq_len 3072 \
    --enc_in 19 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --train_epochs 50 \
    --patience 25 \
    --percent 100 \
    --e_layers 3 \
    --d_model 768 \
    --n_heads 16 \
    --dropout 0.2 \
    --patch_size 16 \
    --stride 8 \
    --num_workers 4 \
    --itr 1 \
    --gpu_loc 0 \
    --aug_method decomp \
    --aug_p 0.7 \
    --aug_win 129 \
    --aug_scale_low 0.8 \
    --aug_scale_high 1.2 \
    --aug_noise 0.05 \
    --save_file_name "decomp_aggressive_results.txt"

echo "积极增强训练完成"

# 4. 只对背景类增强（针对类别不平衡）
echo "=== 4. 背景类增强训练（解决类别不平衡）==="
python main_cls.py \
    --model_id "${MODEL_ID}_decomp_bg_only" \
    --model PAttn \
    --data eeg_seizure \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --seq_len 3072 \
    --enc_in 19 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --train_epochs 50 \
    --patience 25 \
    --percent 100 \
    --e_layers 3 \
    --d_model 768 \
    --n_heads 16 \
    --dropout 0.2 \
    --patch_size 16 \
    --stride 8 \
    --num_workers 4 \
    --itr 1 \
    --gpu_loc 0 \
    --aug_method decomp \
    --aug_p 0.6 \
    --aug_win 129 \
    --aug_scale_low 0.85 \
    --aug_scale_high 1.15 \
    --aug_noise 0.03 \
    --aug_only_bg \
    --save_file_name "decomp_bg_only_results.txt"

echo "背景类增强训练完成"

# 5. WW数据增强
echo "=== 5.WW数据增强==="
python main_cls.py \
    --model_id "PAttn_EEG_WindowWarping_$(date +%Y%m%d_%H%M%S)" \
    --model PAttn --data eeg_seizure \
    --root_path "/home/y0chen55/seizure_detect/LLMsForTimeSeries/resampled_12s" \
    --data_path "seizure_data" \
    --features M \
    --auto_pos_weight \
    --seq_len 3072 \
    --enc_in 19 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --train_epochs 50 \
    --patience 25 \
    --percent 100 \
    --e_layers 3 \
    --d_model 768 \
    --n_heads 16 \
    --dropout 0.2 \
    --patch_size 16 \
    --stride 8 \
    --num_workers 4 \
    --itr 1 \
    --gpu_loc 0 \
    --aug_method ww \
    --aug_ww_p_low 0.3 \
    --aug_ww_p_high 0.7 \
    --aug_ww_win_ratio_low 0.1 \
    --aug_ww_win_ratio_high 0.3 \
    --aug_ww_speed_low 0.8 \
    --aug_ww_speed_high 1.2 \
    --aug_ww_margin 0.5 \
    --save_file_name "window_warping_results.txt"


echo "=== 所有训练完成 ==="
echo "Ending on DATE: $(date)"

