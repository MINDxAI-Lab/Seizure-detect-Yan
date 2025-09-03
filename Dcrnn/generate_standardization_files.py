#!/usr/bin/env python3
"""
Generate standardization files (means and stds) for TUSZ v2.0.3 dataset
"""

import os
import sys
import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

# Add current directory to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))
from data.data_utils import computeFFT
from constants import FREQUENCY

def load_preprocessed_clip(clip_path):
    """Load a preprocessed clip from h5 file"""
    try:
        with h5py.File(clip_path, 'r') as hf:
            clip_data = hf['clip'][()]
        return clip_data
    except Exception as e:
        print(f"Error loading {clip_path}: {e}")
        return None

def load_raw_and_process_clip(h5_path, clip_idx, clip_len=12, is_fft=True):
    """Load raw resampled data and process a specific clip"""
    try:
        with h5py.File(h5_path, 'r') as f:
            signal_array = f["resampled_signal"][()]
            resampled_freq = f["resample_freq"][()]
        
        assert resampled_freq == FREQUENCY
        
        # Extract clip
        physical_clip_len = int(FREQUENCY * clip_len)
        physical_time_step_size = int(FREQUENCY * 1)  # 1 second time steps
        
        start_window = clip_idx * physical_clip_len
        end_window = start_window + physical_clip_len
        
        if end_window > signal_array.shape[1]:
            return None
        
        curr_slc = signal_array[:, start_window:end_window]
        
        # Generate time steps
        start_time_step = 0
        time_steps = []
        while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
            end_time_step = start_time_step + physical_time_step_size
            curr_time_step = curr_slc[:, start_time_step:end_time_step]
            
            if is_fft:
                curr_time_step, _ = computeFFT(curr_time_step, n=physical_time_step_size)
            
            time_steps.append(curr_time_step)
            start_time_step = end_time_step
        
        eeg_clip = np.stack(time_steps, axis=0)
        return eeg_clip
        
    except Exception as e:
        print(f"Error processing {h5_path}, clip {clip_idx}: {e}")
        return None

def compute_statistics(file_markers_dir, preprocessed_dir, resampled_dir, clip_len=12, use_fft=True, max_samples=None):
    """Compute mean and std statistics from training data"""
    
    # Read training file list
    train_sz_file = os.path.join(file_markers_dir, f"trainSet_seq2seq_{clip_len}s_sz.txt")
    train_nosz_file = os.path.join(file_markers_dir, f"trainSet_seq2seq_{clip_len}s_nosz.txt")
    
    train_files = []
    
    # Read seizure files
    with open(train_sz_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')
            train_files.append(filename)
    
    # Read non-seizure files  
    with open(train_nosz_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')
            train_files.append(filename)
    
    print(f"Found {len(train_files)} training files")
    
    # Collect all data for statistics
    all_data = []
    processed_count = 0
    failed_count = 0
    expected_shape = None
    
    for filename in tqdm(train_files, desc="Loading training data"):
        clip_data = None
        
        # Try preprocessed directory first
        if preprocessed_dir and os.path.exists(preprocessed_dir):
            preprocessed_path = os.path.join(preprocessed_dir, filename)
            if os.path.exists(preprocessed_path):
                clip_data = load_preprocessed_clip(preprocessed_path)
        
        # If not found in preprocessed, try to process from raw
        if clip_data is None and resampled_dir:
            # Extract info from filename: basename.edf_clipidx.h5
            base_name = filename.replace('.h5', '')
            if '_' in base_name and base_name.count('.edf_') == 1:
                raw_name, clip_idx_str = base_name.split('.edf_')
                raw_name += '.h5'
                clip_idx = int(clip_idx_str)
                
                raw_path = os.path.join(resampled_dir, raw_name)
                if os.path.exists(raw_path):
                    clip_data = load_raw_and_process_clip(raw_path, clip_idx, clip_len, use_fft)
        
        if clip_data is not None:
            # Check shape consistency and add debug info
            if len(all_data) == 0:
                expected_shape = clip_data.shape
                print(f"Expected clip shape: {expected_shape}")
            elif clip_data.shape != expected_shape:
                print(f"Warning: Shape mismatch! Expected {expected_shape}, got {clip_data.shape} for file {filename}")
                continue  # Skip clips with inconsistent shapes
            
            all_data.append(clip_data)
            processed_count += 1
        else:
            failed_count += 1
        
        # Process in batches to avoid memory issues
        if max_samples is not None and len(all_data) >= max_samples:
            break
    
    print(f"Successfully loaded {processed_count} clips, failed: {failed_count}")
    
    if len(all_data) == 0:
        raise ValueError("No training data found! Check your paths.")
    
    # Convert to numpy array and compute statistics
    print("Computing statistics...")
    if len(all_data) == 0:
        raise ValueError("No valid training data found! Check your paths.")
    
    # Stack all clips into a single array
    all_data = np.stack(all_data, axis=0)  # Shape: (n_samples, seq_len, n_channels, n_features)
    
    print(f"Data shape: {all_data.shape}")
    
    # Compute mean and std across all dimensions except channel and feature
    # We want per-channel, per-feature statistics
    means = np.mean(all_data, axis=(0, 1))  # Shape: (n_channels, n_features)
    stds = np.std(all_data, axis=(0, 1))    # Shape: (n_channels, n_features)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1.0, stds)
    
    print(f"Computed means shape: {means.shape}")
    print(f"Computed stds shape: {stds.shape}")
    print(f"Mean values range: {means.min():.4f} to {means.max():.4f}")
    print(f"Std values range: {stds.min():.4f} to {stds.max():.4f}")
    
    return means, stds

def main():
    parser = argparse.ArgumentParser(description='Generate standardization files for TUSZ v2.0.3')
    parser.add_argument('--file_markers_dir', type=str, required=True,
                        help='Directory containing the new file markers')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='Directory containing preprocessed clips (optional)')
    parser.add_argument('--resampled_dir', type=str, default=None,
                        help='Directory containing resampled raw data (fallback)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to file_markers_dir)')
    parser.add_argument('--clip_len', type=int, default=12,
                        help='Clip length in seconds')
    parser.add_argument('--use_fft', action='store_true', default=True,
                        help='Whether to use FFT')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use for statistics (default: use all available)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.file_markers_dir
    
    print(f"Computing statistics for {args.clip_len}s clips...")
    print(f"Using FFT: {args.use_fft}")
    print(f"File markers dir: {args.file_markers_dir}")
    print(f"Preprocessed dir: {args.preprocessed_dir}")
    print(f"Resampled dir: {args.resampled_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Compute statistics
    means, stds = compute_statistics(
        args.file_markers_dir,
        args.preprocessed_dir, 
        args.resampled_dir,
        args.clip_len,
        args.use_fft,
        args.max_samples
    )
    
    # Save statistics
    os.makedirs(args.output_dir, exist_ok=True)
    
    fft_suffix = "_fft" if args.use_fft else ""
    means_file = os.path.join(args.output_dir, f"means_seq2seq{fft_suffix}_{args.clip_len}s_szdetect_single.pkl")
    stds_file = os.path.join(args.output_dir, f"stds_seq2seq{fft_suffix}_{args.clip_len}s_szdetect_single.pkl")
    
    with open(means_file, 'wb') as f:
        pickle.dump(means, f)
    
    with open(stds_file, 'wb') as f:
        pickle.dump(stds, f)
    
    print(f"Saved means to: {means_file}")
    print(f"Saved stds to: {stds_file}")
    print("Done!")

if __name__ == "__main__":
    main()
