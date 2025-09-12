#!/usr/bin/env python3
"""
数据检查脚本 - 诊断为什么训练集为空
"""

import os
import h5py
import numpy as np
from scipy.fft import fft
from tqdm import tqdm

CLIP_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/clip_data"
LABEL_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/label_data"

def check_dataset_files(split):
    """检查数据集文件情况"""
    print(f"\n=== 检查 {split} 数据集 ===")
    
    cd = os.path.join(CLIP_ROOT, split)
    ld = os.path.join(LABEL_ROOT, split)
    
    # 检查目录是否存在
    if not os.path.exists(cd):
        print(f"[ERROR] Clip数据目录不存在: {cd}")
        return
    if not os.path.exists(ld):
        print(f"[ERROR] Label数据目录不存在: {ld}")
        return
    
    # 获取所有.h5文件
    clip_files = sorted([f for f in os.listdir(cd) if f.endswith(".h5")])
    label_files = sorted([f for f in os.listdir(ld) if f.endswith(".h5")])
    
    print(f"Clip文件数量: {len(clip_files)}")
    print(f"Label文件数量: {len(label_files)}")
    
    # 检查匹配的文件数量
    matched_files = [f for f in clip_files if f in label_files]
    print(f"匹配的文件数量: {len(matched_files)}")
    
    if len(matched_files) == 0:
        print("[ERROR] 没有找到匹配的clip和label文件!")
        print("前5个clip文件:", clip_files[:5])
        print("前5个label文件:", label_files[:5])
        return
    
    # 随机选择几个文件检查数据形状
    sample_files = matched_files[:min(10, len(matched_files))]
    print(f"\n检查前{len(sample_files)}个文件的数据形状:")
    
    valid_count = 0
    shape_stats = {}
    
    for fn in tqdm(sample_files, desc=f"检查{split}数据"):
        try:
            # 检查label文件
            with h5py.File(os.path.join(ld, fn), 'r') as f:
                key = list(f.keys())[0]
                lbl = int(np.array(f[key]))
            
            # 检查clip文件
            with h5py.File(os.path.join(cd, fn), 'r') as f:
                key = list(f.keys())[0]
                eeg = np.array(f[key])
            
            print(f"文件: {fn}")
            print(f"  - EEG原始形状: {eeg.shape}")
            print(f"  - 标签: {lbl}")
            
            # 应用FFT
            if eeg.ndim >= 3:
                eeg_fft = np.log(np.abs(fft(eeg, axis=2)[:, :, :100]) + 1e-30)
                print(f"  - FFT后形状: {eeg_fft.shape}")
                
                # 统计形状
                shape_key = f"{eeg_fft.shape}"
                shape_stats[shape_key] = shape_stats.get(shape_key, 0) + 1
                
                # 检查是否符合要求 (时间维度=10)
                if eeg_fft.ndim == 3 and eeg_fft.shape[1] == 10 and eeg_fft.shape[2] == 100:
                    valid_count += 1
                    print(f"  - ✓ 符合要求 (ndim=3, time=10, freq=100)")
                else:
                    print(f"  - ✗ 不符合要求 (需要: ndim=3, time=10, freq=100)")
            else:
                print(f"  - ✗ 维度不足3")
            
            print()
            
        except Exception as e:
            print(f"文件 {fn} 读取错误: {e}")
    
    print(f"\n=== {split} 数据集统计 ===")
    print(f"总匹配文件数: {len(matched_files)}")
    print(f"符合要求的文件数: {valid_count}")
    print(f"符合要求的比例: {valid_count/len(sample_files)*100:.1f}% (基于样本)")
    
    print(f"\n形状统计:")
    for shape, count in shape_stats.items():
        print(f"  {shape}: {count}个文件")
    
    return len(matched_files), valid_count

def main():
    print("开始检查数据...")
    
    splits = ["train", "eval"]
    
    total_files = 0
    total_valid = 0
    
    for split in splits:
        files, valid = check_dataset_files(split)
        if files is not None:
            total_files += files
            total_valid += valid
    
    print(f"\n=== 总体统计 ===")
    print(f"总文件数: {total_files}")
    print(f"预估符合要求的文件数: {total_valid}")
    
    # 计算在当前抽样率下的预期数据量
    subset_ratio = 0.05
    print(f"\n=== 抽样分析 (抽样率={subset_ratio}) ===")
    
    for split in ["train", "eval"]:
        cd = os.path.join(CLIP_ROOT, split)
        ld = os.path.join(LABEL_ROOT, split)
        
        if os.path.exists(cd) and os.path.exists(ld):
            clip_files = [f for f in os.listdir(cd) if f.endswith(".h5")]
            matched_files = [f for f in clip_files if os.path.exists(os.path.join(ld, f))]
            
            sampled_count = max(int(np.ceil(len(matched_files) * subset_ratio)), 1)
            print(f"{split}: {len(matched_files)} -> {sampled_count} (抽样后)")

if __name__ == "__main__":
    main()
