import numpy as np
import mne
from scipy.signal import resample
import glob
import os
from tqdm import tqdm
import random
import h5py

def split_chb_patients(chb_directory, train_size=18, val_size=3, test_size=3, seed=42):
    """
    随机划分CHB-MIT患者到训练、验证、测试集
    
    Args:
        chb_directory: CHB-MIT数据根目录
        train_size: 训练集患者数量
        val_size: 验证集患者数量  
        test_size: 测试集患者数量
        seed: 随机种子
        
    Returns:
        dict: {'train': [患者列表], 'val': [患者列表], 'test': [患者列表]}
    """
    # 获取所有患者目录
    all_patients = []
    for item in os.listdir(chb_directory):
        patient_path = os.path.join(chb_directory, item)
        if os.path.isdir(patient_path) and item.startswith('chb'):
            all_patients.append(item)
    
    all_patients = sorted(all_patients)  # 确保顺序一致
    print(f"Found {len(all_patients)} patients: {all_patients}")
    
    # 检查患者数量是否足够
    total_needed = train_size + val_size + test_size
    if len(all_patients) < total_needed:
        raise ValueError(f"Not enough patients! Found {len(all_patients)}, need {total_needed}")
    
    # 随机打乱患者列表
    random.seed(seed)
    shuffled_patients = all_patients.copy()
    random.shuffle(shuffled_patients)
    
    # 划分患者
    train_patients = shuffled_patients[:train_size]
    val_patients = shuffled_patients[train_size:train_size + val_size]
    test_patients = shuffled_patients[train_size + val_size:train_size + val_size + test_size]
    
    split_dict = {
        'train': sorted(train_patients),
        'val': sorted(val_patients), 
        'test': sorted(test_patients)
    }
    
    print(f"Patient split (seed={seed}):")
    print(f"  Train ({len(train_patients)}): {train_patients}")
    print(f"  Val ({len(val_patients)}): {val_patients}")
    print(f"  Test ({len(test_patients)}): {test_patients}")
    
    return split_dict

def build_data_dic_chb(chb_directory, patient_list=None):
    """
    构建CHB-MIT数据文件字典
    CHB-MIT数据结构: chb01/chb01_01.edf, chb01/chb01_02.edf, ...
    
    Args:
        chb_directory: CHB-MIT数据根目录
        patient_list: 要处理的患者列表，如果为None则处理所有患者
    """
    all_files = []
    
    # 获取要处理的患者列表
    if patient_list is None:
        patients_to_process = [d for d in os.listdir(chb_directory) 
                             if os.path.isdir(os.path.join(chb_directory, d)) and d.startswith('chb')]
    else:
        patients_to_process = patient_list
    
    for patient_dir in tqdm(patients_to_process, desc="CHB patients"):
        patient_path = os.path.join(chb_directory, patient_dir)
        if os.path.isdir(patient_path):
            # CHB-MIT文件命名: chbXX_YY.edf
            for edf_file in glob.glob(os.path.join(patient_path, '*.edf')):
                # 移除.edf后缀，保持与TUSZ格式一致
                all_files.append(edf_file[:-4])
    return np.array(all_files)

def get_chb_seiz_times(file_base):
    """
    从CHB-MIT的summary文件或者文件名中获取癫痫发作时间
    CHB-MIT有两种标注方式：
    1. summary文件中的标注
    2. 文件名中包含seizure信息
    """
    out = []
    
    # 方法1: 查找对应的summary文件
    patient_dir = os.path.dirname(file_base)
    patient_name = os.path.basename(patient_dir)
    summary_file = os.path.join(patient_dir, f"{patient_name}-summary.txt")
    
    if os.path.exists(summary_file):
        file_name = os.path.basename(file_base) + '.edf'
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                current_file = None
                start_time = None
                end_time = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('File Name:'):
                        current_file = line.split(':', 1)[1].strip()
                        start_time = None
                        end_time = None
                    elif line.startswith('Seizure Start Time:') and current_file == file_name:
                        try:
                            time_str = line.split(':', 1)[1].strip()
                            start_time = int(time_str.split()[0])
                        except:
                            pass
                    elif line.startswith('Seizure End Time:') and current_file == file_name:
                        try:
                            time_str = line.split(':', 1)[1].strip()
                            end_time = int(time_str.split()[0])
                            if start_time is not None and end_time is not None:
                                out.append((start_time, end_time))
                                print(f"Found seizure in {file_name}: {start_time}-{end_time}s")
                        except:
                            pass
        except Exception as e:
            print(f"Error reading summary file {summary_file}: {e}")
    
    # 方法2: 如果没有summary文件，检查文件名是否包含seizure信息
    # 有些CHB文件通过文件名可以判断是否包含癫痫
    base_name = os.path.basename(file_base)
    if 'seizure' in base_name.lower() or '_sz' in base_name.lower():
        # 如果文件名暗示包含癫痫，假设整个文件都是癫痫（这需要根据实际情况调整）
        try:
            raw = raw_eeg_loader_chb(file_base)
            duration = raw.times[-1]
            out.append((0, duration))
            print(f"Detected seizure file by name: {base_name}, duration: 0-{duration}s")
        except:
            pass
    
    return out

def raw_eeg_loader_chb(path_base):
    """CHB-MIT数据加载器"""
    return mne.io.read_raw_edf(path_base + '.edf', verbose=False)

def Seiz_tuple_gen_chb(root, T, patient_list=None):
    """
    为CHB-MIT数据生成癫痫片段元组
    
    Args:
        root: CHB-MIT数据根目录
        T: 时间窗口长度（秒）
        patient_list: 要处理的患者列表，如果为None则处理所有患者
    """
    out = []
    for D in tqdm(build_data_dic_chb(root, patient_list), desc="seizure"):
        seizure_times = get_chb_seiz_times(D)
        if seizure_times:
            raw = raw_eeg_loader_chb(D)
            fs = raw.info['sfreq']
            duration = raw.times[-1]
            
            for start_time, end_time in seizure_times:
                # 确保时间在有效范围内
                if start_time >= 0 and end_time <= duration:
                    # 计算可以提取的T秒片段数量
                    seiz_duration = end_time - start_time
                    num_clips = int(seiz_duration // T)
                    
                    for i in range(num_clips):
                        clip_start = start_time + i * T
                        clip_idx = int(clip_start / T)
                        out.append([D, clip_idx, 1])  # 1表示癫痫
    return out

def Bckg_tuple_gen_chb(root, T, patient_list=None):
    """
    为CHB-MIT数据生成背景片段元组
    
    Args:
        root: CHB-MIT数据根目录
        T: 时间窗口长度（秒）
        patient_list: 要处理的患者列表，如果为None则处理所有患者
    """
    out = []
    for D in tqdm(build_data_dic_chb(root, patient_list), desc="background"):
        seizure_times = get_chb_seiz_times(D)
        raw = raw_eeg_loader_chb(D)
        duration = raw.times[-1]
        
        # 创建癫痫时间掩码
        seizure_mask = np.zeros(int(duration), dtype=bool)
        for start_time, end_time in seizure_times:
            if start_time >= 0 and end_time <= duration:
                seizure_mask[int(start_time):int(end_time)] = True
        
        # 生成背景片段（避开癫痫时间）
        max_clips = int(duration // T)
        for i in range(max_clips):
            clip_start = i * T
            clip_end = (i + 1) * T
            
            # 检查这个片段是否与癫痫时间重叠
            if clip_end <= duration:
                segment_mask = seizure_mask[int(clip_start):int(clip_end)]
                # 如果片段中癫痫时间少于10%，认为是背景
                if np.sum(segment_mask) / len(segment_mask) < 0.1:
                    out.append([D, i, 0])  # 0表示背景
    return out

def save_eeg_clip_chb(seizure_file, bckg_file, time_step, T, to_freq, h5_out, h5_lbl, seed):
    """
    为CHB-MIT数据保存EEG片段
    CHB-MIT通常有18或23个通道，需要适配
    """
    # CHB-MIT常见通道名（可能需要根据具体数据调整）
    INCLUDED_CHANNELS_CHB = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ', 'T7-FT9'  # 根据需要调整
    ]
    
    os.makedirs(h5_out, exist_ok=True)
    os.makedirs(h5_lbl, exist_ok=True)
    
    # 解析元组文件
    combined = parsetxtfile(seizure_file, bckg_file, seed)
    
    # 按文件分组
    by_file = {}
    for base, idx, lbl in combined:
        by_file.setdefault(base, []).append((idx, lbl))
    
    for base, items in tqdm(by_file.items(), desc="CHB clips"):
        try:
            raw = raw_eeg_loader_chb(base)
            
            # CHB-MIT数据通道选择策略
            available_channels = raw.ch_names
            
            # 选择可用的通道（优先使用预定义的通道）
            picks = []
            for ch in INCLUDED_CHANNELS_CHB:
                if ch in available_channels:
                    picks.append(ch)
            
            # 如果预定义通道不够，使用所有可用通道
            if len(picks) < 18:
                picks = available_channels[:min(len(available_channels), 23)]
            
            # 确保至少有18个通道
            if len(picks) < 18:
                print(f"Warning: {base} has only {len(picks)} channels, skipping...")
                continue
            
            raw.pick_channels(picks[:19])  # 保持与TUSZ一致，使用19个通道
            
            sig = raw.get_data()
            fs = raw.info['sfreq']
            sig = resample_data(sig, to_freq, fs)
            
            for idx, lbl in items:
                start = int(idx * to_freq * T)
                slc = [sig[:, start + t*to_freq:start + (t+1)*to_freq] for t in range(T)]
                
                if any(s.shape[1] != to_freq for s in slc):
                    continue
                
                clip = np.stack(slc, 0)
                if clip.shape != (T, 19, to_freq):
                    continue
                
                name = os.path.basename(base) + f"_{idx}.h5"
                with h5py.File(os.path.join(h5_out, name), 'w') as f:
                    f.create_dataset('x', data=clip)
                with h5py.File(os.path.join(h5_lbl, name), 'w') as f:
                    f.create_dataset('y', data=lbl)
                    
        except Exception as e:
            print(f"Error processing {base}: {e}")
            continue

def resample_data(signals, to_freq, fs):
    """重采样数据"""
    N = signals.shape[1]
    num = int(to_freq * (N / fs))
    return resample(signals, num=num, axis=1)

def parsetxtfile(seizure_file, bckg_file, seed):
    """解析元组文件"""
    random.seed(seed)
    combined = []
    
    # 读取癫痫文件
    with open(seizure_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = eval(line.strip())  # [file_base, idx, label]
                combined.append(parts)
    
    # 读取背景文件
    with open(bckg_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = eval(line.strip())
                combined.append(parts)
    
    return combined

def make_tuple_files_chb(dir_, root, T, train_bool, patient_list=None):
    """
    为CHB-MIT数据创建元组文件
    
    Args:
        dir_: 输出目录
        root: CHB-MIT数据根目录
        T: 时间窗口长度（秒）
        train_bool: 是否为训练集（用于数据平衡）
        patient_list: 要处理的患者列表
    """
    seiz = Seiz_tuple_gen_chb(root, T, patient_list)
    bkg = Bckg_tuple_gen_chb(root, T, patient_list)
    
    print(f"Generated {len(seiz)} seizure clips and {len(bkg)} background clips")
    
    if train_bool:
        random.seed(123)
        bkg = random.sample(bkg, min(len(bkg), len(seiz)))  # 平衡数据
        print(f"Balanced to {len(seiz)} seizure clips and {len(bkg)} background clips")
    
    os.makedirs(dir_, exist_ok=True)
    
    with open(os.path.join(dir_, 'seiz_tuple.txt'), 'w') as f:
        for x in seiz:
            f.write(f"{x}\n")
    
    with open(os.path.join(dir_, 'bckg_tuple.txt'), 'w') as f:
        for x in bkg:
            f.write(f"{x}\n")

# 主程序
if __name__ == "__main__":
    # CHB-MIT数据处理配置
    DATA_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/CHBMIT/files/chbmit/1.0.0"
    OUT_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST"
    T, to_freq, seed = 10, 200, 42
    
    # 首先进行患者划分
    print("=== CHB-MIT Patient Split ===")
    patient_splits = split_chb_patients(
        chb_directory=DATA_ROOT,
        train_size=18, 
        val_size=3, 
        test_size=3, 
        seed=42
    )
    
    # 保存患者划分结果
    split_info_path = os.path.join(OUT_ROOT, "patient_splits_chb.txt")
    with open(split_info_path, 'w') as f:
        f.write("CHB-MIT Patient Splits (seed=42)\n")
        f.write("=" * 50 + "\n")
        for split_name, patients in patient_splits.items():
            f.write(f"{split_name.upper()} ({len(patients)} patients): {patients}\n")
    print(f"Patient split info saved to: {split_info_path}")
    
    # 处理数据集：train（包含原train+val）和eval（原test）
    for split_name in ['train', 'eval']:
        print(f"\n=== Processing {split_name.upper()} Dataset ===")
        
        if split_name == 'train':
            # 训练集包含原来的train和val患者
            current_patients = patient_splits['train'] + patient_splits['val']
            train_bool = True
        else:  # split_name == 'eval'
            # eval集使用原来的test患者
            current_patients = patient_splits['test']
            train_bool = False
        
        # 设置路径
        tpl_dir = os.path.join(OUT_ROOT, "scripts", f"tuples_chb_{split_name}")
        clip_dir = os.path.join(OUT_ROOT, "clip_data_chb", split_name)
        lbl_dir = os.path.join(OUT_ROOT, "label_data_chb", split_name)
        
        print(f"Patients: {current_patients} (total: {len(current_patients)})")
        print(f"Input directory: {DATA_ROOT}")
        print(f"Output clip directory: {clip_dir}")
        print(f"Output label directory: {lbl_dir}")
        
        # 生成元组文件
        make_tuple_files_chb(
            dir_=tpl_dir, 
            root=DATA_ROOT, 
            T=T, 
            train_bool=train_bool,
            patient_list=current_patients
        )
        
        # 生成H5文件
        save_eeg_clip_chb(
            seizure_file=os.path.join(tpl_dir, 'seiz_tuple.txt'),
            bckg_file=os.path.join(tpl_dir, 'bckg_tuple.txt'),
            time_step=1, 
            T=T, 
            to_freq=to_freq, 
            h5_out=clip_dir, 
            h5_lbl=lbl_dir, 
            seed=seed
        )
        
        print(f"{split_name.upper()} dataset processing completed!")
    
    print("\n=== All CHB-MIT Data Processing Completed! ===")
    print(f"Patient split info: {split_info_path}")
    print("Output directories:")
    print(f"  train: clip_data_chb/train/ and label_data_chb/train/ (包含21人：18训练+3验证)")
    print(f"  eval: clip_data_chb/eval/ and label_data_chb/eval/ (包含3人测试)")
