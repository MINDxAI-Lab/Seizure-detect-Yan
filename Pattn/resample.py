import numpy as np
import mne
from scipy.signal import resample
import glob
import os
from tqdm import tqdm
import random
import h5py

# This script processes EEG data for seizure detection, including resampling, tuple generation, and saving clips.

def build_data_directory_patient(train_set_directory, patient):
    """
    For a given patient, find all EDF file bases (without .edf extension) in their session directories.
    Assumes each session has one montage subfolder containing EDF files.
    Returns a list of unique file bases.
    """
    base = os.path.join(train_set_directory, patient)
    out = []
    for ses in os.listdir(base):
        ses_dir = os.path.join(base, ses)
        if not os.path.isdir(ses_dir): continue
        # Assume one montage subfolder per session
        montages = [d for d in os.listdir(ses_dir) if os.path.isdir(os.path.join(ses_dir, d))]
        if not montages: continue
        data_dir = os.path.join(ses_dir, montages[0])
        # Find all EDFs
        for edf in glob.glob(os.path.join(data_dir, '*.edf')):
            out.append(edf[:-4])  # Remove .edf extension
    return np.unique(out)

def build_data_dic(train_set_directory):
    """
    Build a list of all EDF file bases for all patients in the training set directory.
    """
    all_dirs = []
    for p in tqdm(os.listdir(train_set_directory), desc="patients"):
        path = os.path.join(train_set_directory, p)
        if os.path.isdir(path):
            all_dirs.extend(build_data_directory_patient(train_set_directory, p))
    return np.array(all_dirs)

def raw_eeg_loader(path_base):
    """
    Load raw EEG data from an EDF file using MNE.
    """
    return mne.io.read_raw_edf(path_base + '.edf', verbose=False)

def resample_data(signals, to_freq, fs):
    """
    Resample EEG signals to a target frequency.
    signals: np.ndarray of shape (channels, samples)
    to_freq: target frequency (Hz)
    fs: original sampling frequency (Hz)
    """
    N = signals.shape[1]
    num = int(to_freq * (N / fs))
    return resample(signals, num=num, axis=1)

def get_seiz_times(path_base):
    """
    Parse the .csv_bi file for a given EEG recording to extract seizure start and end times.
    Returns a list of (start, end) tuples in seconds.
    """
    out = []
    with open(path_base + '.csv_bi') as f:
        for L in f:
            if 'seiz' in L:
                parts = L.split(',')
                out.append((float(parts[1]), float(parts[2])))
    return out

def parsetxtfile(seizure_file, bckg_file, seed):
    """
    Read and combine tuples from background and seizure txt files.
    Each tuple: [file_base, index, label]. Shuffles the combined list.
    Handles both int and np.int64 formats.
    """
    random.seed(seed)
    def _read(fn):
        lines = open(fn).read().splitlines()
        tuples = []
        for L in lines:
            lst = L.strip()[1:-1].split(',')
            # Handle both regular integers and np.int64() format
            path = eval(lst[0])
            idx = eval(lst[1].strip()) if 'np.int64' in lst[1] else int(lst[1].strip())
            label = eval(lst[2].strip()) if 'np.int64' in lst[2] else int(lst[2].strip())
            tuples.append([path, idx, label])
        return tuples
    combined = _read(bckg_file) + _read(seizure_file)
    random.shuffle(combined)
    return combined

def Seiz_tuple_gen(root, T):
    """
    Generate tuples for seizure segments.
    For each file, for each seizure interval, create tuples for each T-second window overlapping the seizure.
    Each tuple: [file_base, window_index, label=1]
    """
    dirs = build_data_dic(root)
    out = []
    for D in tqdm(dirs, desc="seiz tuples"):
        for start, end in get_seiz_times(D):
            idxs = np.arange(np.floor(start/T), np.ceil(end/T), dtype=int)
            for c in idxs:
                out.append([D, c, 1])
    return out

def Bckg_tuple_gen(root, T):
    """
    Generate tuples for background (non-seizure) segments.
    For each file with no seizures, create tuples for each T-second window.
    Each tuple: [file_base, window_index, label=0]
    """
    dirs = build_data_dic(root)
    out = []
    for D in tqdm(dirs, desc="bckg tuples"):
        if not get_seiz_times(D):
            dur = raw_eeg_loader(D).times[-1]
            max_idx = int(np.floor(dur/T) - 2)
            if max_idx > 0:
                for c in range(max_idx):
                    out.append([D, c, 0])
    return out

def make_tuple_files(dir_, root, T, train_bool):
    """
    Generate and save tuple files for seizure and background segments.
    If training, balance background samples to match seizure samples.
    """
    seiz = Seiz_tuple_gen(root, T)
    bkg  = Bckg_tuple_gen(root, T)
    if train_bool:
        random.seed(123)
        bkg = random.sample(bkg, len(seiz))  # Balance classes for training
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, 'seiz_tuple.txt'),'w') as f:
        for x in seiz: f.write(f"{x}\n")
    with open(os.path.join(dir_, 'bckg_tuple.txt'),'w') as f:
        for x in bkg:  f.write(f"{x}\n")

def save_eeg_clip(seizure_file, bckg_file, time_step, T, to_freq, h5_out, h5_lbl, seed):
    """
    For each tuple (seizure/background), extract T windows of EEG data, resample, and save as HDF5 files.
    Only includes 19 standard EEG channels. Each clip is saved with its label.
    """
    INCLUDED_CHANNELS = [
        'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG C3','EEG C4',
        'EEG P3','EEG P4','EEG O1','EEG O2','EEG F7','EEG F8',
        'EEG T3','EEG T4','EEG T5','EEG T6','EEG FZ','EEG CZ','EEG PZ'
    ]
    os.makedirs(h5_out,   exist_ok=True)
    os.makedirs(h5_lbl,   exist_ok=True)
    combined = parsetxtfile(seizure_file, bckg_file, seed)
    # Group tuples by file base
    by_file = {}
    for base, idx, lbl in combined:
        by_file.setdefault(base, []).append((idx,lbl))
    for base, items in tqdm(by_file.items(), desc="clips"):
        raw = raw_eeg_loader(base)
        # Get montage suffix from channel names
        mon = raw.ch_names[0].split('-')[1]
        chans = [f"{ch}-{mon}" for ch in INCLUDED_CHANNELS]
        # Only keep those channels present in the recording
        picks = [c for c in chans if c in raw.ch_names]
        if len(picks)<19: continue  # Skip if not all channels are present
        raw.pick_channels(picks)

        sig = raw.get_data()  # Shape: (channels, samples)
        fs  = 1/(raw.times[1]-raw.times[0])  # Original sampling frequency
        sig = resample_data(sig, to_freq, fs)  # Resample to target frequency
        for idx, lbl in items:
            start = int(idx * to_freq * T)
            # Extract T consecutive windows of length to_freq
            slc = [sig[:, start + t*to_freq:start + (t+1)*to_freq] for t in range(T)]
            if any(s.shape[1]!=to_freq for s in slc): continue  # Skip incomplete windows
            clip = np.stack(slc,0)  # Shape: (T, 19, to_freq)
            if clip.shape!=(T,19,to_freq): continue  # Skip if shape is not correct
            name = os.path.basename(base) + f"_{idx}.h5"
            with h5py.File(os.path.join(h5_out,   name),'w') as f: f.create_dataset('x', data=clip)
            with h5py.File(os.path.join(h5_lbl,   name),'w') as f: f.create_dataset('y', data=lbl)

if __name__=="__main__":
    # Main entry point: set up paths and parameters, then generate tuples and save EEG clips.
    SPLIT    = "eval"   # "train" or "eval","dev"
    DATA_ROOT= "/home/y0chen55/seizure_detect/TUSZ/edf"
    OUT_ROOT = "/home/y0chen55/seizure_detect/LLMsForTimeSeries/resampled_12s"
    T, to_freq, seed = 12, 256, 42  # T: window length (seconds), to_freq: target Hz, seed: random seed
    train_bool = (SPLIT=="train")
    tpl_dir  = os.path.join(OUT_ROOT,"scripts",f"tuples_{SPLIT}")
    edf_dir  = os.path.join(DATA_ROOT, SPLIT)
    clip_dir = os.path.join(OUT_ROOT,"clip_data", SPLIT)
    lbl_dir  = os.path.join(OUT_ROOT,"label_data", SPLIT)

    # Generate tuple files for seizure and background segments
    make_tuple_files( tpl_dir, edf_dir, T, train_bool )
    # Save EEG clips and labels as HDF5 files
    save_eeg_clip( 
        os.path.join(tpl_dir,'seiz_tuple.txt'),
        os.path.join(tpl_dir,'bckg_tuple.txt'),
        1, T, to_freq, clip_dir, lbl_dir, seed
    )
