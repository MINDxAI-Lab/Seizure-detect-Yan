import numpy as np
import mne
from scipy.signal import resample
import glob
import os
from tqdm import tqdm
import random
import h5py

def build_data_directory_patient(train_set_directory, patient):
    base = os.path.join(train_set_directory, patient)
    out = []
    for ses in os.listdir(base):
        ses_dir = os.path.join(base, ses)
        if not os.path.isdir(ses_dir): continue
        # assume one montage subfolder per session
        montages = [d for d in os.listdir(ses_dir) if os.path.isdir(os.path.join(ses_dir, d))]
        if not montages: continue
        data_dir = os.path.join(ses_dir, montages[0])
        # find all EDFs
        for edf in glob.glob(os.path.join(data_dir, '*.edf')):
            out.append(edf[:-4])
    return np.unique(out)

def build_data_dic(train_set_directory):
    all_dirs = []
    for p in tqdm(os.listdir(train_set_directory), desc="patients"):
        path = os.path.join(train_set_directory, p)
        if os.path.isdir(path):
            all_dirs.extend(build_data_directory_patient(train_set_directory, p))
    return np.array(all_dirs)

def raw_eeg_loader(path_base):
    return mne.io.read_raw_edf(path_base + '.edf', verbose=False)

def resample_data(signals, to_freq, fs):
    N = signals.shape[1]
    num = int(to_freq * (N / fs))
    return resample(signals, num=num, axis=1)

def get_seiz_times(path_base):
    out = []
    with open(path_base + '.csv_bi') as f:
        for L in f:
            if 'seiz' in L:
                parts = L.split(',')
                out.append((float(parts[1]), float(parts[2])))
    return out

def parsetxtfile(seizure_file, bckg_file, seed):
    random.seed(seed)
    def _read(fn):
        lines = open(fn).read().splitlines()
        tuples = []
        for L in lines:
            lst = L.strip()[1:-1].split(',')
            tuples.append([eval(lst[0]), int(lst[1]), int(lst[2])])
        return tuples
    combined = _read(bckg_file) + _read(seizure_file)
    random.shuffle(combined)
    return combined

def Seiz_tuple_gen(root, T):
    dirs = build_data_dic(root)
    out = []
    for D in tqdm(dirs, desc="seiz tuples"):
        for start, end in get_seiz_times(D):
            idxs = np.arange(np.floor(start/T), np.ceil(end/T), dtype=int)
            for c in idxs:
                out.append([D, c, 1])
    return out

def Bckg_tuple_gen(root, T):
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
    seiz = Seiz_tuple_gen(root, T)
    bkg  = Bckg_tuple_gen(root, T)
    if train_bool:
        random.seed(123)
        bkg = random.sample(bkg, len(seiz))
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, 'seiz_tuple.txt'),'w') as f:
        for x in seiz: f.write(f"{x}\n")
    with open(os.path.join(dir_, 'bckg_tuple.txt'),'w') as f:
        for x in bkg:  f.write(f"{x}\n")

def save_eeg_clip(seizure_file, bckg_file, time_step, T, to_freq, h5_out, h5_lbl, seed):
    INCLUDED_CHANNELS = [
        'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG C3','EEG C4',
        'EEG P3','EEG P4','EEG O1','EEG O2','EEG F7','EEG F8',
        'EEG T3','EEG T4','EEG T5','EEG T6','EEG FZ','EEG CZ','EEG PZ'
    ]
    os.makedirs(h5_out,   exist_ok=True)
    os.makedirs(h5_lbl,   exist_ok=True)
    combined = parsetxtfile(seizure_file, bckg_file, seed)
    # group by file base
    by_file = {}
    for base, idx, lbl in combined:
        by_file.setdefault(base, []).append((idx,lbl))
    for base, items in tqdm(by_file.items(), desc="clips"):
        raw = raw_eeg_loader(base)
        mon = raw.ch_names[0].split('-')[1]
        chans = [f"{ch}-{mon}" for ch in INCLUDED_CHANNELS]
        # only keep those channels present
        picks = [c for c in chans if c in raw.ch_names]
        if len(picks)<19: continue
        raw.pick_channels(picks)

        sig = raw.get_data()
        fs  = 1/(raw.times[1]-raw.times[0])
        sig = resample_data(sig, to_freq, fs)
        for idx, lbl in items:
            start = int(idx * to_freq * T)
            slc = [sig[:, start + t*to_freq:start + (t+1)*to_freq] for t in range(T)]
            if any(s.shape[1]!=to_freq for s in slc): continue
            clip = np.stack(slc,0)
            if clip.shape!=(T,19,to_freq): continue
            name = os.path.basename(base) + f"_{idx}.h5"
            with h5py.File(os.path.join(h5_out,   name),'w') as f: f.create_dataset('x', data=clip)
            with h5py.File(os.path.join(h5_lbl,   name),'w') as f: f.create_dataset('y', data=lbl)

if __name__=="__main__":
    SPLIT    = "eval"   # "train" or "eval","dev"
    DATA_ROOT= "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/TUSZ/edf"
    OUT_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/12s_window"
    T, to_freq, seed = 12, 200, 42
    train_bool = (SPLIT=="train")
    tpl_dir  = os.path.join(OUT_ROOT,"scripts",f"tuples_{SPLIT}")
    edf_dir  = os.path.join(DATA_ROOT, SPLIT)
    clip_dir = os.path.join(OUT_ROOT,"clip_data", SPLIT)
    lbl_dir  = os.path.join(OUT_ROOT,"label_data", SPLIT)

    make_tuple_files( tpl_dir, edf_dir, T, train_bool )
    save_eeg_clip( 
        os.path.join(tpl_dir,'seiz_tuple.txt'),
        os.path.join(tpl_dir,'bckg_tuple.txt'),
        1, T, to_freq, clip_dir, lbl_dir, seed
    )
