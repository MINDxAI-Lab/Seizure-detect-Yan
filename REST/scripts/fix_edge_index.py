import h5py
import numpy as np
import os

def check_h5_file(eeg_path, label_path, name):
    print(f"\n🔎 Checking {name} data")

    if not os.path.exists(eeg_path):
        print(f"❌ EEG file not found: {eeg_path}")
        return
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return

    with h5py.File(eeg_path, "r") as f_eeg, h5py.File(label_path, "r") as f_label:
        if "eeg" not in f_eeg:
            print("❌ 'eeg' key not found in EEG file.")
            return
        if "labels" not in f_label:
            print("❌ 'labels' key not found in label file.")
            return

        EEG = f_eeg["eeg"]
        Labels = f_label["labels"]

        print(f"✅ EEG shape: {EEG.shape}")
        print(f"✅ Label shape: {Labels.shape}")

        if EEG.shape[0] != Labels.shape[0]:
            print(f"❌ Mismatch between EEG samples and Labels: {EEG.shape[0]} != {Labels.shape[0]}")

        # Check per sample
        valid_samples = 0
        for i in range(min(EEG.shape[0], 100)):  # فقط أول 100 عينة
            eeg = EEG[i]
            if eeg.ndim == 2:
                eeg = eeg[:, np.newaxis, :]
            elif eeg.ndim == 3 and eeg.shape[1] != 1:
                eeg = eeg[:, np.newaxis, :]

            if eeg.shape[0] != 19:
                print(f"❌ Sample {i}: expected 19 channels, got {eeg.shape[0]}")
            if eeg.shape[-1] < 100:
                print(f"⚠️ Sample {i}: skipped (too short: {eeg.shape[-1]} points)")
                continue

            valid_samples += 1

        print(f"✅ Valid samples with enough length: {valid_samples}/{EEG.shape[0]}")

def check_numpy(path, name):
    if not os.path.exists(path):
        print(f"❌ {name} file not found: {path}")
        return
    try:
        data = np.load(path)
        print(f"✅ {name} loaded. Shape: {data.shape}, Dtype: {data.dtype}")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

def check_mat(path):
    import scipy.io as sio
    try:
        mat = sio.loadmat(path)
        if "adj_mat" in mat:
            adj = mat["adj_mat"]
            print(f"✅ adj_mat.mat loaded. Shape: {adj.shape}")
        else:
            print("❌ 'adj_mat' not found in .mat file.")
    except Exception as e:
        print(f"❌ Error loading adj_mat.mat: {e}")

if __name__ == "__main__":
    base = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/graph_data"
    elec_pos_path = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/scripts/elec_pos.npy"
    adj_mat_path = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/scripts/adj_mat.mat"

    check_h5_file(f"{base}/clip_data_train.h5", f"{base}/label_train.h5", "Train")
    check_h5_file(f"{base}/clip_data_val.h5", f"{base}/label_val.h5", "Validation")
    check_h5_file(f"{base}/clip_data_test.h5", f"{base}/label_test.h5", "Test")

    check_numpy(elec_pos_path, "elec_pos.npy")
    check_mat(adj_mat_path)
