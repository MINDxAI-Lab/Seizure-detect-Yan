import os, pickle, numpy as np, torch, h5py
from tqdm import tqdm
from scipy.fft import fft
from sklearn import metrics
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from REST import LitREST

CLIP_ROOT  = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/clip_data"
LABEL_ROOT = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/label_data"
PKL_PATH   = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/REST/adj_mx_3d.pkl"
POS_PATH   = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/scripts/elec_pos.npy"

BATCH_SIZE = 256
EPOCHS     = 100
DEVICES    = [0]
FFT        = True
DEBUG_N    = None
FAST_DEV   = False

def make_adj_matrix(pkl_file):
    obj = pickle.load(open(pkl_file, "rb"))
    if isinstance(obj, dict) and "adj_mat" in obj:
        adj = obj["adj_mat"]
    elif isinstance(obj, list):
        adj = np.asarray([x for x in obj if np.asarray(x).ndim >= 2][0], dtype=np.float32)
    else:
        adj = np.asarray(obj, dtype=np.float32)
    if adj.ndim == 3:
        adj = adj[:, :, 0]
    ei, ew = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))
    return ew.unsqueeze(0), ei

load_pos = lambda p: torch.tensor(np.load(p), dtype=torch.float32)

def load_dataset(split, ew, ei, pos, use_fft=True):
    cd, ld = os.path.join(CLIP_ROOT, split), os.path.join(LABEL_ROOT, split)
    files  = sorted(f for f in os.listdir(cd) if f.endswith(".h5"))
    if DEBUG_N: files = files[:DEBUG_N]
    data = []
    for fn in tqdm(files, desc=split):
        lp = os.path.join(ld, fn)
        if not os.path.exists(lp): continue
        try: lbl = int(np.array(h5py.File(lp)["/"][0]))
        except Exception: continue
        eeg = np.array(h5py.File(os.path.join(cd, fn))["/"][0])
        if use_fft: eeg = np.log(np.abs(fft(eeg, axis=2)[:, :, :100]) + 1e-30)
        data.append(Data(x=torch.tensor(eeg).transpose(1,0), y=torch.tensor([lbl]),
                         edge_weight=ew, edge_index=ei, elec_pos=pos))
    return data

def train():
    ew, ei = make_adj_matrix(PKL_PATH)
    pos    = load_pos(POS_PATH)
    tr_ds  = load_dataset("train", ew, ei, pos, FFT)
    va_ds  = load_dataset("eval",  ew, ei, pos, FFT)
    tr_ld  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    va_ld  = DataLoader(va_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    model  = LitREST(fire_rate=0.5, multi=False, T=1)
    L.Trainer(max_epochs=EPOCHS, devices=DEVICES, accelerator="gpu", precision=32,
              strategy=DDPStrategy(find_unused_parameters=True), fast_dev_run=FAST_DEV,
              log_every_n_steps=10, check_val_every_n_epoch=1).fit(model, tr_ld, va_ld)
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in va_ld:
            b = b.to(model.device)
            o = global_mean_pool(model(b), b.batch)
            preds.append(torch.sigmoid(o).cpu())
            gts.append(b.y.float())
    print("Validation AUC:", metrics.roc_auc_score(torch.cat(gts), torch.cat(preds)))
    torch.save(model.state_dict(), "trained_rest.pt")

if __name__ == "__main__":
    train()
