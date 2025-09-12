import os, h5py
CLIP_DIR = "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/clip_data/eval"
LABEL_DIR= "/blue/liu.yunmei/y0chen55.louisville/shrouq/REST/label_data/eval"

missing, corrupt = [], []
for fn in sorted(os.listdir(CLIP_DIR)):
    lab = os.path.join(LABEL_DIR, fn)
    if not os.path.exists(lab):
        missing.append(fn); continue
    try:
        with h5py.File(lab, "r") as f:
            pass
    except OSError:
        corrupt.append(fn)

print(f"total clips: {len(os.listdir(CLIP_DIR))}")
print(f"missing labels: {len(missing)}")
print(f"corrupt labels: {len(corrupt)}")
