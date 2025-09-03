#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, h5py, argparse, sys
from pathlib import Path
from tqdm import tqdm
from io import StringIO

# ---------- helpers for labels ----------
def _find_label_sidecar(edf_path: str):
    p = Path(edf_path)
    tse = p.with_suffix(".tse_bi")
    csv_bi = p.with_suffix(".csv_bi")
    if tse.exists(): return str(tse), "tse"
    if csv_bi.exists(): return str(csv_bi), "csv"
    cand = list(p.parent.glob(p.stem + "*.tse_bi"))
    if cand: return str(cand[0]), "tse"
    cand = list(p.parent.glob(p.stem + "*.csv_bi"))
    if cand: return str(cand[0]), "csv"
    return None, None

def _parse_intervals_from_tse(tse_path: str):
    iv = []
    with open(tse_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())
            parts = [p for p in parts if p]
            if len(parts) < 3: continue
            try:
                if not re.search(r"seiz", parts[0], re.IGNORECASE): continue
                floats = [float(x) for x in parts if re.match(r"^[+-]?\d+(\.\d+)?$", x)]
                if len(floats) >= 2 and floats[1] > floats[0]:
                    iv.append((floats[0], floats[1]))
            except Exception:
                continue
    return iv

def _parse_intervals_from_csv(csv_path: str):
    iv = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        # Skip comment lines starting with #
        lines = []
        for line in f:
            line = line.strip()
            if not line.startswith('#') and line:
                lines.append(line)
        
        if not lines:
            return iv
            
        # Create a StringIO object from the filtered lines
        from io import StringIO
        csv_content = StringIO('\n'.join(lines))
        
        try:
            reader = csv.DictReader(csv_content)
            if reader.fieldnames is None:
                return iv
                
            # Look for time and label columns
            c_on  = [c for c in reader.fieldnames if "start" in c.lower()]
            c_off = [c for c in reader.fieldnames if "end" in c.lower() or "stop" in c.lower()]
            c_evt = [c for c in reader.fieldnames if "label" in c.lower()]
            
            if not (c_on and c_off and c_evt):
                return iv
                
            for row in reader:
                try:
                    start_time = float(row[c_on[0]])
                    stop_time = float(row[c_off[0]])
                    label = str(row[c_evt[0]]).lower()
                    
                    # Check if this is a seizure event
                    if stop_time > start_time and ('seiz' in label or 'sz' in label):
                        iv.append((start_time, stop_time))
                except (ValueError, KeyError, IndexError):
                    continue
        except Exception:
            pass
            
    return iv

def _has_overlap(t0, t1, intervals, min_overlap=0.0):
    for a, b in intervals:
        ov = max(0.0, min(t1, b) - max(t0, a))
        if ov > min_overlap: return True
    return False

def _infer_split_from_edf_path(edf_path: str):
    s = edf_path.lower()
    if "/dev/"  in s: return "dev"
    if "/eval/" in s or "/test/" in s: return "test"
    if "/train/" in s: return "train"
    return "train"

def build_edf_index(raw_root: str):
    """一次性建立 basename -> EDF 完整路径 的索引，并按 train>dev>eval 优先。"""
    print(f"Building EDF index from: {raw_root}")
    if not Path(raw_root).exists():
        print(f"ERROR: Raw data directory does not exist: {raw_root}")
        sys.exit(1)
        
    idx = {}
    edf_files = list(Path(raw_root).rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")
    
    if len(edf_files) == 0:
        print("ERROR: No EDF files found!")
        sys.exit(1)
        
    for p in tqdm(edf_files, desc="Indexing EDFs", unit="edf"):
        stem = p.stem
        s = str(p).lower()
        pri = 0
        if "/train/" in s: pri = 3
        elif "/dev/" in s: pri = 2
        elif "/eval/" in s or "/test/" in s: pri = 1
        if stem not in idx or pri > idx[stem][0]:
            idx[stem] = (pri, str(p))
    
    result = {k: v for k, (_, v) in idx.items()}
    print(f"Built index with {len(result)} unique basenames")
    return result

# ---------- main ----------
def main(resampled_dir: str, raw_data_dir: str, output_dir: str, clip_len: int):
    print(f"Starting file marker generation...")
    print(f"Resampled dir: {resampled_dir}")
    print(f"Raw data dir: {raw_data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Clip length: {clip_len}s")
    
    resampled_dir = Path(resampled_dir)
    out_dir = Path(output_dir)
    
    if not resampled_dir.exists():
        print(f"ERROR: Resampled directory does not exist: {resampled_dir}")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件（对齐原 repo 命名）
    print("Opening output files...")
    f_train_sz = open(out_dir / f"trainSet_seq2seq_{clip_len}s_sz.txt",  "w", encoding="utf-8")
    f_train_ns = open(out_dir / f"trainSet_seq2seq_{clip_len}s_nosz.txt","w", encoding="utf-8")
    f_dev_sz   = open(out_dir / f"devSet_seq2seq_{clip_len}s_sz.txt",    "w", encoding="utf-8")
    f_dev_ns   = open(out_dir / f"devSet_seq2seq_{clip_len}s_nosz.txt",  "w", encoding="utf-8")
    f_test_sz  = open(out_dir / f"testSet_seq2seq_{clip_len}s_sz.txt",   "w", encoding="utf-8")
    f_test_ns  = open(out_dir / f"testSet_seq2seq_{clip_len}s_nosz.txt", "w", encoding="utf-8")
    writers = {
        ("train",1): f_train_sz, ("train",0): f_train_ns,
        ("dev",1):   f_dev_sz,   ("dev",0):   f_dev_ns,
        ("test",1):  f_test_sz,  ("test",0):  f_test_ns,
    }

    # 列出重采样 H5；建立 EDF 索引（带进度条）
    h5_list = sorted([p for p in resampled_dir.glob("*.h5")])
    print(f"Found {len(h5_list)} resampled H5 files.")
    
    if len(h5_list) == 0:
        print("ERROR: No H5 files found in resampled directory!")
        sys.exit(1)
    
    edf_idx = build_edf_index(raw_data_dir)

    n_written = {"train": [0,0], "dev": [0,0], "test": [0,0]}  # [nosz, sz]
    seizure_files_found = 0
    total_seizure_intervals = 0
    processed_files = 0
    skipped_files = 0

    print("Processing H5 files...")
    for h5p in tqdm(h5_list, desc="Generating markers", unit="file"):
        base = h5p.stem  # aaaaaaaq_s006_t000
        edf_full = edf_idx.get(base)
        if not edf_full:
            skipped_files += 1
            if skipped_files <= 5:  # Show first 5 skipped files
                print(f"WARNING: No matching EDF found for {base}")
            continue
            
        split = _infer_split_from_edf_path(edf_full)

        # 读取重采样信号长度与采样率
        try:
            with h5py.File(h5p, "r") as f:
                sig = f["resampled_signal"][()]   # (C, T)
                fs  = int(f["resample_freq"][()]) # 期望为 200 Hz
        except Exception as e:
            print(f"ERROR reading {h5p}: {e}")
            continue

        T_sec = sig.shape[1] // fs
        if T_sec < clip_len:
            continue
        n_clips = T_sec // clip_len  # 12s 非重叠

        # 解析真实标签区间（缓存按 EDF 做，因每个 H5 仅对应一个 EDF，这里直接读取一次）
        lb_path, lb_kind = _find_label_sidecar(edf_full)
        intervals = []
        if lb_path:
            intervals = _parse_intervals_from_tse(lb_path) if lb_kind == "tse" else _parse_intervals_from_csv(lb_path)
            if intervals:
                seizure_files_found += 1
                total_seizure_intervals += len(intervals)
                if seizure_files_found <= 5:  # Show first 5 seizure files
                    print(f"Found {len(intervals)} seizure intervals in {base} ({lb_kind})")

        # 写出占位名：<basename>.edf_<idx>.h5,label
        for idx_clip in range(n_clips):
            w0, w1 = idx_clip * clip_len, (idx_clip + 1) * clip_len
            label = 1 if _has_overlap(w0, w1, intervals, min_overlap=0.0) else 0
            placeholder = f"{base}.edf_{idx_clip}.h5"
            writers[(split, label)].write(f"{placeholder},{label}\n")
            n_written[split][label] += 1
            
        processed_files += 1

    for f in writers.values(): f.close()

    print("="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total H5 files found: {len(h5_list)}")
    print(f"Files processed: {processed_files}")
    print(f"Files skipped (no matching EDF): {skipped_files}")
    print("Written (non-seizure, seizure) per split:")
    for s in ("train","dev","test"):
        print(f"  {s}: {tuple(n_written[s])}")
    print(f"Total files with seizure annotations: {seizure_files_found}")
    print(f"Total seizure intervals found: {total_seizure_intervals}")
    print(f"Markers saved under: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate file markers directly from resampled H5 (no preprocess).")
    ap.add_argument("--resampled_dir", type=str, required=True, help="Directory containing resampled H5 (one per EDF).")
    ap.add_argument("--raw_data_dir", type=str, required=True, help="Root of raw TUSZ to locate EDF and labels.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output dir for marker txt files.")
    ap.add_argument("--clip_len", type=int, default=12, help="Clip length seconds (12 or 60).")
    args = ap.parse_args()
    
    try:
        main(args.resampled_dir, args.raw_data_dir, args.output_dir, args.clip_len)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
