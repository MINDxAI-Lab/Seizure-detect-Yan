import os, pickle, numpy as np, torch, h5py
from datetime import datetime
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
# 新增：可视化和进度条回调
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import glob

class Config:
    """Configuration parameters for the training script."""
    # --- Data Paths ---
    CLIP_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/12s_window/clip_data"
    LABEL_ROOT = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/12s_window/label_data"
    PKL_PATH = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/REST/adj_mx_3d.pkl"
    POS_PATH = "/blue/liu.yunmei/y0chen55.louisville/seizure_detect/REST/scripts/elec_pos.npy"
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 256
    EPOCHS = 100
    FFT = True
    
    # --- Model Parameters ---
    FIRE_RATE = 0.5
    MULTI = False
    T = 12
    
    # --- Dataset Subset Ratios ---
    SUBSET_RATIO = 1.0
    SUBSET_RATIO_TEST = 1.0
    SUBSET_RATIO_EVAL = 1.0
    
    # --- Hardware & Performance ---
    DEVICES = [0]
    NUM_WORKERS = 8
    PIN_MEMORY = True
    ACCELERATOR = "gpu"
    PRECISION = 32
    
    # --- Debugging & Development ---
    DEBUG_N = None
    FAST_DEV = False
    
    # --- Logging & Callbacks ---
    LOG_EVERY_N_STEPS = 10
    CHECK_VAL_EVERY_N_EPOCH = 1
    TQDM_REFRESH_RATE = 10


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

def plot_training_curves(fig_dir="figs"):
    """绘制训练过程中的损失曲线，保存到figs目录"""
    try:
        # 查找Lightning日志文件，优先查找current目录
        metrics_file = None
        
        # 先检查当前运行的日志
        if os.path.exists("lightning_logs/current/metrics.csv"):
            metrics_file = "lightning_logs/current/metrics.csv"
        else:
            # 查找所有版本目录中的metrics.csv
            log_dirs = glob.glob("lightning_logs/version_*/")
            if log_dirs:
                # 使用最新的日志目录
                latest_log_dir = max(log_dirs, key=os.path.getctime)
                potential_metrics = os.path.join(latest_log_dir, "metrics.csv")
                if os.path.exists(potential_metrics):
                    metrics_file = potential_metrics
        
        if not metrics_file:
            print("No metrics.csv found in Lightning logs, skipping training curve plotting")
            return
        
        print(f"Reading metrics from: {metrics_file}")
        
        # 读取训练指标
        df = pd.read_csv(metrics_file)
        
        if df.empty:
            print("Metrics file is empty, skipping training curve plotting")
            return
        
        print(f"Available columns: {list(df.columns)}")
        
        # 确保figs目录存在（与现有可视化相同的目录）
        os.makedirs(fig_dir, exist_ok=True)
        
        # 绘制训练损失曲线
        plt.figure(figsize=(12, 8))
        
        # 子图1: 训练和验证损失对比
        plt.subplot(2, 2, 1)
        train_loss = df.dropna(subset=['train_loss'])
        val_loss = df.dropna(subset=['val_loss']) if 'val_loss' in df.columns else pd.DataFrame()
        
        if not train_loss.empty:
            plt.plot(train_loss['epoch'], train_loss['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        if not val_loss.empty:
            plt.plot(val_loss['epoch'], val_loss['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 学习率（如果有的话）
        plt.subplot(2, 2, 2)
        lr_data = df.dropna(subset=['lr-Adam']) if 'lr-Adam' in df.columns else None
        if lr_data is not None and not lr_data.empty:
            plt.plot(lr_data['epoch'], lr_data['lr-Adam'], 'g-', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate Schedule')
        
        # 子图3: 训练步数 vs 损失（更详细的视图）
        plt.subplot(2, 2, 3)
        if not train_loss.empty and 'step' in train_loss.columns:
            plt.plot(train_loss['step'], train_loss['train_loss'], 'r-', alpha=0.5, linewidth=0.8)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss per Step')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No step data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training Loss per Step')
        
        # 子图4: 损失的移动平均
        plt.subplot(2, 2, 4)
        if not train_loss.empty and len(train_loss) > 10:
            window = min(50, len(train_loss) // 10)
            smoothed_loss = train_loss['train_loss'].rolling(window=window, center=True).mean()
            plt.plot(train_loss['epoch'], train_loss['train_loss'], 'b-', alpha=0.3, label='Raw')
            plt.plot(train_loss['epoch'], smoothed_loss, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Smoothed Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for smoothing', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Smoothed Training Loss')
        
        plt.tight_layout()
        # 保存到figs目录，与其他可视化图表在同一位置
        training_curves_path = os.path.join(fig_dir, "training_curves.png")
        plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {training_curves_path}")
        
        # 验证文件确实保存了
        if os.path.exists(training_curves_path):
            file_size = os.path.getsize(training_curves_path)
            print(f"Training curves file size: {file_size} bytes")
        else:
            print("Warning: Training curves file was not created!")
        
        # 绘制F1分数曲线
        train_f1 = df.dropna(subset=['train_f1']) if 'train_f1' in df.columns else pd.DataFrame()
        val_f1 = df.dropna(subset=['val_f1']) if 'val_f1' in df.columns else pd.DataFrame()
        
        if not train_f1.empty or not val_f1.empty:
            plt.figure(figsize=(12, 8))
            
            # 子图1: 训练和验证F1对比
            plt.subplot(2, 2, 1)
            if not train_f1.empty:
                plt.plot(train_f1['epoch'], train_f1['train_f1'], 'b-', label='Train F1', alpha=0.7)
            if not val_f1.empty:
                plt.plot(val_f1['epoch'], val_f1['val_f1'], 'r-', label='Val F1', alpha=0.7)
            
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Training and Validation F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)  # F1分数范围在0-1之间
            
            # 子图2: F1分数的移动平均
            plt.subplot(2, 2, 2)
            if not train_f1.empty and len(train_f1) > 10:
                window = min(50, len(train_f1) // 10)
                smoothed_f1 = train_f1['train_f1'].rolling(window=window, center=True).mean()
                plt.plot(train_f1['epoch'], train_f1['train_f1'], 'b-', alpha=0.3, label='Raw Train F1')
                plt.plot(train_f1['epoch'], smoothed_f1, 'b-', linewidth=2, label=f'Smoothed Train F1 (window={window})')
            if not val_f1.empty and len(val_f1) > 10:
                window = min(50, len(val_f1) // 10)
                smoothed_val_f1 = val_f1['val_f1'].rolling(window=window, center=True).mean()
                plt.plot(val_f1['epoch'], val_f1['val_f1'], 'r-', alpha=0.3, label='Raw Val F1')
                plt.plot(val_f1['epoch'], smoothed_val_f1, 'r-', linewidth=2, label=f'Smoothed Val F1 (window={window})')
            
            if not train_f1.empty or not val_f1.empty:
                plt.xlabel('Epoch')
                plt.ylabel('F1 Score')
                plt.title('Smoothed F1 Scores')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
            else:
                plt.text(0.5, 0.5, 'No F1 data available for smoothing', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Smoothed F1 Scores')
            
            # 子图3: 训练步数 vs F1（如果有step数据）
            plt.subplot(2, 2, 3)
            if not train_f1.empty and 'step' in train_f1.columns:
                plt.plot(train_f1['step'], train_f1['train_f1'], 'g-', alpha=0.5, linewidth=0.8)
                plt.xlabel('Training Step')
                plt.ylabel('F1 Score')
                plt.title('Training F1 Score per Step')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
            else:
                plt.text(0.5, 0.5, 'No step F1 data available', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Training F1 Score per Step')
            
            # 子图4: F1分数改善趋势
            plt.subplot(2, 2, 4)
            if not train_f1.empty and not val_f1.empty:
                # 计算最佳F1分数
                best_train_f1 = train_f1['train_f1'].max()
                best_val_f1 = val_f1['val_f1'].max()
                
                plt.bar(['Train F1', 'Val F1'], [best_train_f1, best_val_f1], 
                       color=['blue', 'red'], alpha=0.7)
                plt.ylabel('Best F1 Score')
                plt.title('Best F1 Scores Achieved')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # 添加数值标签
                plt.text(0, best_train_f1 + 0.02, f'{best_train_f1:.3f}', ha='center', va='bottom')
                plt.text(1, best_val_f1 + 0.02, f'{best_val_f1:.3f}', ha='center', va='bottom')
            else:
                plt.text(0.5, 0.5, 'Insufficient F1 data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Best F1 Scores Achieved')
            
            plt.tight_layout()
            # 保存F1分数图表
            f1_curves_path = os.path.join(fig_dir, "f1_curves.png")
            plt.savefig(f1_curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"F1 curves saved to: {f1_curves_path}")
            
            # 验证F1图表文件确实保存了
            if os.path.exists(f1_curves_path):
                file_size = os.path.getsize(f1_curves_path)
                print(f"F1 curves file size: {file_size} bytes")
            else:
                print("Warning: F1 curves file was not created!")
        else:
            print("No F1 score data found in metrics, skipping F1 curves plotting")
        
        # 打印训练统计信息
        if not train_loss.empty:
            print(f"Training Statistics:")
            print(f"  - Initial train loss: {train_loss['train_loss'].iloc[0]:.4f}")
            print(f"  - Final train loss: {train_loss['train_loss'].iloc[-1]:.4f}")
            print(f"  - Best train loss: {train_loss['train_loss'].min():.4f}")
            print(f"  - Total epochs: {train_loss['epoch'].max():.0f}")
        
        if not val_loss.empty:
            print(f"Validation Statistics:")
            print(f"  - Initial val loss: {val_loss['val_loss'].iloc[0]:.4f}")
            print(f"  - Final val loss: {val_loss['val_loss'].iloc[-1]:.4f}")
            print(f"  - Best val loss: {val_loss['val_loss'].min():.4f}")
        
        # 打印F1分数统计信息
        train_f1 = df.dropna(subset=['train_f1']) if 'train_f1' in df.columns else pd.DataFrame()
        val_f1 = df.dropna(subset=['val_f1']) if 'val_f1' in df.columns else pd.DataFrame()
        
        if not train_f1.empty:
            print(f"Training F1 Statistics:")
            print(f"  - Initial train F1: {train_f1['train_f1'].iloc[0]:.4f}")
            print(f"  - Final train F1: {train_f1['train_f1'].iloc[-1]:.4f}")
            print(f"  - Best train F1: {train_f1['train_f1'].max():.4f}")
        
        if not val_f1.empty:
            print(f"Validation F1 Statistics:")
            print(f"  - Initial val F1: {val_f1['val_f1'].iloc[0]:.4f}")
            print(f"  - Final val F1: {val_f1['val_f1'].iloc[-1]:.4f}")
            print(f"  - Best val F1: {val_f1['val_f1'].max():.4f}")
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")
        print("Generating a fallback training curves plot...")
        
        # 生成一个简单的占位图表
        try:
            os.makedirs(fig_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Training curves not available\n(No valid metrics.csv found)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title('Training Curves - Data Not Available')
            plt.axis('off')
            training_curves_path = os.path.join(fig_dir, "training_curves.png")
            plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Fallback training curves saved to: {training_curves_path}")
        except:
            print("Failed to create fallback training curves plot")

def _select_subset(files, ratio):
    if ratio >= 1.0 or len(files) == 0:
        return files
    k = max(int(np.ceil(len(files) * ratio)), 1)
    # 随机无放回抽样，保持输出顺序稳定
    rng = np.random.default_rng()
    idx = rng.choice(len(files), size=k, replace=False)
    idx.sort()
    return [files[i] for i in idx]

def load_dataset(split, ew, ei, pos, use_fft=True, subset_ratio=1.0):
    cd, ld = os.path.join(Config.CLIP_ROOT, split), os.path.join(Config.LABEL_ROOT, split)
    files  = sorted(f for f in os.listdir(cd) if f.endswith(".h5"))
    if Config.DEBUG_N:
        files = files[:Config.DEBUG_N]
    # 先筛选出存在对应标签文件的样本，避免出现空数据集
    files = [f for f in files if os.path.exists(os.path.join(ld, f))]
    # 抽样：训练用传入的 subset_ratio，测试固定 0.1，验证集不抽样
    if split == "train":
        files = _select_subset(files, subset_ratio)
    elif split == "test":
        files = _select_subset(files, Config.SUBSET_RATIO_TEST)
    elif split == "eval":
        files = _select_subset(files, Config.SUBSET_RATIO_EVAL)

    data = []
    for fn in tqdm(files, desc=f"{split} (subset={'{:.2f}'.format(subset_ratio) if split=='train' else '{:.2f}'.format(Config.SUBSET_RATIO_EVAL) if split=='eval' else ('{:.2f}'.format(Config.SUBSET_RATIO_EVAL) if split=='test' else '1.0')})"):
        lp = os.path.join(ld, fn)
        try:
            with h5py.File(lp, 'r') as f:
                key = list(f.keys())[0]  # 获取第一个键（文件名）
                lbl_data = f[key]  # 获取标签数据
                # 标签文件包含标量值，直接读取
                lbl = int(np.array(lbl_data))
        except Exception:
            continue
        
        try:
            with h5py.File(os.path.join(cd, fn), 'r') as f:
                key = list(f.keys())[0]  # 获取第一个键（文件名）
                eeg = np.array(f[key])  # 直接读取数据，不使用[0]索引
        except Exception:
            continue
            
        if use_fft:
            eeg = np.log(np.abs(fft(eeg, axis=2)[:, :, :100]) + 1e-30)
        data.append(Data(x=torch.tensor(eeg, dtype=torch.float32).transpose(1,0), y=torch.tensor([lbl], dtype=torch.long),
                         edge_weight=ew, edge_index=ei, elec_pos=pos))
    return data

def train_subset():
    ew, ei = make_adj_matrix(Config.PKL_PATH)
    pos    = load_pos(Config.POS_PATH)
    tr_ds  = load_dataset("train", ew, ei, pos, Config.FFT, subset_ratio=Config.SUBSET_RATIO)
    va_ds  = load_dataset("eval",  ew, ei, pos, Config.FFT, subset_ratio=Config.SUBSET_RATIO_EVAL)

    print(f"Loaded datasets: train={len(tr_ds)} | eval={len(va_ds)}")
    if len(tr_ds) == 0:
        print("[Error] Empty training dataset after filtering. 请检查训练集是否有对应的标签文件或提高抽样比例。")
        return
    if len(va_ds) == 0:
        print("[Warning] Empty eval dataset after filtering. 将跳过评估阶段。")

    tr_ld  = DataLoader(tr_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    va_ld  = DataLoader(va_ds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY) if len(va_ds) > 0 else None
    model  = LitREST(fire_rate=Config.FIRE_RATE, multi=Config.MULTI, T=Config.T)
    
    # 添加CSV Logger以生成metrics.csv文件
    csv_logger = CSVLogger("lightning_logs", name="", version="current")
    
    trainer = L.Trainer(max_epochs=Config.EPOCHS, devices=Config.DEVICES, accelerator=Config.ACCELERATOR, precision=Config.PRECISION,
                        strategy=DDPStrategy(find_unused_parameters=True), fast_dev_run=Config.FAST_DEV,
                        log_every_n_steps=Config.LOG_EVERY_N_STEPS, check_val_every_n_epoch=Config.CHECK_VAL_EVERY_N_EPOCH,
                        logger=csv_logger,
                        callbacks=[TQDMProgressBar(refresh_rate=Config.TQDM_REFRESH_RATE)])
    trainer.fit(model, tr_ld, va_ld)
    

    # 获取当前时间戳，创建唯一子文件夹
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_dir = os.path.join("figs", now_str)
    os.makedirs(fig_dir, exist_ok=True)

    # 绘制训练过程可视化
    plot_training_curves(fig_dir=fig_dir)


    if va_ld is None:
        torch.save(model.state_dict(), "trained_rest.pt")
        return

    # 评估阶段（带进度条）
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in tqdm(va_ld, desc="Evaluating", leave=False):
            b = b.to(model.device)
            o = global_mean_pool(model(b), b.batch)
            preds.append(torch.sigmoid(o).cpu())
            gts.append(b.y.float())

    # 计算指标
    y_true = torch.cat(gts).view(-1).cpu().numpy()
    y_prob = torch.cat(preds).view(-1).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    auc = metrics.roc_auc_score(y_true, y_prob)
    ap  = metrics.average_precision_score(y_true, y_prob)
    f1  = metrics.f1_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)

    print(f"Validation Metrics: AUC={auc:.4f} | AP={ap:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Precision={pre:.4f} | Acc={acc:.4f}")


    # 可视化保存到唯一子文件夹
    # ROC 曲线
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, "roc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # PR 曲线
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall_curve, precision_curve, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(fig_dir, "pr.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # 混淆矩阵
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 分数直方图（按类别）
    plt.figure()
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Negative")
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Positive")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(fig_dir, "score_hist.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"Saved figures to: {roc_path}, {pr_path}, {cm_path}, {hist_path}")
    print(f"Training curves also saved to: {os.path.join(fig_dir, 'training_curves.png')}")

    torch.save(model.state_dict(), "trained_rest.pt")

if __name__ == "__main__":
    train_subset()
