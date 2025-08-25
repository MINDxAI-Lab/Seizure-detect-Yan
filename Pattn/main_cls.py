from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from tqdm import tqdm
from models.PAttn import PAttn

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import time
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import numpy as np
from datetime import datetime

import argparse
import random
    
warnings.filterwarnings('ignore')

fix_seed = 2021
# Set random seeds for reproducibility
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='PAttn Binary Classification')
# Argument parser for all model, data, and augmentation parameters

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, default='data.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='label')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=3072)  # 12 seconds * 256 Hz for EEG
parser.add_argument('--label_len', type=int, default=0)  # Not needed for classification
parser.add_argument('--pred_len', type=int, default=0)   # Not needed for classification

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=19)  # Number of input channels (19 for EEG)
parser.add_argument('--patch_size', type=int, default=16)

parser.add_argument('--model', type=str, default='PAttn')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--save_file_name', type=str, default='cls_results.txt')
parser.add_argument('--gpu_loc', type=int, default=0)
parser.add_argument('--method', type=str, default='PAttn')

## ---- Data Augmentation Parameters ----
parser.add_argument('--aug_method', type=str, default='none', choices=['none', 'decomp', 'ww', 'both'], 
                   help='Data augmentation method: none, decomp (decomposition), ww (window warping), both')

# Decomposition augmentation parameters
parser.add_argument('--aug_p', type=float, default=0.5, help='Probability of applying decomposition augmentation')
parser.add_argument('--aug_win', type=int, default=129, help='Window size for moving average trend decomposition (~0.5s @256Hz)')
parser.add_argument('--aug_scale_low', type=float, default=0.9, help='Lower bound for residual scaling')
parser.add_argument('--aug_scale_high', type=float, default=1.1, help='Upper bound for residual scaling')
parser.add_argument('--aug_noise', type=float, default=0.02, help='Noise ratio relative to residual standard deviation')
parser.add_argument('--aug_only_bg', action='store_true', help='Apply decomposition augmentation only to background (non-seizure) samples')

# Window Warping augmentation parameters
parser.add_argument('--aug_ww_p_low', type=float, default=0.3, help='Lower bound for WW trigger probability')
parser.add_argument('--aug_ww_p_high', type=float, default=0.7, help='Upper bound for WW trigger probability')
parser.add_argument('--aug_ww_win_ratio_low', type=float, default=0.1, help='Lower bound for sub-window ratio')
parser.add_argument('--aug_ww_win_ratio_high', type=float, default=0.3, help='Upper bound for sub-window ratio')
parser.add_argument('--aug_ww_speed_low', type=float, default=0.8, help='Lower bound for time warping speed ratio')
parser.add_argument('--aug_ww_speed_high', type=float, default=1.2, help='Upper bound for time warping speed ratio')
parser.add_argument('--aug_ww_margin', type=float, default=0.5, help='Margin from boundaries in seconds')
parser.add_argument('--aug_ww_only_bg', action='store_true', help='Apply WW augmentation only to background (non-seizure) samples')

# Loss function parameters
parser.add_argument('--pos_weight', type=float, default=None, help='Positive class weight for BCEWithLogitsLoss (for handling class imbalance)')
parser.add_argument('--auto_pos_weight', action='store_true', help='Automatically calculate pos_weight based on class distribution in training data')

args = parser.parse_args()

## Set augmentation flags based on aug_method
args.aug_decomp = args.aug_method in ['decomp', 'both']
args.aug_ww = args.aug_method in ['ww', 'both']

device_address = 'cuda:'+str(args.gpu_loc)

def select_optimizer(model, args):
    """
    Select and return the optimizer for model training.
    Currently uses Adam optimizer with the specified learning rate.
    """
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model_optim

def calculate_pos_weight(train_loader, device):
    """
    Calculate the positive class weight for BCEWithLogitsLoss based on the class distribution in the training data.
    Returns pos_weight value (float).
    """
    total_positive = 0
    total_negative = 0
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_y = batch_y.float()
        total_positive += batch_y.sum().item()
        total_negative += (batch_y == 0).sum().item()
    
    if total_positive == 0:
        print("Warning: No positive samples found in training data!")
        return 1.0
    
    pos_weight = total_negative / total_positive
    print(f"Class distribution - Negative: {total_negative}, Positive: {total_positive}")
    print(f"Calculated pos_weight: {pos_weight:.4f}")
    
    return pos_weight

def plot_training_curves(save_dir, training_history, fig_dir="figs"):
    """
    Plot and save training curves (loss, accuracy, F1, learning rate) for each epoch.
    Saves figures to a timestamped subdirectory. (Will replace this part with Tensor Board, need more time)
    """
    try:
        # 创建带时间戳的子目录
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_fig_dir = os.path.join(fig_dir, now_str)
        os.makedirs(full_fig_dir, exist_ok=True)
        
        if not training_history or len(training_history) == 0:
            print("No training history available, skipping training curve plotting")
            return full_fig_dir
        
        # 提取训练历史数据
        epochs = [entry['epoch'] for entry in training_history]
        train_losses = [entry['train_loss'] for entry in training_history]
        val_losses = [entry.get('val_loss', None) for entry in training_history]
        train_acc = [entry.get('train_acc', None) for entry in training_history]
        val_acc = [entry.get('val_acc', None) for entry in training_history]
        train_f1 = [entry.get('train_f1', None) for entry in training_history]
        val_f1 = [entry.get('val_f1', None) for entry in training_history]
        learning_rates = [entry.get('lr', None) for entry in training_history]
        
        # 创建有效数据的索引对，保持epoch对应关系
        val_data = [(epochs[i], val_losses[i]) for i in range(len(epochs)) if val_losses[i] is not None]
        train_acc_data = [(epochs[i], train_acc[i]) for i in range(len(epochs)) if train_acc[i] is not None]
        val_acc_data = [(epochs[i], val_acc[i]) for i in range(len(epochs)) if val_acc[i] is not None]
        train_f1_data = [(epochs[i], train_f1[i]) for i in range(len(epochs)) if train_f1[i] is not None]
        val_f1_data = [(epochs[i], val_f1[i]) for i in range(len(epochs)) if val_f1[i] is not None]
        lr_data = [(epochs[i], learning_rates[i]) for i in range(len(epochs)) if learning_rates[i] is not None]
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 8))
        
        # 子图1: 训练和验证损失对比
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
        if len(val_data) > 0:
            val_epochs, val_losses_filtered = zip(*val_data)
            plt.plot(val_epochs, val_losses_filtered, 'r-', label='Val Loss', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 学习率
        plt.subplot(2, 2, 2)
        if len(lr_data) > 0:
            lr_epochs, lr_values = zip(*lr_data)
            plt.plot(lr_epochs, lr_values, 'g-', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate Schedule')
        
        # 子图3: 准确率
        plt.subplot(2, 2, 3)
        if len(train_acc_data) > 0:
            train_acc_epochs, train_acc_values = zip(*train_acc_data)
            plt.plot(train_acc_epochs, train_acc_values, 'b-', label='Train Acc', alpha=0.7)
        if len(val_acc_data) > 0:
            val_acc_epochs, val_acc_values = zip(*val_acc_data)
            plt.plot(val_acc_epochs, val_acc_values, 'r-', label='Val Acc', alpha=0.7)
        
        if len(train_acc_data) > 0 or len(val_acc_data) > 0:
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training and Validation Accuracy')
        
        # 子图4: F1分数
        plt.subplot(2, 2, 4)
        if len(train_f1_data) > 0:
            train_f1_epochs, train_f1_values = zip(*train_f1_data)
            plt.plot(train_f1_epochs, train_f1_values, 'b-', label='Train F1', alpha=0.7)
        if len(val_f1_data) > 0:
            val_f1_epochs, val_f1_values = zip(*val_f1_data)
            plt.plot(val_f1_epochs, val_f1_values, 'r-', label='Val F1', alpha=0.7)
        
        if len(train_f1_data) > 0 or len(val_f1_data) > 0:
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Training and Validation F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No F1 data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training and Validation F1 Score')
        
        plt.tight_layout()
        training_curves_path = os.path.join(full_fig_dir, "training_curves.png")
        plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {training_curves_path}")
        
        # 打印训练统计信息
        if len(train_losses) > 0:
            print(f"Training Statistics:")
            print(f"  - Initial train loss: {train_losses[0]:.4f}")
            print(f"  - Final train loss: {train_losses[-1]:.4f}")
            print(f"  - Best train loss: {min(train_losses):.4f}")
            print(f"  - Total epochs: {len(epochs)}")
        
        if len(val_data) > 0:
            val_losses_filtered = [x[1] for x in val_data]
            print(f"  - Initial val loss: {val_losses_filtered[0]:.4f}")
            print(f"  - Final val loss: {val_losses_filtered[-1]:.4f}")
            print(f"  - Best val loss: {min(val_losses_filtered):.4f}")
        
        return full_fig_dir
        
    except Exception as e:
        print(f"Error creating training curves: {e}")
        return None

def get_predictions(model, data_loader, device):
    """
    Run inference on a data loader and return predicted labels and true labels for F1 score calculation.
    """
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # For consistency
            
            outputs = model(batch_x)
            # outputs shape: [batch_size, 1] for binary classification
            
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return all_preds, all_labels

def vali_cls(model, vali_data, vali_loader, criterion, args, device):
    """
    Validation loop for binary classification.
    Returns average loss and accuracy for the validation set.
    """
    total_loss = []
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        vali_bar = tqdm(enumerate(vali_loader), total=len(vali_loader), 
                       desc='Validation', leave=False, ncols=100)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in vali_bar:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # BCEWithLogitsLoss requires FloatTensor labels
            
            outputs = model(batch_x)  # [batch_size, 1]
            
            loss = criterion(outputs.squeeze(), batch_y)
            total_loss.append(loss.item())
            
            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # Update progress bar with current loss
            if len(total_loss) > 0:
                vali_bar.set_postfix({'Loss': f'{np.mean(total_loss):.4f}'})
        
        vali_bar.close()
    
    total_loss = np.average(total_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    
    model.train()
    return total_loss, accuracy

def test_cls(model, test_data, test_loader, args, device):
    """
    Test loop for binary classification.
    Returns accuracy, precision, recall, F1, and AUC for the test set.
    """
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), 
                       desc='Testing', leave=False, ncols=100)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_bar:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # For consistency, though not used in loss calculation here
            
            outputs = model(batch_x)  # [batch_size, 1]
            
            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # Probability of seizure class
            
        test_bar.close()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    print(f'Test Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    
    return accuracy, precision, recall, f1, auc

###########################################################
# Main training loop: runs for args.itr iterations
# Each iteration trains a new model, validates, tests, and saves results
###########################################################
accuracies = []
precisions = []
recalls = []
f1_scores = []
aucs = []

print(f"Starting EEG Seizure Detection Training")
print(f"Model: {args.model_id}")
print(f"Total iterations: {args.itr}")
print(f"Data Augmentation: {args.aug_method}")
if args.aug_decomp:
    print(f"  - Decomposition: enabled (p={args.aug_p}, win={args.aug_win}, only_bg={args.aug_only_bg})")
if args.aug_ww:
    print(f"  - Window Warping: enabled (p=[{args.aug_ww_p_low},{args.aug_ww_p_high}], ratio=[{args.aug_ww_win_ratio_low},{args.aug_ww_win_ratio_high}], only_bg={args.aug_ww_only_bg})")
if args.aug_method == 'none':
    print(f"  - No augmentation")
print("=" * 60)

for ii in range(args.itr):
    print(f"\nIteration {ii+1}/{args.itr}")
    # Format string for experiment setting (for logging/checkpoints)
    setting = '{}_sl{}_dm{}_nh{}_el{}_df{}_itr{}'.format(
        args.model_id, args.seq_len, args.d_model, args.n_heads, 
        args.e_layers, args.d_ff, ii)
    
    # Create unique checkpoint path with timestamp to avoid conflicts
    import time
    timestamp = int(time.time())
    path = './checkpoints/' + args.model_id + '_' + str(ii) + '_' + str(timestamp)
    if not os.path.exists(path):
        os.makedirs(path)

    # Always train fresh (no skip mechanism)
    # Each run will have unique timestamp
    
    # Load training data and fit scaler
    train_data, train_loader = data_provider(args, 'train')
    
    # Pass the training scaler to validation and test datasets
    vali_data, vali_loader = data_provider(args, 'val', train_scaler=train_data.scaler)
    test_data, test_loader = data_provider(args, 'test', train_scaler=train_data.scaler)
    
    device = torch.device(device_address)
    time_now = time.time()
    train_steps = len(train_loader)

    # Initialize model
    if args.model == 'PAttn':
        model = PAttn(args, device)
        print(f"Model loaded on device: {device}")
        model.to(device)

    # Set up optimizer and early stopping
    model_optim = select_optimizer(model, args)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Set up binary classification loss with optional pos_weight
    pos_weight_value = None
    if args.auto_pos_weight:
        pos_weight_value = calculate_pos_weight(train_loader, device)
        print(f"Auto-calculated pos_weight: {pos_weight_value:.4f}")
    elif args.pos_weight is not None:
        pos_weight_value = args.pos_weight
        print(f"Manual pos_weight: {pos_weight_value}")
    
    if pos_weight_value is not None:
        pos_weight = torch.tensor(pos_weight_value).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss without pos_weight")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    
    # Initialize training history for this iteration
    training_history = []
    
    is_first = True
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        train_preds = []
        train_labels = []
        epoch_time = time.time()
        
        # Create epoch progress bar
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f'Epoch {epoch+1}/{args.train_epochs}', 
                        leave=True, ncols=120)
        
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in train_bar:
            if is_first:
                # Print input and label shapes for the first batch
                print(f"Data shapes - Input: {batch_x.shape}, Labels: {batch_y.shape}")
                is_first = False
            
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # BCEWithLogitsLoss requires FloatTensor labels
            
            outputs = model(batch_x)  # [batch_size, 1]
            
            loss = criterion(outputs.squeeze(), batch_y)
            train_loss.append(loss.item())
            
            # Calculate predictions for accuracy
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

            # Update progress bar with current metrics
            current_loss = np.mean(train_loss[-10:])  # Average of last 10 batches
            train_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'LR': f'{model_optim.param_groups[0]["lr"]:.2e}'
            })
            
            loss.backward()
            model_optim.step()

        train_bar.close()
        print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f}s")

        train_loss = np.average(train_loss)
        
        # Calculate training accuracy and F1 score
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        # Validation loss and accuracy
        vali_loss, vali_acc = vali_cls(model, vali_data, vali_loader, criterion, args, device)
        
        # Validation F1 score
        vali_preds, vali_labels_true = get_predictions(model, vali_loader, device)
        vali_f1 = f1_score(vali_labels_true, vali_preds)

        print(f"Epoch {epoch + 1}/{args.train_epochs} | Train Loss: {train_loss:.6f} | Vali Loss: {vali_loss:.6f} | Train Acc: {train_acc:.4f} | Vali Acc: {vali_acc:.4f} | Train F1: {train_f1:.4f} | Vali F1: {vali_f1:.4f}")

        # Record training history for plotting
        current_lr = model_optim.param_groups[0]['lr']
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': vali_loss,
            'train_acc': train_acc,
            'val_acc': vali_acc,
            'train_f1': train_f1,
            'val_f1': vali_f1,
            'lr': current_lr
        }
        training_history.append(epoch_record)

        # Update learning rate
        if args.cos:
            scheduler.step()
            print(f"Learning Rate: {model_optim.param_groups[0]['lr']:.2e}")
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        
        # Early stopping based on validation loss
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Load best model and test
    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    
    # Test and evaluate metrics
    accuracy, precision, recall, f1, auc = test_cls(model, test_data, test_loader, args, device)
    
    # Generate and save training curves for this iteration
    print(f"\nGenerating training curves for iteration {ii+1}...")
    fig_dir = plot_training_curves(path, training_history, fig_dir="figs")
    if fig_dir:
        print(f"Visualization saved for iteration {ii+1}")
    
    # Store metrics for summary
    accuracies.append(round(accuracy, 4))
    precisions.append(round(precision, 4))
    recalls.append(round(recall, 4))
    f1_scores.append(round(f1, 4))
    aucs.append(round(auc, 4))
    
    print(f"Iteration {ii+1} completed successfully")

# Final results summary and saving
if len(accuracies) == 0:
    print("No results to display")
    exit()

accuracies = np.array(accuracies)
precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)
aucs = np.array(aucs)

print("\n" + "="*70)
print("FINAL EEG SEIZURE DETECTION RESULTS")
print("="*70)
print(f"Accuracy  - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
print(f"Precision - Mean: {np.mean(precisions):.4f}, Std: {np.std(precisions):.4f}")
print(f"Recall    - Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")
print(f"F1-Score  - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")
print(f"AUC       - Mean: {np.mean(aucs):.4f}, Std: {np.std(aucs):.4f}")
print("="*70)

# Save results to file
with open(args.save_file_name, 'a') as f:
    f.write(f"\n{'='*50}\n")
    f.write(f"Model: {args.model_id}\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*50}\n")
    f.write(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
    f.write(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
    f.write(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
    f.write(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
    f.write(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")
    f.write(f"{'='*50}\n")

print(f"Results saved to: {args.save_file_name}")
