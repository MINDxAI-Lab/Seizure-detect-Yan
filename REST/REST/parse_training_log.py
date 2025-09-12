#!/usr/bin/env python3
"""
从SLURM输出文件中解析训练日志并生成训练过程可视化
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict

def parse_training_log(log_file):
    """
    解析训练日志文件，提取训练和验证损失
    """
    train_losses = []
    val_losses = []
    epochs = []
    steps = []
    
    # 正则表达式模式
    # 匹配类似: Epoch 4:  34%|███▍      | 50/148 [01:30<02:57,  1.81s/it, v_num=9757137, train_loss=0.154]
    epoch_pattern = r'Epoch\s+(\d+):\s+\d+%.*?train_loss=([\d.]+)'
    
    # 匹配验证损失 (如果有的话)
    val_pattern = r'val_loss=([\d.]+)'
    
    # 匹配验证指标输出
    metrics_pattern = r'Validation Metrics: AUC=([\d.]+).*?AP=([\d.]+).*?F1=([\d.]+).*?Recall=([\d.]+).*?Precision=([\d.]+).*?Acc=([\d.]+)'
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        print(f"Parsing log file: {log_file}")
        print(f"File size: {len(content)} characters")
        
        # 解析训练损失
        train_matches = re.findall(epoch_pattern, content)
        print(f"Found {len(train_matches)} training loss entries")
        
        for epoch_str, loss_str in train_matches:
            try:
                epoch = int(epoch_str)
                loss = float(loss_str)
                epochs.append(epoch)
                train_losses.append(loss)
            except ValueError:
                continue
        
        # 解析验证损失 (通常在同一行)
        val_matches = re.findall(val_pattern, content)
        print(f"Found {len(val_matches)} validation loss entries")
        
        # 查找最终验证指标
        metrics_matches = re.findall(metrics_pattern, content)
        final_metrics = None
        if metrics_matches:
            final_metrics = {
                'AUC': float(metrics_matches[-1][0]),
                'AP': float(metrics_matches[-1][1]),
                'F1': float(metrics_matches[-1][2]),
                'Recall': float(metrics_matches[-1][3]),
                'Precision': float(metrics_matches[-1][4]),
                'Accuracy': float(metrics_matches[-1][5])
            }
            print(f"Final validation metrics found: {final_metrics}")
        
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_matches,  # 这里可能需要进一步处理
            'final_metrics': final_metrics
        }
        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

def plot_training_curves_from_log(data, output_dir="figs"):
    """
    从解析的日志数据生成训练曲线图
    """
    if not data or not data['train_losses']:
        print("No training data found to plot")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = np.array(data['epochs'])
    train_losses = np.array(data['train_losses'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 训练损失随epoch变化
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss vs Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 训练损失统计信息
    axes[0, 1].text(0.1, 0.9, f'Initial Loss: {train_losses[0]:.4f}', transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].text(0.1, 0.8, f'Final Loss: {train_losses[-1]:.4f}', transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].text(0.1, 0.7, f'Best Loss: {train_losses.min():.4f}', transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].text(0.1, 0.6, f'Total Epochs: {epochs.max()}', transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].text(0.1, 0.5, f'Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%', 
                    transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].set_title('Training Statistics')
    axes[0, 1].axis('off')
    
    # 子图3: 训练损失的移动平均
    if len(train_losses) > 5:
        window = min(max(len(train_losses) // 10, 3), 20)
        smoothed_losses = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        smoothed_epochs = epochs[window-1:]
        
        axes[1, 0].plot(epochs, train_losses, 'b-', alpha=0.3, label='Raw Loss')
        axes[1, 0].plot(smoothed_epochs, smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Smoothed Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor smoothing', ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Smoothed Training Loss')
    
    # 子图4: 最终验证指标 (如果有的话)
    if data['final_metrics']:
        metrics = data['final_metrics']
        y_pos = np.arange(len(metrics))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 1].barh(y_pos, metric_values, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(metric_names)
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_title('Final Validation Metrics')
        axes[1, 1].set_xlim(0, 1)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            axes[1, 1].text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{value:.3f}', va='center', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'No validation metrics\nfound in log', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "training_curves_from_log.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")
    
    # 验证文件保存
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")
    else:
        print("Warning: File was not saved!")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Parse training log and generate visualization')
    parser.add_argument('log_file', help='Path to the SLURM output file (.out)')
    parser.add_argument('--output-dir', default='figs', help='Output directory for plots (default: figs)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.log_file):
        print(f"Error: Log file {args.log_file} not found!")
        return
    
    # 解析日志
    print("Parsing training log...")
    data = parse_training_log(args.log_file)
    
    if data:
        # 生成可视化
        print("Generating training curves...")
        plot_training_curves_from_log(data, args.output_dir)
        
        # 打印摘要
        if data['train_losses']:
            print(f"\nTraining Summary:")
            print(f"  - Epochs processed: {len(set(data['epochs']))}")
            print(f"  - Training steps: {len(data['train_losses'])}")
            print(f"  - Initial loss: {data['train_losses'][0]:.4f}")
            print(f"  - Final loss: {data['train_losses'][-1]:.4f}")
            print(f"  - Best loss: {min(data['train_losses']):.4f}")
    else:
        print("Failed to parse training log or no data found.")

if __name__ == "__main__":
    main()
