import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data.data_utils import *
from data.dataloader_detection import load_dataset_detection
from data.dataloader_classification import load_dataset_classification
from data.dataloader_densecnn_classification import load_dataset_densecnn_classification
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
from model.model import DCRNNModel_classification, DCRNNModel_nextTimePred
from model.densecnn import DenseCNN
from model.lstm import LSTMModel
from model.cnnlstm import CNN_LSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from datetime import datetime


def plot_training_curves(save_dir, training_history, fig_dir="figs"):
    """绘制训练过程的损失曲线和指标曲线，保持与train_rest.py相同的风格"""
    try:
        # 在函数内部导入，避免顶层导入错误
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
            print(f"Validation Statistics:")
            print(f"  - Initial val loss: {val_losses_filtered[0]:.4f}")
            print(f"  - Final val loss: {val_losses_filtered[-1]:.4f}")
            print(f"  - Best val loss: {min(val_losses_filtered):.4f}")
        
        # 打印准确率统计信息
        if len(train_acc_data) > 0:
            train_acc_values = [x[1] for x in train_acc_data]
            print(f"Training Accuracy Statistics:")
            print(f"  - Initial train acc: {train_acc_values[0]:.4f}")
            print(f"  - Final train acc: {train_acc_values[-1]:.4f}")
            print(f"  - Best train acc: {max(train_acc_values):.4f}")
        
        if len(val_acc_data) > 0:
            val_acc_values = [x[1] for x in val_acc_data]
            print(f"Validation Accuracy Statistics:")
            print(f"  - Initial val acc: {val_acc_values[0]:.4f}")
            print(f"  - Final val acc: {val_acc_values[-1]:.4f}")
            print(f"  - Best val acc: {max(val_acc_values):.4f}")
        
        # 打印F1分数统计信息
        if len(train_f1_data) > 0:
            train_f1_values = [x[1] for x in train_f1_data]
            print(f"Training F1 Statistics:")
            print(f"  - Initial train F1: {train_f1_values[0]:.4f}")
            print(f"  - Final train F1: {train_f1_values[-1]:.4f}")
            print(f"  - Best train F1: {max(train_f1_values):.4f}")
        
        if len(val_f1_data) > 0:
            val_f1_values = [x[1] for x in val_f1_data]
            print(f"Validation F1 Statistics:")
            print(f"  - Initial val F1: {val_f1_values[0]:.4f}")
            print(f"  - Final val F1: {val_f1_values[-1]:.4f}")
            print(f"  - Best val F1: {max(val_f1_values):.4f}")
        
        return full_fig_dir
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")
        # 生成一个简单的占位图表
        try:
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            full_fig_dir = os.path.join(fig_dir, now_str)
            os.makedirs(full_fig_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Training curves not available\n(Error occurred during plotting)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title('Training Curves - Error Occurred')
            plt.axis('off')
            training_curves_path = os.path.join(full_fig_dir, "training_curves.png")
            plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Fallback training curves saved to: {training_curves_path}")
            return full_fig_dir
        except:
            print("Failed to create fallback training curves plot")
            return None


def plot_evaluation_results(y_true, y_pred, y_prob, fig_dir):
    """绘制评估结果的可视化图表，保持与train_rest.py相同的风格"""
    try:
        # 在函数内部导入，避免顶层导入错误
        from sklearn import metrics
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 计算指标
        auc = metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        ap = metrics.average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        f1 = metrics.f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        rec = metrics.recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        pre = metrics.precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)
        acc = metrics.accuracy_score(y_true, y_pred)
        
        print(f"Evaluation Metrics: AUC={auc:.4f} | AP={ap:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Precision={pre:.4f} | Acc={acc:.4f}")
        
        # ROC 曲线
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
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
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pr_path = os.path.join(fig_dir, "pr.png")
            plt.savefig(pr_path, dpi=150)
            plt.close()
            
            print(f"ROC and PR curves saved to: {roc_path}, {pr_path}")
        
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
        if len(np.unique(y_true)) > 1:
            plt.figure()
            plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Negative", density=True)
            plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Positive", density=True)
            plt.xlabel("Predicted Probability")
            plt.ylabel("Density")
            plt.title("Score Distribution by Class")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(fig_dir, "score_hist.png")
            plt.savefig(hist_path, dpi=150)
            plt.close()
            
            print(f"Confusion matrix and score histogram saved to: {cm_path}, {hist_path}")
        else:
            print(f"Confusion matrix saved to: {cm_path}")
        
    except Exception as e:
        print(f"Error plotting evaluation results: {e}")


def evaluate_for_visualization(model, dataloader, args, device):
    """专门用于可视化的评估函数，返回真实标签、预测标签和预测概率"""
    model.eval()
    
    # 根据任务类型选择合适的损失函数
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    
    with torch.no_grad():
        for x, y, seq_lengths, supports, _, _ in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device)  # (batch_size,)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            if args.model_name == "dcrnn":
                logits = model(x, seq_lengths, supports)
            elif args.model_name == "densecnn":
                x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes)
                logits = model(x)
            elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                logits = model(x, seq_lengths)
            else:
                raise NotImplementedError

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > 0.5).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                # 对于多分类，取最大概率对应的类别的概率
                y_prob = np.max(y_prob, axis=1)  # 或者可以取特定类别的概率
                y_pred = np.argmax(F.softmax(logits, dim=1).cpu().numpy(), axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    
    return y_true_all, y_pred_all, y_prob_all


def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    # 初始化训练历史记录
    training_history = []

    # Build dataset
    log.info('Building dataset...')
    if args.task == 'detection':
        dataloaders, _, scaler = load_dataset_detection(
            input_dir=args.input_dir,
            raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            max_seq_len=args.max_seq_len,
            standardize=True,
            num_workers=args.num_workers,
            augmentation=args.data_augment,
            adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            use_fft=args.use_fft,
            sampling_ratio=1,
            seed=123,
            preproc_dir=args.preproc_dir)
    elif args.task == 'classification':
        if args.model_name != 'densecnn':
            dataloaders, _, scaler = load_dataset_classification(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                padding_val=0.,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                preproc_dir=args.preproc_dir)
        else:
            print("Using densecnn dataloader!")
            dataloaders, _, scaler = load_dataset_densecnn_classification(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                padding_val=0.,
                augmentation=args.data_augment,
                use_fft=args.use_fft,
                preproc_dir=args.preproc_dir
            )
    else:
        raise NotImplementedError

    # Build model
    log.info('Building model...')
    if args.model_name == "dcrnn":
        model = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "densecnn":
        with open("./model/dense_inception/params.json", "r") as f:
            params = json.load(f)
        params = DottedDict(params)
        data_shape = (args.max_seq_len*100, args.num_nodes) if args.use_fft else (args.max_seq_len*200, args.num_nodes)
        model = DenseCNN(params, data_shape=data_shape, num_classes=args.num_classes)
    elif args.model_name == "lstm":
        model = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        model = CNN_LSTM(args.num_classes)
    else:
        raise NotImplementedError

    if args.do_train:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)
        else:  # fine-tune from pretrained model
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                pretrained_model = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  # placeholder
                pretrained_model = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model)

                model = utils.build_finetune_model(
                    model_new=model,
                    model_pretrained=pretrained_model,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')

        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        model = model.to(device)

        # Train
        training_history = train(model, dataloaders, args, device, args.save_dir, log, tbx)

        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))

    # 生成可视化图表
    log.info('Generating visualization plots...')
    try:
        # 绘制训练曲线
        fig_dir = plot_training_curves(args.save_dir, training_history)
        
        if fig_dir is not None:
            # 获取测试集的详细预测结果用于可视化
            log.info('Generating detailed evaluation plots...')
            y_true, y_pred, y_prob = evaluate_for_visualization(model, dataloaders['test'], args, device)
            
            # 绘制评估结果
            plot_evaluation_results(y_true, y_pred, y_prob, fig_dir)
            
            log.info(f'All visualization plots saved to: {fig_dir}')
        else:
            log.warning('Failed to create visualization plots directory')
            
    except Exception as e:
        log.error(f'Error generating visualization plots: {e}')
        print(f"Visualization error: {e}")


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    Returns training history for visualization
    """

    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # 初始化训练历史记录
    training_history = []

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    epoch_train_losses = []  # 记录每个epoch内的训练损失
    
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        epoch_train_losses = []  # 重置每个epoch的损失
        
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for x, y, seq_lengths, supports, _, _ in train_loader:
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
                if args.model_name == "dcrnn":
                    logits = model(x, seq_lengths, supports)
                elif args.model_name == "densecnn":
                    x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes) # (batch_size, seq_len, num_nodes)
                    logits = model(x)
                elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                    logits = model(x, seq_lengths)
                else:
                    raise NotImplementedError
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,)                
                loss = loss_fn(logits, y)
                loss_val = loss.item()
                epoch_train_losses.append(loss_val)

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

            # 记录每个epoch的平均训练损失和学习率
            epoch_avg_train_loss = np.mean(epoch_train_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 初始化当前epoch的记录
            epoch_record = {
                'epoch': epoch,
                'train_loss': epoch_avg_train_loss,
                'lr': current_lr
            }

            # 在每个epoch结束时评估训练集性能
            log.info('Evaluating training set at epoch {}...'.format(epoch))
            model.eval()  # 切换到评估模式
            train_results = evaluate(model,
                                   train_loader,
                                   args,
                                   save_dir,
                                   device,
                                   is_test=False,
                                   nll_meter=None,
                                   eval_set='train')
            
            # 添加训练结果到epoch记录
            epoch_record['train_acc'] = train_results.get('acc', None)
            epoch_record['train_f1'] = train_results.get('F1', None)
            
            # 切换回训练模式
            model.train()

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating validation set at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # 添加验证结果到epoch记录
                epoch_record['val_loss'] = eval_results['loss']
                epoch_record['val_acc'] = eval_results.get('acc', None)
                epoch_record['val_f1'] = eval_results.get('F1', None)

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)
            
            # 将epoch记录添加到训练历史
            training_history.append(epoch_record)

        # Step lr scheduler
        scheduler.step()
    
    return training_history


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, seq_lengths, supports, _, file_name in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device)  # (batch_size,)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, num_classes)
            if args.model_name == "dcrnn":
                logits = model(x, seq_lengths, supports)
            elif args.model_name == "densecnn":
                x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes) # (batch_size, len*freq, num_nodes)
                logits = model(x)
            elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                logits = model(x, seq_lengths)
            else:
                raise NotImplementedError

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    # Threshold search, for detection only
    if (args.task == "detection") and (eval_set == 'dev') and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_args())
