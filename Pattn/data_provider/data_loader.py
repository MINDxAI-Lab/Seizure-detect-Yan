import os
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from typing import Tuple
import matplotlib.pyplot as plt
import random
import h5py
import glob 

warnings.filterwarnings('ignore')

def decompose( 
    x: torch.Tensor, period: int = 7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose input time series into trend, seasonality and residual components using STL.

    Args:
        x (torch.Tensor): Input time series. Shape: (1, seq_len).
        period (int, optional): Period of seasonality. Defaults to 7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decomposed components. Shape: (1, seq_len).
    """
    # print('in' , x.shape)
    # x = x.squeeze(0).cpu().numpy()
    if len(x.shape) ==2 : 
        x = x.squeeze(0)
    decomposed = STL(x, period=period).fit()
    trend = decomposed.trend.astype(np.float32)
    seasonal = decomposed.seasonal.astype(np.float32)
    residual = decomposed.resid.astype(np.float32)
    return (
        torch.from_numpy(trend).unsqueeze(0),
        torch.from_numpy(seasonal).unsqueeze(0),
        torch.from_numpy(residual).unsqueeze(0),
    )

# --- Decomposition-based augmentation (time-domain) ---

def _moving_avg_trend(x: torch.Tensor, win: int):
    """
    x: (C, L)  -> trend, resid: (C, L)
    Compute simple moving average trend for each channel using grouped convolution.
    Boundaries are padded with replicated values to avoid phase shift.
    """
    C, L = x.shape
    pad = (win - 1) // 2
    weight = torch.ones(C, 1, win, device=x.device) / win  # grouped convolution kernel (one per channel)
    x1 = F.pad(x.unsqueeze(0), (pad, pad), mode="replicate")  # (1,C,L+2p) - pad both ends
    trend = F.conv1d(x1, weight, groups=C).squeeze(0)        # (C,L) - grouped convolution
    resid = x - trend
    return trend, resid

def _decomp_augment(x: torch.Tensor, win=129, scale_low=0.9, scale_high=1.1, noise_ratio=0.02):
    """
    Decomposition-based augmentation: apply channel-independent scaling to residual 
    components plus small noise. Formula: x' = trend + s * resid + eps
    """
    trend, resid = _moving_avg_trend(x, win)
    # Random scaling factor per channel
    s = torch.empty((x.size(0), 1), device=x.device).uniform_(scale_low, scale_high)
    resid = resid * s
    # Small noise proportional to residual standard deviation to maintain SNR
    eps = torch.randn_like(resid) * (resid.std(dim=1, keepdim=True) + 1e-8) * noise_ratio
    return trend + resid + eps

# --- Window Warping augmentation ---

def _window_warping_step1_and_step2(seq_len: int, sampling_rate: int = 256, 
                                   p_low: float = 0.3, p_high: float = 0.7,
                                   win_ratio_low: float = 0.1, win_ratio_high: float = 0.3,
                                   margin_sec: float = 0.5, 
                                   speed_ratios: list = None):
    """
    Step 1: Trigger and sampling + Step 2: Speed ratio selection
    
    This function implements the first two steps of Window Warping augmentation:
    1. Randomly decide whether to trigger WW and sample a sub-window
    2. Select a time scaling ratio for the warping operation
    
    Args:
        seq_len: Total sequence length (number of samples)
        sampling_rate: Sampling rate in Hz 
        p_low, p_high: Range for trigger probability sampling
        win_ratio_low, win_ratio_high: Range for sub-window size ratio
        margin_sec: Margin from boundaries in seconds to avoid edge effects
        speed_ratios: Optional list of candidate time scaling ratios, default [0.67, 0.8, 1.25, 1.5]
    
    Returns:
        tuple: (triggered: bool, sub_window_start: int, sub_window_end: int, speed_ratio: float)
    """
    # Step 1: Trigger decision and window sampling  
    # Randomly decide whether to trigger Window Warping
    trigger_prob = random.uniform(p_low, p_high)
    if random.random() > trigger_prob:
        return False, 0, 0, 1.0
    
    # Calculate margin in samples to avoid boundary effects
    margin_samples = int(margin_sec * sampling_rate)
    
    # Ensure sufficient space for window selection
    if seq_len <= 2 * margin_samples:
        return False, 0, 0, 1.0
    
    # Available region length after removing margins
    available_len = seq_len - 2 * margin_samples
    
    # Randomly select sub-window size ratio
    win_ratio = random.uniform(win_ratio_low, win_ratio_high)
    
    # Calculate sub-window length in samples
    sub_win_len = int(available_len * win_ratio)
    
    # Ensure minimum sub-window length of 1
    if sub_win_len < 1:
        return False, 0, 0, 1.0
    
    # Randomly select sub-window start position within available region
    max_start = available_len - sub_win_len
    if max_start <= 0:
        # Sub-window covers entire available region
        sub_win_start = margin_samples
        sub_win_end = seq_len - margin_samples
    else:
        relative_start = random.randint(0, max_start)
        sub_win_start = margin_samples + relative_start
        sub_win_end = sub_win_start + sub_win_len
    
    # Step 2: Speed ratio selection
    if speed_ratios is None:
        speed_ratios = [0.67, 0.8, 1.25, 1.5]  # default speed ratio options
    
    # Randomly select a time scaling ratio
    speed_ratio = random.choice(speed_ratios)
    
    return True, sub_win_start, sub_win_end, speed_ratio

def _window_warping_step3(x: torch.Tensor, start_idx: int, end_idx: int, speed_ratio: float):
    """
    Step 3: Time mapping (length-preserving warping)
    
    Apply time scaling to a sub-window segment while maintaining overall sequence length.
    The process involves: 1) extract sub-window, 2) resample by speed ratio, 
    3) interpolate back to original sub-window length, 4) replace in original signal.
    
    Args:
        x: Input signal of shape (C, L) where C=channels, L=sequence length
        start_idx: Sub-window start index (inclusive)
        end_idx: Sub-window end index (exclusive) 
        speed_ratio: Time scaling ratio (>1 for speed-up, <1 for slow-down)
    
    Returns:
        torch.Tensor: Transformed signal with same shape (C, L)
    """
    # Validate input parameters
    C, L = x.shape
    if start_idx < 0 or end_idx > L or start_idx >= end_idx:
        return x  # Invalid parameters, return original signal
    
    # Extract sub-window for processing
    sub_window = x[:, start_idx:end_idx]  # (C, sub_win_len)
    original_sub_len = end_idx - start_idx
    
    # Step 3.1: Temporary resampling by speed ratio
    # New length L' = floor(speed_ratio * original_sub_len)
    temp_len = int(speed_ratio * original_sub_len)
    
    # Ensure minimum temporary length of 1
    if temp_len < 1:
        temp_len = 1
    
    # Use PyTorch's interpolate for resampling
    # Input: (C, sub_win_len) -> (1, C, sub_win_len) for interpolate compatibility
    # Output: (1, C, temp_len) -> (C, temp_len)
    sub_window_unsqueezed = sub_window.unsqueeze(0)  # (1, C, sub_win_len)
    
    if speed_ratio < 1.0:
        # Slow-down/downsampling: use anti-aliasing
        # Linear mode provides reasonable anti-aliasing effect
        temp_resampled = F.interpolate(
            sub_window_unsqueezed, 
            size=temp_len, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)  # (C, temp_len)
    else:
        # Speed-up/upsampling: use linear interpolation
        temp_resampled = F.interpolate(
            sub_window_unsqueezed, 
            size=temp_len, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)  # (C, temp_len)
    
    # Step 3.2: Interpolate back to original length
    # Interpolate (C, temp_len) back to (C, original_sub_len)
    temp_resampled_unsqueezed = temp_resampled.unsqueeze(0)  # (1, C, temp_len)
    final_resampled = F.interpolate(
        temp_resampled_unsqueezed,
        size=original_sub_len,
        mode='linear',
        align_corners=False
    ).squeeze(0)  # (C, original_sub_len)
    
    # Step 3.3: Replace sub-window in original signal
    result = x.clone()  # Copy original signal
    result[:, start_idx:end_idx] = final_resampled  # Replace sub-window with warped version
    
    return result
    
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='' ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()

        self.model_id = model_id
        
        self.period = 24 
        self.channel= 7
        self.enc_in =1 
        
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def draw_decompose(self, x , trend, seasonal, residual):
        plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
        # Plot each array
        x = x.reshape(-1,)
        trend = trend.reshape(-1,)
        seasonal = seasonal.reshape(-1,)
        residual = residual.reshape(-1,)
        print(x.shape , trend.shape)
        
        plt.plot(x, label='x A')
        plt.plot(trend, label='trend B')
        plt.plot(seasonal, label='seasonal C')
        plt.plot(residual, label='residual D')
        ii = random.randint(0,100)
        # Adding labels
        plt.xlabel('Index')  # Assuming the index represents the x-axis
        plt.ylabel('Value')  # The y-axis label
        plt.title('Plot of Four Arrays')  # Title of the plot
        plt.legend()
        plt.savefig(f'/p/selfdrivingpj/projects_time/NeurIPS2023-One-Fits-All/Long-term_Forecasting/figures/{ii}.jpg')
        plt.cla()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [int(12 * 30 * 24 * self.train_ratio), 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print(self.set_type ,self.data_x.shape)
                
    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [c, seq_len]
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [c, pred_len]
        return x , y ,  seq_x_mark, seq_y_mark
        
        
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, train_all=False  , model_id = ''):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.model_id = model_id 
        self.period = 60 
        self.channel= 7
        self.enc_in =1 
     
        self.tot_len = (len(self.data_x) - self.seq_len - self.pred_len + 1)
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [c, seq_len]
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [c, pred_len]
        return x , y ,  seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False , train_ratio=1.0 , model_id=''):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.model_id= model_id
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        if 'weather' in data_path:
            # per 10min
            self.period = 36
            self.channel= 21
        if 'traffic' in data_path:
            # per hour 
            self.period = 24
            self.channel= 862
        if 'electricity' in data_path:
            # per hour 
            self.period = 24
            self.channel= 321
        if 'illness' in data_path:
            # 1week
            self.period = 12
            self.channel= 7

        self.enc_in = 1 
            
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(self.data_x.shape)
        
    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [c, seq_len]
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [c, pred_len]
        return x , y ,  seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_EEG_Seizure(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='seizure_data',
                 target='label', scale=True, timeenc=0, freq='h',
                 percent=100, max_len=-1, train_all=False, train_ratio=1.0, model_id='', 
                 shared_scaler=None,
                 # ---- Decomposition-based augmentation hyperparameters ----
                 aug_decomp=False, aug_p=0.5, aug_win=129,
                 aug_scale_low=0.9, aug_scale_high=1.1,
                 aug_noise=0.02, aug_only_bg=False,
                 # ---- Window Warping augmentation hyperparameters ----
                 aug_ww=False, aug_ww_p_low=0.3, aug_ww_p_high=0.7,
                 aug_ww_win_ratio_low=0.1, aug_ww_win_ratio_high=0.3,
                 aug_ww_speed_low=0.8, aug_ww_speed_high=1.2,
                 aug_ww_margin=0.5, aug_ww_only_bg=False):
        """
        EEG Seizure Detection Dataset
        
        Args:
            root_path: Path to the root directory containing clip_data and label_data folders
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len] - for classification we only need seq_len
            features: 'M' for multivariate (all channels), 'S' for single channel
            data_path: Name of the dataset folder
            target: Target variable name (not used in this dataset)
            scale: Whether to apply standardization
            timeenc: Time encoding method (not used)
            freq: Frequency parameter (not used)
            percent: Percentage of data to use
            max_len: Maximum length (not used)
            train_all: Whether to use all training data
            train_ratio: Ratio of training data to use
            model_id: Model identifier
            shared_scaler: Pre-fitted scaler from training dataset (for val/test)
            
            # Decomposition-based augmentation parameters
            aug_decomp: Whether to enable decomposition-based augmentation
            aug_p: Probability of applying augmentation
            aug_win: Window size for moving average trend decomposition
            aug_scale_low: Lower bound for residual scaling
            aug_scale_high: Upper bound for residual scaling
            aug_noise: Noise ratio relative to residual standard deviation
            aug_only_bg: Whether to apply augmentation only to background (non-seizure) samples
            
            # Window Warping augmentation parameters
            aug_ww: Whether to enable Window Warping augmentation
            aug_ww_p_low: Lower bound for WW trigger probability
            aug_ww_p_high: Upper bound for WW trigger probability
            aug_ww_win_ratio_low: Lower bound for sub-window ratio
            aug_ww_win_ratio_high: Upper bound for sub-window ratio
            aug_ww_speed_low: Lower bound for time warping speed ratio
            aug_ww_speed_high: Upper bound for time warping speed ratio
            aug_ww_margin: Margin from boundaries in seconds
            aug_ww_only_bg: Whether to apply WW augmentation only to background (non-seizure) samples
        """
        # Set sequence length
        if size is None:
            self.seq_len = 3072  # 12 seconds * 256 Hz = 3072 samples
        else:
            self.seq_len = size[0]
        
        # Initialize parameters
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        # Map flag to actual directory name
        flag_to_dir = {'train': 'train', 'val': 'eval', 'test': 'eval'}
        self.flag = flag_to_dir[flag]
        self.original_flag = flag  # Keep original flag for scaler logic
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.train_ratio = train_ratio
        self.model_id = model_id
        
        # Initialize augmentation parameters
        self.aug_decomp = aug_decomp
        self.aug_p = aug_p
        self.aug_win = aug_win
        self.aug_scale_low = aug_scale_low
        self.aug_scale_high = aug_scale_high
        self.aug_noise = aug_noise
        self.aug_only_bg = aug_only_bg
        
        # Initialize Window Warping augmentation parameters
        self.aug_ww = aug_ww
        self.aug_ww_p_low = aug_ww_p_low
        self.aug_ww_p_high = aug_ww_p_high
        self.aug_ww_win_ratio_low = aug_ww_win_ratio_low
        self.aug_ww_win_ratio_high = aug_ww_win_ratio_high
        self.aug_ww_speed_low = aug_ww_speed_low
        self.aug_ww_speed_high = aug_ww_speed_high
        self.aug_ww_margin = aug_ww_margin
        self.aug_ww_only_bg = aug_ww_only_bg
        
        self.root_path = root_path
        self.data_path = data_path
        
        # EEG specific parameters
        self.period = 256  # Sampling frequency
        self.channel = 19  # Number of EEG channels
        self.enc_in = 19 if features == 'M' else 1  # Input channels
        
        # Handle scaler
        if scale:
            if shared_scaler is not None:
                # Use pre-fitted scaler from training dataset
                self.scaler = shared_scaler
            else:
                # Create new scaler (only for training dataset)
                self.scaler = StandardScaler()
        
        # Load data
        self.__read_data__()
        
        print(f"Dataset {flag}: {len(self.data_files)} samples, {self.enc_in} channels")
        
    def __read_data__(self):
        """Load EEG data from H5 files"""
        # Construct paths
        clip_dir = os.path.join(self.root_path, 'clip_data', self.flag)
        label_dir = os.path.join(self.root_path, 'label_data', self.flag)
        
        if not os.path.exists(clip_dir) or not os.path.exists(label_dir):
            raise ValueError(f"Data directories not found: {clip_dir} or {label_dir}")
        
        # Get all H5 files
        clip_files = sorted(glob.glob(os.path.join(clip_dir, '*.h5')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.h5')))
        
        if len(clip_files) != len(label_files):
            raise ValueError(f"Mismatch between clip files ({len(clip_files)}) and label files ({len(label_files)})")
        
        # Apply percentage filter
        if self.percent < 100:
            num_samples = int(len(clip_files) * self.percent / 100)
            clip_files = clip_files[:num_samples]
            label_files = label_files[:num_samples]
        
        self.data_files = list(zip(clip_files, label_files))
        
        # Fit scaler only for training data, skip if scaler already provided
        if self.scale and self.original_flag == 'train' and not hasattr(self.scaler, 'mean_'):
            self._fit_scaler()
        
    def _fit_scaler(self):
        """Fit scaler on a subset of training data"""
        if self.original_flag != 'train':
            return
            
        print("Fitting scaler on training data...")
        # Sample some files to fit the scaler
        sample_size = min(100, len(self.data_files))
        sample_indices = np.random.choice(len(self.data_files), sample_size, replace=False)
        
        all_data = []
        for idx in sample_indices:
            clip_file, _ = self.data_files[idx]
            with h5py.File(clip_file, 'r') as f:
                data = f['x'][:]  # Shape: (T, C, F) = (12, 19, 256)
                # Reshape to (T*F, C) for fitting scaler
                data = data.transpose(1, 0, 2).reshape(self.channel, -1).T  # (T*F, C)
                all_data.append(data)
        
        # Concatenate all data and fit scaler
        all_data = np.concatenate(all_data, axis=0)
        self.scaler.fit(all_data)
        print(f"Scaler fitted on {all_data.shape[0]} samples with {all_data.shape[1]} features")
        
    def __getitem__(self, index):
        """Get a single sample"""
        clip_file, label_file = self.data_files[index]
        
        # Load clip data
        with h5py.File(clip_file, 'r') as f:
            x = f['x'][:]  # Shape: (T, C, F) = (12, 19, 256)
        
        # Load label data
        with h5py.File(label_file, 'r') as f:
            y = f['y'][()]  # Scalar: 0 or 1
        
        # Reshape data: (T, C, F) -> (C, T*F)
        # T=12 time steps, C=19 channels, F=256 frequency samples per time step
        x = x.transpose(1, 0, 2)  # (C, T, F)
        x = x.reshape(self.channel, -1)  # (C, T*F) = (19, 3072)
        
        # Apply scaling if needed
        if self.scale and hasattr(self, 'scaler'):
            # Transpose for scaler: (C, T*F) -> (T*F, C)
            x_scaled = self.scaler.transform(x.T)  # (T*F, C)
            x = x_scaled.T  # Back to (C, T*F)
        
        # Handle feature selection
        if self.features == 'S':
            # Use only the first channel for single-variate
            x = x[0:1, :]  # (1, T*F)
        # For 'M', use all channels: (19, T*F)
        
        # Ensure correct sequence length
        if x.shape[1] > self.seq_len:
            x = x[:, :self.seq_len]
        elif x.shape[1] < self.seq_len:
            # Pad with zeros if needed
            pad_size = self.seq_len - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)  # (C, seq_len)
        y = torch.tensor(y, dtype=torch.float32)  # scalar

        # --- Apply decomposition augmentation only to training data ---
        if (self.original_flag == 'train'                          # Only apply during training
            and self.aug_decomp
            and torch.rand(1).item() < self.aug_p
            and (not self.aug_only_bg or (self.aug_only_bg and y.item() == 0.0))):
            x = _decomp_augment(x,
                                win=self.aug_win,
                                scale_low=self.aug_scale_low,
                                scale_high=self.aug_scale_high,
                                noise_ratio=self.aug_noise)

        # --- Apply Window Warping augmentation only to training data ---
        if (self.original_flag == 'train'                          # Only apply during training
            and self.aug_ww
            and (not self.aug_ww_only_bg or (self.aug_ww_only_bg and y.item() == 0.0))):
            
            # Step 1 & 2: Trigger sampling + speed ratio selection
            triggered, sub_win_start, sub_win_end, speed_ratio = _window_warping_step1_and_step2(
                seq_len=self.seq_len,
                sampling_rate=self.period,  # 256 Hz
                p_low=self.aug_ww_p_low,
                p_high=self.aug_ww_p_high,
                win_ratio_low=self.aug_ww_win_ratio_low,
                win_ratio_high=self.aug_ww_win_ratio_high,
                margin_sec=self.aug_ww_margin,
                speed_ratios=[0.67, 0.8, 1.25, 1.5]  # Specified speed ratio options
            )
            
            if triggered:
                # Step 3: Time mapping (length-preserving warping)
                x = _window_warping_step3(x, sub_win_start, sub_win_end, speed_ratio)
                
                # Debug output commented out to avoid excessive training logs
                # print(f"WW applied: sub_window [{sub_win_start}:{sub_win_end}], speed_ratio={speed_ratio:.2f}")
                # TODO: Step 4 may require additional processing (e.g., seizure boundary checks)

        # Create dummy time marks (not used in classification)
        seq_x_mark = np.zeros((self.seq_len, 4))  # Dummy time features
        seq_y_mark = np.zeros((1, 4))  # Dummy time features for label
        
        return x, y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_files)
        
    def inverse_transform(self, data):
        """Inverse transform the data if scaler was applied"""
        if self.scale and hasattr(self, 'scaler'):
            return self.scaler.inverse_transform(data)
        return data
        