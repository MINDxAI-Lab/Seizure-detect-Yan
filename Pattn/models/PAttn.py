import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from  models.Attention import MultiHeadAttention

# Model definition for PAttn: Patch-based Attention for EEG Binary Classification
# Includes patch extraction, multi-head attention, and a single-output classification head.
    
class Encoder_LLaTA(nn.Module):
    """
    Transformer-based encoder for time series (not used in main PAttn).
    Projects input to hidden_dim and applies TransformerEncoder.
    """
    def __init__(self, input_dim , hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_LLaTA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.linear(x)
        # Transformer expects [seq_len, batch, features]
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        return x 

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    Used for time series decomposition (trend extraction).
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad both ends of the time series to preserve length after pooling
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # Pool along time axis
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block: splits time series into trend (moving average) and residual.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PAttn(nn.Module):
    """
    Patch-based Attention Model for EEG Binary Classification.
    - Extracts overlapping patches from EEG time series
    - Applies linear embedding and multi-head self-attention
    - Aggregates features and outputs a single logit for BCEWithLogitsLoss
    """
    def __init__(self, configs, device):
        super(PAttn, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size 
        self.stride = configs.patch_size //2 
        
        self.d_model = configs.d_model
        self.method = configs.method
       
        # Number of patches extracted from the sequence
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        # Pad sequence for patch extraction
        self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
        # Linear embedding for each patch
        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        # Multi-head attention block
        self.basic_attn = MultiHeadAttention(d_model =self.d_model )
        # Classification head for binary classification (single output for BCEWithLogitsLoss)
        self.cls_head = nn.Linear(self.d_model, 1)

    def norm(self, x, dim =0, means= None , stdev=None):
        """
        Normalize input along specified dimension.
        If means and stdev are provided, denormalize; otherwise, compute and normalize.
        Returns normalized x, means, stdev.
        """
        if means is not None :  
            return x * stdev + means
        else : 
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
            return x , means ,  stdev 
            
    def forward(self, x):
        """
        Forward pass for PAttn model.
        Input: x [Batch, Channel, seq_len]
        Output: logits [Batch, 1] for binary classification
        """
        if self.method == 'PAttn':
            B , C = x.size(0) , x.size(1)
            # Normalize each channel's time series
            x , means, stdev  = self.norm(x , dim=2)
            # Pad sequence for patch extraction
            x = self.padding_patch_layer(x)
            # Extract overlapping patches: [Batch, Channel, patch_num, patch_size]
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            # Patch embedding: [Batch, Channel, patch_num, d_model]
            x = self.in_layer(x)
            # Reshape for attention: [(Batch * Channel), patch_num, d_model]
            x =  rearrange(x, 'b c m l -> (b c) m l')
            # Self-attention across patches
            x , _ = self.basic_attn( x ,x ,x )
            # Global average pooling across patches: [(Batch * Channel), d_model]
            x = x.mean(dim=1)
            # Reshape back: [Batch, Channel, d_model]
            x = rearrange(x, '(b c) l -> b c l', b=B, c=C)
            # Global average pooling across channels: [Batch, d_model]
            x = x.mean(dim=1)
            # Classification head: [Batch, 1] - binary classification logits for BCEWithLogitsLoss
            x = self.cls_head(x)
            return x  
            