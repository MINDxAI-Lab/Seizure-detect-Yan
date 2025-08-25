import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention.py: Implements scaled dot-product attention and multi-head attention modules

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # Scaling factor (usually sqrt(d_k))
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch, n_head, len_q, d_k]
        k: [batch, n_head, len_k, d_k]
        v: [batch, n_head, len_v, d_v]
        mask: [batch, 1, len_q, len_k] or None
        '''
        # Compute scaled dot-product attention scores
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [batch, n_head, len_q, len_k]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # Mask out positions

        # Softmax and dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        # Weighted sum of values
        output = torch.matmul(attn, v)  # [batch, n_head, len_q, d_v]
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self,  d_model = -1 ,n_head = 8 , d_k = -1 , d_v = -1 , dropout=0.1):
        super().__init__()
        self.n_head = n_head
        d_k =  d_model // n_head 
        d_v = d_k
        self.d_k = d_k
        self.d_v = d_v

        # Linear projections for queries, keys, values
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # Output projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        '''
        q, k, v: [batch, seq_len, d_model]
        mask: [batch, seq_len, seq_len] or None
        Returns: output [batch, seq_len, d_model], attention weights
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q  # For residual connection

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention: [batch, n_head, seq_len, d_k/d_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Apply scaled dot-product attention
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # Final output projection and dropout
        q = self.dropout(self.fc(q))
        # Residual connection and layer normalization
        q += residual
        q = self.layer_norm(q)
        
        return q, attn