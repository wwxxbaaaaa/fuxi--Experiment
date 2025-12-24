# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import SequentialModel

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 优化：强制转float32计算，防止FP16溢出
        dtype = x.dtype
        x = x.float()
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype)

class FuXiAlpha(SequentialModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of FuXi Blocks.')
        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--num_buckets', type=int, default=64, help='Number of time buckets.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.max_len = args.history_max
        
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_len + 1, self.emb_size)
        
        # 优化：真正的相对位置编码表 (Range: -L ~ +L)
        self.relative_bias_table = nn.Embedding(2 * self.max_len + 1, self.num_heads)
        nn.init.trunc_normal_(self.relative_bias_table.weight, std=0.02)

        self.fuxi_blocks = nn.ModuleList([
            FuXiBlock(self.emb_size, self.num_heads, args.dropout)
            for _ in range(self.num_layers)
        ])
        
        self.final_norm = RMSNorm(self.emb_size)
        self.dropout = nn.Dropout(args.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, feed_dict):
        item_seq = feed_dict['history_items'] 
        lengths = feed_dict['lengths'].long() 
        batch_size, seq_len = item_seq.shape
        
        # 优化：Expand代替Repeat节省显存
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.i_embeddings(item_seq) + self.p_embeddings(pos)
        x = self.dropout(x)

        # 优化：计算相对位置矩阵 (i-j) 并查表
        range_vec = torch.arange(seq_len, device=self.device)
        # Broadcasting计算差值: [L, L]
        relative_idx = range_vec[None, :] - range_vec[:, None] 
        relative_idx += self.max_len # 偏移至非负区间
        relative_idx = relative_idx.clamp(0, 2 * self.max_len)
        
        # 获取Bias并调整维度: [L, L, H] -> [1, H, L, L]
        rel_bias = self.relative_bias_table(relative_idx).permute(2, 0, 1).unsqueeze(0)

        # 优化：使用 -1e9 代替 -inf，防止SiLU产生NaN
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device) * -1e9, diagonal=1)

        for block in self.fuxi_blocks:
            x = block(x, rel_bias, causal_mask)
            
        x = self.final_norm(x)

        # 取序列最后一位作为特征
        seq_feat = x[torch.arange(batch_size, device=self.device), lengths - 1] 

        if 'item_id' in feed_dict:
            target_ids = feed_dict['item_id']
            target_embs = self.i_embeddings(target_ids)
            scores = (seq_feat.unsqueeze(1) * target_embs).sum(dim=-1)
        else:
            all_embs = self.i_embeddings.weight
            scores = torch.matmul(seq_feat, all_embs.t())
            
        return {'prediction': scores}


class FuXiBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ams = AMSLayer(d_model, n_heads, dropout)
        self.mffn = MFFNLayer(d_model, dropout)

    def forward(self, x, p_bias, mask=None):
        # Pre-Norm结构：x + Layer(Norm(x))
        ams_out = self.ams(x, p_bias, mask)
        x = self.mffn(ams_out, x) # MFFN含有一阶融合和二阶SwiGLU
        return x

class AMSLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.scale = self.d_h ** -0.5
        
        # 优化：合并QKV投影，加速矩阵运算
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_u = nn.Linear(d_model, d_model, bias=False) # Gating u
        
        self.rms_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, p_bias, mask=None):
        batch_size, seq_len, d_model = x.shape
        x_norm = self.rms_norm(x)
        
        # 并行计算QKV
        qkv = self.qkv(x_norm).reshape(batch_size, seq_len, 3, self.n_heads, self.d_h)
        q, k, v = qkv.unbind(dim=2) # [B, L, H, D]
        
        q = q.transpose(1, 2) # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 语义注意力分数
        attn_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 优化：直接注入相对位置Bias (Unified Bias)
        if p_bias is not None:
            attn_score = attn_score + p_bias

        if mask is not None:
            attn_score = attn_score + mask.unsqueeze(0).unsqueeze(0)
            
        # 论文核心：使用SiLU替代Softmax
        attn_prob = F.silu(attn_score)
        attn_prob = self.dropout(attn_prob)
        
        out = torch.matmul(attn_prob, v) 
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        
        # Gating机制 (公式4)
        u = F.silu(self.w_u(x_norm))
        out = out * u 
        
        return out

class MFFNLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        # Stage 1: 融合
        self.w_0 = nn.Linear(d_model, d_model, bias=False)
        
        # Stage 2: SwiGLU
        self.w_1 = nn.Linear(d_model, d_model * 2, bias=False) 
        self.w_3 = nn.Linear(d_model, d_model, bias=False)
        
        self.rms_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 优化：新增内部Dropout，增强稀疏数据鲁棒性
        self.activation_dropout = nn.Dropout(dropout)

    def forward(self, ams_out, x_input):
        # Stage 1: 投影并融合原始输入
        o = self.w_0(ams_out) + x_input
        
        # Stage 2: SwiGLU计算
        o_norm = self.rms_norm(o)
        x1, x2 = self.w_1(o_norm).chunk(2, dim=-1)
        
        hidden = F.silu(x1) * x2
        hidden = self.activation_dropout(hidden) # 内部Dropout
        
        out = self.w_3(hidden) + o # 残差连接
        return out