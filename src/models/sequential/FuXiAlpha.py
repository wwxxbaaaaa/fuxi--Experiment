# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.BaseModel import SequentialModel

# 自定义 RMSNorm 解决低版本 PyTorch 兼容性问题
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

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
        self.num_buckets = args.num_buckets

        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_len + 1, self.emb_size)
        self.time_buckets = nn.Embedding(self.num_buckets, self.num_heads)
        self.pos_bias = nn.Embedding(self.max_len + 1, self.num_heads)

        self.fuxi_blocks = nn.ModuleList([
            FuXiBlock(self.emb_size, self.num_heads, self.max_len, args.dropout)
            for _ in range(self.num_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def forward(self, feed_dict):
        item_seq = feed_dict['history_items']
        lengths = feed_dict['lengths']
        batch_size, seq_len = item_seq.shape
        
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.i_embeddings(item_seq) + self.p_embeddings(pos)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        t_bias, p_bias = None, self.pos_bias.weight
        for block in self.fuxi_blocks:
            x = block(x, t_bias, p_bias)
            
        # 1. 获取序列特征 (保持之前的 .long() 修复)
        feat = x[torch.arange(batch_size), (lengths - 1).long()]
        
        # 2. 【关键修复】动态计算得分
        # 训练时，feed_dict 会包含 'item_id' (目标物品+负样本)
        if 'item_id' in feed_dict:
            item_ids = feed_dict['item_id'] # [Batch, num_candidates]
            item_embs = self.i_embeddings(item_ids) # [Batch, num_candidates, emb_size]
            # 特征与特定物品 Embedding 做点积
            # feat [B, D] -> [B, 1, D] 以便广播
            scores = (feat.unsqueeze(1) * item_embs).sum(dim=-1) # [Batch, num_candidates]
        else:
            # 预测全量时 (如果没有指定 item_id)
            all_embs = self.i_embeddings.weight # [Total_Items, D]
            scores = torch.matmul(feat, all_embs.transpose(0, 1)) # [Batch, Total_Items]
            
        return {'prediction': scores}
        
    # 【注意】请删除自定义的 predict 函数，因为 forward 已经可以处理所有情况

class FuXiBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        self.ams = AMSLayer(d_model, n_heads, max_len, dropout)
        self.mffn = MFFNLayer(d_model, dropout)

    def forward(self, x, t_bias, p_bias):
        ams_out = self.ams(x, t_bias, p_bias)
        x = self.mffn(ams_out, x)
        return x

class AMSLayer(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_u = nn.Linear(d_model, d_model)
        self.rms_norm = RMSNorm(d_model)
        
    def forward(self, x, t_bias, p_bias):
        batch_size, seq_len, d_model = x.shape
        x_norm = self.rms_norm(x)
        
        q = self.w_q(x_norm).view(batch_size, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        k = self.w_k(x_norm).view(batch_size, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        v = self.w_v(x_norm).view(batch_size, seq_len, self.n_heads, self.d_h).transpose(1, 2)
        
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_h ** 0.5)
        
        if p_bias is not None:
            # 修复点：动态切片
            if seq_len <= p_bias.size(0):
                curr_p_bias = p_bias[:seq_len, :]
                attn_score = attn_score + curr_p_bias.transpose(0, 1).unsqueeze(0).unsqueeze(2)
            
        attn_prob = F.silu(attn_score)
        
        out = torch.matmul(attn_prob, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_h)
        
        u = F.silu(self.w_u(x_norm))
        out = self.rms_norm(out) * u
        return out

class MFFNLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.w_0 = nn.Linear(d_model, d_model)
        self.w_1 = nn.Linear(d_model, d_model * 2) 
        self.w_2 = nn.Linear(d_model, d_model) 
        self.rms_norm = RMSNorm(d_model)

    def forward(self, ams_out, x_input):
        o = self.w_0(ams_out) + x_input
        res = o
        o_norm = self.rms_norm(o)
        
        x12 = self.w_1(o_norm)
        x1, x2 = x12.chunk(2, dim=-1)
        swiglu_out = (F.silu(x1) * x2)
        
        out = self.w_2(swiglu_out) + res
        return out