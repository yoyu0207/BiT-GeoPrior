"""
transformer_block.py — 共用 Transformer Encoder Block
BiT 系列模型共享同一个 Transformer 块，避免重复定义。
"""

import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder Block（单层）。"""

    def __init__(self, dim: int, heads: int,
                 mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
