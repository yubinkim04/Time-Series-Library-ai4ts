# no_rope.py
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = config.is_causal

        # NOTE: RoPE has been removed for ablation.
        # self.rope = RotaryPositionalEmbedding(...)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # shape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # === RoPE ablation: do NOT rotate q/k ===
        # q, k = self.rope(q, k, x=x)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.ModuleList([CausalSelfAttention(config) for _ in range(config.n_layer)])
        self.pns = nn.ModuleList([nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(config.n_layer)])
        self.lns = nn.ModuleList([nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(config.n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for _ in range(config.n_layer)])

        self.input_proj = nn.Linear(config.n_channel, config.n_embd)
        self.input_norm = nn.LayerNorm(config.n_embd)
        self.output_norm = nn.LayerNorm(config.n_embd)
        self.output_proj = nn.Linear(config.n_embd, config.n_channel)

        # This additive learned positional embedding remains (it is NOT RoPE).
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

    def forward(self, x):
        B, L, C = x.shape
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :L]
        x = self.input_norm(x)
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        transformer_config = SimpleNamespace(
            block_size=1024,
            n_layer=configs.e_layers,
            n_head=configs.n_heads,
            n_embd=configs.d_model,
            dropout=configs.dropout,
            bias=True,
            is_causal=False,          # keep non-causal if you’re doing full-context forecasting
            n_channel=configs.enc_in,
            d_ff=configs.d_ff,
        )
        self.model = Transformer(transformer_config)
        self.forecast_layer = nn.Sequential(
            nn.Linear(configs.seq_len, 512),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, configs.pred_len),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (B, L, D)
        mean, std = x_enc.mean(dim=1, keepdim=True), x_enc.std(dim=1, keepdim=True) + 1e-8
        x_enc = (x_enc - mean) / std
        x_output = self.model(x_enc)
        x_output = self.forecast_layer((x_enc + x_output).permute(0, 2, 1)).permute(0, 2, 1)
        x_output = x_output * std + mean
        return x_output  # (B, L_P, C)
