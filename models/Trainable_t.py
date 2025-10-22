import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import math
 
 
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding (RoPE) for Q/K.
    Expects q, k of shape (batch, heads, seq_len, dim).
    """

    def __init__(self, dim, base: float = 10000.0, trainable_flag=False, n_head=None):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.n_head = n_head
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.trainable_flag = trainable_flag
        if trainable_flag:
            self.trainable_t = nn.Linear(self.dim * self.n_head, n_head)
        inv_freq = (
                1.0
                / (
                    self.base
                    ** (torch.arange(0, self.dim, 2) / self.dim)
                )
            )
        #self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq = nn.Parameter(inv_freq)

    @staticmethod
    def _rotate_half(x):
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)  # (L, )
        freqs = torch.einsum("s,f->sf", t, self.inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        self.cos_cached = cos.unsqueeze(0).unsqueeze(0)
        self.sin_cached = sin.unsqueeze(0).unsqueeze(0)

    def build_cos_sin_trainable(self, x):
        B, L, D = x.shape
        t = self.trainable_t(x)  # (B, L, h)
        t = F.softplus(t)
        t = t.cumsum(dim=1)  # (B, L, h)
        freqs = torch.einsum("bsh,f->bhsf", t, self.inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (B, h, L, d)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (B, h, L, d)
        return cos, sin

    def _get_cos_sin(self, seq_len: int, device, dtype):
        need_build = (
            self.cos_cached is None
            or self.cos_cached.size(2) < seq_len
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        )
        if need_build:
            self._build_cache(seq_len, device, dtype)
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_index: int = 0, x=None):
        b, h, s, d = q.shape
        assert d == self.dim, f"Expected dim {self.dim}, got {d}"
        device, dtype = q.device, q.dtype
        if self.trainable_flag:
            assert x is not None, "x should not be None when trainable_flag=True"
            cos, sin = self.build_cos_sin_trainable(x)
        else:
            cos, sin = self._get_cos_sin(start_index + s, device, dtype)
        cos = cos[:, :, start_index:start_index + s, :]
        sin = sin[:, :, start_index:start_index + s, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

 
 
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
 
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
 
 
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
 
        self.rope = RotaryPositionalEmbedding(
            config.n_embd // self.n_head,
            n_head=self.n_head,
            trainable_flag=True,  # <-- start with fixed RoPE
        )
 
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
 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = self.rope(q, k, x=x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
 
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
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
            is_causal=False,
            n_channel=configs.enc_in,
            d_ff=configs.d_ff
        )
        self.model = Transformer(transformer_config)
        #self.forecast_layer = nn.Linear(configs.seq_len, configs.pred_len)
        self.forecast_layer = nn.Sequential(nn.Linear(configs.seq_len, 512),
                                            nn.GELU(),
                                            nn.Dropout(configs.dropout),
                                            nn.Linear(512, configs.pred_len))
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
 
 