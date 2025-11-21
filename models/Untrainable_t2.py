import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import math
import numpy as np
from typing import Tuple

# ================================================================
# Symplectic Positional Embedding (IDENTICAL to your second code)
# ================================================================

@torch.compiler.disable
@torch.no_grad()
def init_nonrope_bands(module, L_max: int, rho_std: float = 0.03):
    H, K = module.H, module.K
    device = module.alpha_base.device
    dtype  = module.alpha_base.dtype  #（fp32/fp16/bf16）

    # 1) band: log-uniform in [2π/L_max, π/2]
    omega_min = (2.0 * math.pi) / float(L_max)
    omega_max = (0.5 * math.pi)

    log_omin = math.log(omega_min)
    log_omax = math.log(omega_max)

    log_omega = torch.linspace(log_omin, log_omax, K, device=device, dtype=dtype)  # (K,)
    # alpha_base/beta_base
    alpha_base = log_omega.reshape(1, K)
    beta_base  = log_omega.reshape(1, K)

    module.alpha_base.copy_(alpha_base)
    module.beta_base.copy_(beta_base)

    module.delta_alpha.zero_()
    module.delta_beta.zero_()
    module.rho.normal_(mean=0.0, std=rho_std)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def _pairwise_blocks(x: torch.Tensor) -> torch.Tensor:
    # *prefix, D = x.shape
    # assert D % 2 == 0, "D must be even"
    return x.reshape(*list(x.size()[:-1]), -1, 2)

def _apply_J2_block(u: torch.Tensor) -> torch.Tensor:
    x, y = u.unbind(dim=-1)
    return torch.stack((y, -x), dim=-1)

def apply_J(x: torch.Tensor) -> torch.Tensor:
    xb = _pairwise_blocks(x)
    yb = _apply_J2_block(xb)
    return yb.reshape_as(x)  # yb.view_as(x)

def _init_base_omegas(head_dim: int, base: int = 10_000) -> torch.Tensor:
    assert head_dim % 2 == 0
    # (K,)
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

def _expand_to_HK(t: torch.Tensor, H, K) -> torch.Tensor:
    H = int(H)
    K = int(K)
    if t.size(0) == 1 and t.size(1) == 1:
        return t.expand(H, K)
    if t.size(0) == 1 and t.size(1) == K:
        return t.expand(H, -1)
    if t.size(0) == H and t.size(1) == 1:
        return t.expand(-1, K)
    return t

def apply_symplectic_pos_emb(
    x: torch.Tensor,
    alpha: torch.Tensor,  # (H,K)
    beta: torch.Tensor,   # (H,K)
    rho: torch.Tensor,    # (H,K)
    transpose: bool = False,
    rho_clip: float = 0.99,
    eps: float = 1e-6,
    clock=None,
) -> torch.Tensor:
    B, L, H, D = list(x.detach().size())
    K = D // 2

    alpha = alpha.reshape(H, K)
    beta  = beta.reshape(H, K)
    rho = rho.reshape(H, K)
    # rho   = torch.clamp(rho, -rho_clip, rho_clip)
    clipping = rho_clip * torch.ones_like(rho)
    rho = torch.maximum(torch.minimum(rho, clipping), -clipping)

    a = torch.exp(alpha)
    b = torch.exp(beta)
    c = rho * torch.sqrt(a * b)

    detK  = a * b - c * c
    omega = torch.sqrt(detK + eps)

    if clock is None:
        t = torch.arange(L, device=x.device).reshape(1, L, 1, 1)
    else:
        t = clock.to(device=x.device)
    if transpose:
        t = -t

    omega_f = omega.to(device=x.device).reshape(1, 1, H, K)

    theta  = t * omega_f
    cos_t  = theta.cos()
    sin_om = theta.sin() / (omega_f + eps)

    x1, x2 = x.view(B, L, H, K, 2).unbind(-1)

    y1 = cos_t * x1 + sin_om * (c * x1 + b * x2)
    y2 = cos_t * x2 + sin_om * (-a * x1 - c * x2)

    return torch.stack((y1, y2), dim=-1).view(B, L, H, D)

# ---------------- module ----------------

class SymplecticPE(nn.Module):
    """
    Symplectic Positional Embedding (SyPE) with flexible parameter sharing.

    share_mode:
      - "global"         -> delta/rho shapes (1,1)       -> 3 params
      - "per_head"       -> delta/rho shapes (H,1)       -> 3*H
      - "per_block"      -> delta/rho shapes (1,K)       -> 3*K
      - "per_head_block" -> delta/rho shapes (H,K)       -> 3*H*K

    NOTE:
      RoPE base spectrum (alpha_base/beta_base) is ALWAYS (1,K) so you never lose the per-block RoPE frequencies.
      Deltas are added on top and broadcast to (H,K) according to share_mode.
    """
    def __init__(
        self,
        n_head: int,
        head_dim: int,
        base: int = 10_000,
        learnable: bool = True,
        share_mode: str = "per_head_block",
        fix_rope_base: bool = True,
        apply_J_for_keys_by_default: bool = True,
        # non-RoPE base init (optional)
        nonrope_init: bool = False,
        nonrope_log_mean: float = -3.0,
        nonrope_log_std: float = 0.02,
        nonrope_rho_std: float = 0.02,
        learnable_clock: bool = True,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        assert share_mode in {"global", "per_head", "per_block", "per_head_block"}
        self.H = n_head
        self.D = head_dim
        self.K = head_dim // 2
        self.share_mode = share_mode
        self.apply_J_for_keys_by_default = apply_J_for_keys_by_default
        self.fix_rope_base = fix_rope_base

        # RoPE base spectrum: (1,K) always
        if not nonrope_init:
            omega = _init_base_omegas(head_dim=head_dim, base=base).reshape(1, self.K)  # (1,K)
            alpha_base = torch.log(omega + 1e-30)   # (1,K)
            beta_base  = torch.log(omega + 1e-30)   # (1,K)
        else:
            # non-RoPE base: per-block random logs (still (1,K), shared across heads)
            alpha_base = torch.empty(1, self.K).normal_(mean=nonrope_log_mean, std=nonrope_log_std)
            beta_base  = torch.empty(1, self.K).normal_(mean=nonrope_log_mean, std=nonrope_log_std)

        # register base
        self.alpha_base = nn.Parameter(alpha_base, requires_grad=(not fix_rope_base and learnable))
        self.beta_base  = nn.Parameter(beta_base,  requires_grad=(not fix_rope_base and learnable))

        # deltas/rho shapes by share_mode
        if share_mode == "global":
            shape = (1, 1)
        elif share_mode == "per_head":
            shape = (self.H, 1)
        elif share_mode == "per_block":
            shape = (1, self.K)
        else:  # "per_head_block"
            shape = (self.H, self.K)

        self.delta_alpha = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        self.delta_beta  = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        if nonrope_init:
            self.rho = nn.Parameter(torch.empty(shape).normal_(mean=0.0, std=nonrope_rho_std), requires_grad=learnable)
        else:
            self.rho = nn.Parameter(torch.zeros(shape), requires_grad=learnable)

        self.learnable_clock = learnable_clock
        if learnable_clock:
            self.clock = nn.Linear(self.H * self.D, self.H * self.K)

    def _expanded_params_HK(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # alpha_base/beta_base: (1,K) -> (H,K)
        alpha_base = self.alpha_base.expand(self.H, self.K)
        beta_base  = self.beta_base .expand(self.H, self.K)

        # deltas/rho: (pH,pK) with pH in {1,H}, pK in {1,K} -> (H,K)
        delta_alpha = _expand_to_HK(self.delta_alpha, self.H, self.K)
        delta_beta  = _expand_to_HK(self.delta_beta,  self.H, self.K)
        rho         = _expand_to_HK(self.rho,         self.H, self.K)

        alpha = alpha_base + delta_alpha
        beta  = beta_base  + delta_beta
        return alpha, beta, rho

    def forward(self, x: torch.Tensor, *, is_key: bool = False, transpose: bool = False, clock=None) -> torch.Tensor:
        alpha, beta, rho = self._expanded_params_HK()
        y = apply_symplectic_pos_emb(x, alpha, beta, rho, transpose=transpose, clock=clock)
        if is_key and self.apply_J_for_keys_by_default:
            y = apply_J(y)
        return y

    def rpe(self, q: torch.Tensor, k: torch.Tensor, clock_x=None):
        if self.learnable_clock:
            assert clock_x is not None, "When learnable_clock=True, pass clock_x of shape (B, L, H*D)."
            clock = self.clock(clock_x).reshape(q.size(0), q.size(1), self.H, -1)  # (B, L, H, K)
            clock = F.softplus(clock)
            clock = clock.cumsum(dim=1)
        else:
            clock = None
        q_pos = self.forward(q, is_key=False, transpose=False, clock=clock)
        k_pos = self.forward(k, is_key=True,  transpose=False, clock=clock)
        return q_pos, k_pos

# ================================================================
# Rest of your model, now USING SymplecticPE instead of RoPE
# ================================================================

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
 
        # --- replaced RotaryPositionalEmbedding with SymplecticPE ---
        self.rope = SymplecticPE(
            n_head=self.n_head,
            head_dim=config.n_embd // self.n_head,
            base=10_000,
            learnable=True,
            share_mode=getattr(config, "share_mode", "per_head_block"),
            fix_rope_base=True,
            apply_J_for_keys_by_default=True,
            nonrope_init=getattr(config, "nonrope_init", False),
            # if you later expose these via args, pull from config the same way:
            # nonrope_log_mean=getattr(config, "nonrope_log_mean", -3.0),
            # nonrope_log_std=getattr(config, "nonrope_log_std", 0.02),
            # nonrope_rho_std=getattr(config, "nonrope_rho_std", 0.02),
            learnable_clock=False,
        )
        # ------------------------------------------------------------
 
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

        # shapes: (B, T, H, D)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)

        # Symplectic position embedding (identical usage to your second code)
        q, k = self.rope.rpe(q, k, clock_x=x)   # q,k: (B, T, H, D)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, T, D)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, H, T, D)
 
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
            d_ff=configs.d_ff,
            # >>> NEW: SyPE config coming from run.py / args <<<
            share_mode=getattr(configs, "share_mode", "per_head_block"),
            nonrope_init=getattr(configs, "nonrope_init", False)
        )
        self.model = Transformer(transformer_config)
        #self.forecast_layer = nn.Linear(configs.seq_len, configs.pred_len)
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
