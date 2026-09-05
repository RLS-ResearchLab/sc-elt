"""
VAE-stage skeleton - Stage 1 of elt-baseline.
Two model variants share this file: RoPE-ViT-VAE (priority) and sincos-ViT-VAE.
Fill in top-down: patchify -> patch embed -> pos-encoding -> attention -> blocks -> VAE.

Ownership convention: VAEEncoder and VAEDecoder each build and own their own
rope_cache_2d (registered as a buffer in __init__), since they may operate on
different spatial grids. Neither forward() takes rope_cache_2d as an argument
from outside -- it's threaded internally to the backbone/blocks/attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import  math
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# 1. Patchify / Unpatchify
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Reused from MAE, adapted for latent channel count.
    x: (B, C, H, W) -> (B, N, patch_dim), N=(H/P)*(W/P), patch_dim=C*P*P
    """
    assert x.ndim ==3 or x.ndim==4,f"EXpected 3D or 4D tensor,got {x.ndim}."
    if x.ndim ==3:
        x = x.unsqueeze(0)
    B,C,H,W = x.shape
    assert H%patch_size == 0,f"H must be divided by the patch size ,got H:{H}, P:{patch_size}"
    assert W%patch_size == 0,f"H must be divided by the patch size ,got H:{H}, P:{patch_size}"
    N = (W//patch_size)*(H//patch_size)
    d_patch = patch_size**2*C
    x = x.reshape(B,C,H//patch_size,patch_size,W//patch_size,patch_size)
    x = x.permute(0,2,4,3,5,1)
    x=x.reshape(B,N,d_patch).contiguous()
    return x


def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int, h: int, w: int) -> torch.Tensor:
    """(B, N, patch_dim) -> (B, C, H, W). Inverse of patchify."""

    assert x.ndim ==3 or x.ndim==2,f"EXpected 2D or 3D tensor,got {x.ndim}."
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B,N,patch_dim = x.shape
    assert patch_dim%out_channels ==0,f"The patch_dim must be devided by the output_chanels,got C:{out_channels},patch_dim:{patch_dim}."
    assert h%patch_size == 0,f"H must be divided by the patch size ,got H:{h}, P:{patch_size}"
    assert w%patch_size == 0,f"H must be divided by the patch size ,got W:{w}, P:{patch_size}"
    assert (h//patch_size)*(w//patch_size) == N,f" the height,width and the patch  size should give the same number of token as the input ,gotH:{h},W:{w}, P:{patch_size} and N:{N} "
    x = x.reshape(B,h//patch_size,w//patch_size,patch_size,patch_size,out_channels)
    x = x.permute(0,5,1,3,2,4)
    x = x.reshape(B,out_channels,h,w).contiguous()
    return x



# ---------------------------------------------------------------------------
# 2. Patch Embedding / Patch Projection
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Linear: patch_dim -> d_model. Start of VAEEncoder, right after patchify."""
    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(in_features=patch_dim,out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class PatchProj(nn.Module):
    """
    Linear: d_model -> patch_dim. Inverse of PatchEmbed -- param order swapped
    (d_model first, patch_dim second) so the signature signals direction.
    End of VAEDecoder, right before unpatchify.
    """
    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features=d_model,
                              out_features=patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
# ---------------------------------------------------------------------------
# 3. Positional Encoding -- TWO variants (config-switched, not stacked)
# ---------------------------------------------------------------------------

# 3.1 Positional Encoding :sin/cos 

def build_1d_sincos_pos_embed(dim: int,positions: torch.Tensor,base: float = 10000.0):
    assert dim % 2 == 0, "dimension must be even."

    i = torch.arange(0,dim,2,dtype=positions.dtype,device=positions.device)
    inv_freq = 1.0 / (base ** (i / dim))
    theta = torch.outer(positions, inv_freq)

    sin = torch.sin(theta)
    cos = torch.cos(theta)

    embed = torch.stack([sin, cos], dim=-1)
    N, D, _ = embed.shape
    embed = embed.reshape(N, 2 * D)
    return embed

def build_2d_sincos_pos_embed(d_model: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    (grid_h*grid_w, d_model). Added right after PatchEmbed, before first block,
    only when attention_type == 'mha'. Skipped entirely for 'rope'.
    """
    assert d_model%4==0 ,"d_model must be divided by 4"
    d_half = d_model//2
    N = grid_w*grid_h
    positions = build_2d_positions(grid_h, grid_w)
    rows = positions[:,0]
    cols = positions[:,1]
    row_embed = build_1d_sincos_pos_embed(dim=d_half,positions=rows)
    col_embed = build_1d_sincos_pos_embed(dim=d_half,positions=cols)
    pos_embed = torch.cat([row_embed,col_embed],dim=-1)
    return pos_embed

# 3.2 Positional Encoding :RoPE

def build_rope_cache(dim: int, seq_len: int, base: float = 10000.0):
    """1D RoPE primitive, called per-axis inside build_rope_cache_2d.
    Returns: (cos, sin), each (seq_len, dim//2)."""
    assert dim % 2 == 0, "the dimension must be even."
    i = torch.arange(0, dim, 2, dtype=torch.float32)          
    inv_freq = 1.0 / (base ** (i / dim))                       
    positions = torch.arange(seq_len, dtype=torch.float32)     
    theta = torch.outer(positions, inv_freq)                  
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, f"last dim must be even, got {x.shape[-1]}"
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
    

def  apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies 1D rope to one axis's slice of head_dim.
    x: (B, n_heads, N, head_dim). cos/sin: (seq_len, head_dim//2) from build_rope_cache,
    where seq_len == N and head_dim//2 == x's last-dim // 2."""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    B, n_heads, N, head_dim = x.shape
    assert cos.shape[-1] == head_dim // 2, f"Expected cos last dim {head_dim // 2}, got {cos.shape}"
    assert sin.shape[-1] == head_dim // 2, f"Expected sin last dim {head_dim // 2}, got {sin.shape}"

    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, head_dim)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)

    rot = rotate_half(x)
    return x * cos + rot * sin
    

def build_2d_positions(grid_h: int, grid_w: int) -> torch.Tensor:
    n = torch.arange(grid_h * grid_w)
    rows = n // grid_w
    cols = n % grid_w
    return torch.stack([rows, cols], dim=-1)

def build_rope_cache_2d(dim: int, grid_h: int, grid_w: int, base: float = 10000.0):
    """
    Axial 2D RoPE: build_rope_cache() x2 (per axis) + build_2d_positions().
    Called once in VAEEncoder.__init__ and once in VAEDecoder.__init__ --
    each owns its own cache (grids may differ).
    """
    assert dim % 4 == 0, "dim must be divisible by 4"

    half_dim = dim // 2
    positions = build_2d_positions(grid_h=grid_h, grid_w=grid_w)

    cos_h, sin_h = build_rope_cache(seq_len=grid_h, dim=half_dim, base=base)
    cos_w, sin_w = build_rope_cache(seq_len=grid_w, dim=half_dim, base=base)
    pos_h, pos_w = positions[:, 0], positions[:, 1]
    cos_h, sin_h = cos_h[pos_h], sin_h[pos_h]
    cos_w, sin_w = cos_w[pos_w], sin_w[pos_w]

    cos = torch.cat([cos_h, cos_w], dim=-1)
    sin = torch.cat([sin_h, sin_w], dim=-1)

    return cos, sin

def apply_rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotates first half of head_dim with row RoPE, second half with col RoPE."""

    dim_half = x.shape[-1] // 2

    x_h, x_w = x[..., :dim_half], x[..., dim_half:]

    cos_h, sin_h = cos[..., :dim_half // 2], sin[..., :dim_half // 2]
    cos_w, sin_w = cos[..., dim_half // 2:], sin[..., dim_half // 2:]

    x_h = apply_rope(x_h, cos_h, sin_h)
    x_w = apply_rope(x_w, cos_w, sin_w)

    return torch.cat([x_h, x_w], dim=-1)

# ---------------------------------------------------------------------------
# 4. Attention Modules
# ---------------------------------------------------------------------------

# --- 4.1 Attention Utilities ---
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, N, d_model) -> (B, n_heads, N, head_dim)"""
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}."
    B, N, d_model = x.shape
    assert d_model % n_heads == 0, f"The d_model must divided by n_head: got d_model:{d_model}, n_head:{n_heads}"
    head_dim = d_model // n_heads
    x = x.reshape(B, N, n_heads, head_dim)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, N, head_dim) -> (B, N, d_model)"""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}."
    B, n_heads, N, head_dim = x.shape
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, N, head_dim * n_heads).contiguous()
    return x


class MultiHeadAttention(nn.Module):
    """Vanilla MHA -- sin-cos model variant. Uses QKVProjection, split_heads,
    scaled_dot_product_attention, merge_heads."""
    def __init__(self, d_model: int, n_heads: int,dropout:float =0.0,bias = True):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout  =nn.Dropout(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model,bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = merge_heads(out)
        out = self.out_proj(out)
        return out

class RoPEAttention(nn.Module):
    """MHA + RoPE on q,k before the dot-product -- PRIORITY variant. Same
    utilities as MultiHeadAttention, plus apply_rope_2d() after split_heads."""
    def __init__(self, d_model: int, n_heads: int,grid_h:int,grid_w:int,dropout:float=0.0,bias = True):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.attn_dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(in_features=d_model,out_features=d_model,bias=bias)
        assert d_model%n_heads == 0,f"d_model must be divided by n_head"
        head_dim = d_model//n_heads
        cos_cache, sin_cache = build_rope_cache_2d(dim=head_dim,grid_h=grid_h,grid_w=grid_w)

        self.register_buffer("cos_cache",cos_cache,persistent=False)
        self.register_buffer("sin_cache",sin_cache,persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        q_rotated = apply_rope_2d(x=q,cos=self.cos_cache,sin=self.sin_cache)
        k_rotated = apply_rope_2d(x=k,cos=self.cos_cache,sin=self.sin_cache)
        out = F.scaled_dot_product_attention(q_rotated, k_rotated, v, dropout_p=self.dropout if self.training else 0.0)
        out = merge_heads(out)
        out = self.out_proj(out)
        return out

        
@dataclass
class AttentionConfig:
    def __post_init__(self):
        assert self.attention_type in ("mha", "rope"), f"invalid attention_type: {self.attention_type}"
        if self.attention_type == "rope":
            assert self.grid_h is not None and self.grid_w is not None, "rope requires grid_h and grid_w"
    d_model: int
    n_heads: int
    attention_type: str          # "mha" | "rope"
    dropout: float = 0.0
    bias: bool = True
    grid_h: Optional[int] = None  # only required for "rope"
    grid_w: Optional[int] = None  # only required for "rope"

#-------------------------------------------------------------
#build attention factory
#--------------------------------------------------------------
def build_attention(cfg: AttentionConfig) -> nn.Module:
    assert cfg.attention_type in SUPPORTED_ATTENTIONS, (
        f"wrong attention type, got {cfg.attention_type}, "
        f"choose from: {list(SUPPORTED_ATTENTIONS)}."
    )

    if cfg.attention_type == "mha":
        return MultiHeadAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            bias=cfg.bias
        )

    if cfg.attention_type == "rope":
        assert cfg.grid_h is not None and cfg.grid_w is not None, \
            "rope attention requires grid_h and grid_w"
        return RoPEAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
            dropout=cfg.dropout,
            bias=cfg.bias
        )

    raise NotImplementedError(cfg.attention_type)
#----------------------------------------------------------------

SUPPORTED_ATTENTIONS = {
    "mha": MultiHeadAttention,
    "rope": RoPEAttention,
    # "gqa": GroupedHeadAttention  -- DEFERRED
}

SUPPORTED_POSITION_ENCODINGS = {
    "sincos": build_2d_sincos_pos_embed,
    "rope": build_rope_cache_2d,
}
# GroupedHeadAttention (GQA): DEFERRED. Orthogonal to position-encoding choice;
# adds a third variable to an already two-model comparison. Revisit post-baseline.

# 5. Transformer Components

class MLP(nn.Module):
    """
    NOTE: in_dim must equal out_dim when used inside TransformerBlock (residual
    add x + mlp(x)). Split signature kept for reuse outside the block only.
    """
    def __init__(self, in_dim: int, out_dim: int, mlp_ratio: float = 4.0,dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim,out_features=out_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TransformerBlock(nn.Module):
    """attention_type: 'mha' | 'rope', selected via SUPPORTED_ATTENTIONS."""
    def __init__(self, attn_cfg: AttentionConfig, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.d_model = attn_cfg.d_model
        self.n_heads = attn_cfg.n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.attn = build_attention(attn_cfg)
        self.mlp = MLP(self.d_model, self.d_model, mlp_ratio,dropout=dropout)  # in==out for residual
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + res
        return x

class TransformerBackbone(nn.Module):
    def __init__(self, attn_cfg: AttentionConfig, depth: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(attn_cfg=attn_cfg,mlp_ratio=mlp_ratio,dropout=dropout)
            for _ in range(depth)
        ])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
# ---------------------------------------------------------------------------
# 6. VAE
# ---------------------------------------------------------------------------
# 6. VAE Config 

@dataclass
class VAEConfig:
    image_size: int
    patch_size: int
    in_channels: int
    d_model: int
    n_heads: int
    depth: int
    latent_dim: int
    attention_type: str  # "mha" | "rope"
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = True
 
    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.patch_dim = self.in_channels * self.patch_size ** 2
 
 #---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    patchify -> PatchEmbed -> (+sincos pos_embed if 'mha') -> TransformerBackbone
    -> mu_logvar_head -> split into mu, logvar.
    RoPE cache ownership lives inside RoPEAttention (built via TransformerBackbone's
    AttentionConfig) -- VAEEncoder itself doesn't touch rope machinery directly.
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(patch_dim=config.patch_dim, d_model=config.d_model)
 
        if config.attention_type == "mha":
            pos_embed = build_2d_sincos_pos_embed(config.d_model, config.grid_h, config.grid_w)
            self.register_buffer("pos_embed", pos_embed, persistent=False)
 
        attn_cfg = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            attention_type=config.attention_type,
            dropout=config.dropout,
            bias=config.bias,
            grid_h=config.grid_h if config.attention_type == "rope" else None,
            grid_w=config.grid_w if config.attention_type == "rope" else None,
        )
        self.backbone = TransformerBackbone(
            attn_cfg, depth=config.depth, mlp_ratio=config.mlp_ratio, dropout=config.dropout
        )
        self.mu_logvar_head = nn.Linear(config.d_model, 2 * config.latent_dim)
 
    def forward(self, x: torch.Tensor):
        """x: (B, C, H, W) -> mu, logvar, each (B, N, latent_dim)"""
        x = patchify(x, patch_size=self.config.patch_size)
        x = self.patch_embed(x)
        if self.config.attention_type == "mha":
            x = x + self.pos_embed
        x = self.backbone(x)
        mu, logvar = self.mu_logvar_head(x).chunk(2, dim=-1)
        return mu, logvar

class VAEDecoder(nn.Module):
    """
    z -> latent_proj -> (+sincos pos_embed if 'mha') -> TransformerBackbone
    -> PatchProjection -> unpatchify.
    Owns its own rope_cache_2d (separate buffer from VAEEncoder's).
    """
    def __init__(self, config:VAEConfig):
        super().__init__()
        self.config = config
        self.latent_proj = nn.Linear(config.latent_dim,config.d_model)
        if config.attention_type == "mha":
            pos_embed = build_2d_sincos_pos_embed(d_model=config.d_model,grid_h=config.grid_h,grid_w=config.grid_w)
            self.register_buffer("pos_embed",pos_embed,persistent=False)
        
        attn_cfg = AttentionConfig(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    attention_type=config.attention_type,
                    dropout=config.dropout,
                    bias=config.bias,
                    grid_h = config.grid_h if config.attention_type == "rope" else None,
                    grid_w=config.grid_w if config.attention_type == "rope" else None
                )
        
        
        self.backbone = TransformerBackbone(attn_cfg=attn_cfg,depth=config.depth,mlp_ratio=config.mlp_ratio,dropout=config.dropout)
        self.patch_proj = PatchProj(d_model=config.d_model,patch_dim=config.patch_dim)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, latent_dim) -> (B, C, H, W)
        latent_proj -> (+pos_embed if mha) -> backbone(..., rope_cache_2d=self.rope_cache_2d)
        -> patch_proj -> unpatchify
        
        """
        x = self.latent_proj(z)
        if  self.config.attention_type == "mha":
            x = x+self.pos_embed
        x = self.backbone(x)
        x = self.patch_proj(x)
        x = unpatchify(x,self.config.patch_size,out_channels=self.config.in_channels,h=self.config.image_size,w=self.config.image_size)
        return x


class VAE(nn.Module):
    """Wraps VAEEncoder + reparameterize + VAEDecoder. attention_type set via config."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.cfg = config
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * std, eps ~ N(0,1)."""
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """Returns: recon, mu, logvar."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        return {
            "recon": self.decoder(z),
            "mu": mu,
            "logvar": logvar
        }

# VAE Variants

VAE_CONFIGS = {

    "T": {
        "d_model": 128,
        "depth": 4,
        "n_heads": 4,
    },

    "S": {
        "d_model": 192,
        "depth": 6,
        "n_heads": 6,
    },

    "B": {
        "d_model": 256,
        "depth": 6,
        "n_heads": 8,
    },

    "L": {
        "d_model": 384,
        "depth": 12,
        "n_heads": 12,
    },

    "XL": {
        "d_model": 512,
        "depth": 16,
        "n_heads": 16,
    },

    "H": {
        "d_model": 768,
        "depth": 24,
        "n_heads": 24,
    },

    "G": {
        "d_model": 1024,
        "depth": 32,
        "n_heads": 32,
    },
}

# Build VAE

def build_vae(cfg):
    model_cfg = cfg["model"]

    if "d_model" in model_cfg:
        vae_cfg = model_cfg

        d_model = vae_cfg["d_model"]
        depth = vae_cfg["depth"]
        n_heads = vae_cfg["n_heads"]

    else:
        vae_cfg = model_cfg["vae"]
        variant = model_cfg["variant"]

        if variant not in VAE_CONFIGS:
            raise ValueError(f"Unknown VAE variant {variant}")

        variant_cfg = VAE_CONFIGS[variant]

        d_model = variant_cfg["d_model"]
        depth = variant_cfg["depth"]
        n_heads = variant_cfg["n_heads"]

    patch_size = vae_cfg["patch_size"]
    in_channels = vae_cfg["in_channels"]
    latent_dim = vae_cfg["latent_dim"]
    attention_type = vae_cfg["attention_type"]

    mlp_ratio = vae_cfg.get("mlp_ratio", 4.0)
    dropout = vae_cfg.get("dropout", 0.0)
    bias = vae_cfg.get("bias", True)

    config = VAEConfig(
        image_size=cfg["data"]["image_size"],
        patch_size=patch_size,
        in_channels=in_channels,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        latent_dim=latent_dim,
        attention_type=attention_type,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        bias=bias,
    )

    return VAE(config)

# ============================================================
# DiT Block Utilities and Helpers
# ============================================================

# Config

@dataclass
class DiTConfig:
    latent_dim: int
    hidden_size: int  # d_model
    depth: int
    num_heads: int
    mlp_ratio: float
    grid_h: int
    grid_w: int
    num_classes: Optional[int] = None
    cfg_dropout: float = 0.1
    dropout: float = 0.01

    # learn_sigma is read from cfg["diffusion"]["learn_sigma"] in build_dit.
    # in_channels and out_channels are derived inside DiT from latent_dim
    # and learn_sigma.

@dataclass
class LoopedDiTConfig:
    dit_config: DiTConfig
    loop_steps: int = 1


# Components -- signatures only, fill in forward() one at a time

class TimestepEmbedder(nn.Module):
    def __init__(self, cfg: DiTConfig,num_timesteps:int):
        super().__init__()
        positions = torch.arange(num_timesteps,dtype=torch.long)
        embedding_cache = build_1d_sincos_pos_embed(dim=cfg.hidden_size,positions=positions)
        self.register_buffer("embedding_cache",embedding_cache)
        mlp_hidden_dim = int(cfg.hidden_size*cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size,mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim,cfg.hidden_size)
        )

    def timestep_embedding(self,t:torch.Tensor)->torch.Tensor:
        emb = self.embedding_cache[t]
        return emb
    
    def forward(self, t):
        return self.mlp(self.embedding_cache[t])


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, cfg_dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(num_classes+1,hidden_size)
        self.cfg_dropout = cfg_dropout
        self.null_index = num_classes
    def forward(self, y):
        if self.training and self.cfg_dropout > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < self.cfg_dropout
            y = torch.where(drop_mask, self.null_index, y)
        return self.embedding(y)

class AdaLNModulation(nn.Module):
    def __init__(self, cfg: DiTConfig, num_modulations: int):
        super().__init__()
        self.num_modulations = num_modulations
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_size,int(cfg.hidden_size*num_modulations))
        )
         # AdaLN-Zero initialization
        nn.init.constant_(self.mlp[-1].weight,0)
        nn.init.constant_(self.mlp[-1].bias,0)

    def forward(self, c:torch.Tensor)->torch.Tensor:
        return self.mlp(c)

class AdaLN(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size,elementwise_affine=False)

    def forward(self, x:torch.Tensor, shift:torch.Tensor, scale:torch.Tensor):
        x = self.norm(x)
        x = x*(1+scale.unsqueeze(1))
        x = x + shift.unsqueeze(1)
        return x
    
class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig,attn_cfg:AttentionConfig):
        super().__init__()
        self.attn = build_attention(attn_cfg)
        self.mlp = MLP(in_dim=cfg.hidden_size,mlp_ratio=cfg.mlp_ratio,dropout=cfg.dropout,out_dim=cfg.hidden_size)
        self.adaLN_attn = AdaLN(cfg)
        self.adaLN_mlp = AdaLN(cfg)
        self.modulation = AdaLNModulation(cfg,num_modulations=6)

    def forward(self, x:torch.Tensor, c:torch.Tensor):
        modulations = self.modulation(c)
        shift_mha,scale_mha,gate_mha,shift_mlp,scale_mlp,gate_mlp = torch.chunk(modulations,6,dim=-1)
        h = self.adaLN_attn(x,shift_mha,scale_mha)
        h = self.attn(h)
        x = x+gate_mha.unsqueeze(1)*h
        h = self.adaLN_mlp(x,shift_mlp,scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1)*h
        return x
    
class DitBackbone(nn.Module):
    def __init__(self,cfg:DiTConfig,attn_cfg:AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(DiTBlock(cfg=cfg,attn_cfg=attn_cfg)
                                 for _ in range(cfg.depth))
    def forward(self,x:torch.Tensor,c:torch.Tensor)->torch.Tensor:
        for blk in self.blocks:
            x = blk(x,c)
        return x
    
class FinalLayer(nn.Module):
    def __init__(self, cfg: DiTConfig, out_channels: int):
        super().__init__()
        self.modulation = AdaLNModulation(cfg,num_modulations=2)
        self.norm = AdaLN(cfg)
        self.linear = nn.Linear(cfg.hidden_size,out_channels)

          # Final zero initialization
        nn.init.constant_(self.linear.weight,0)
        nn.init.constant_(self.linear.bias,0)

    def forward(self, x:torch.Tensor, c:torch.Tensor):
        shift, scale = torch.chunk(self.modulation(c),2,dim=-1)
        x = self.norm(x,shift,scale)
        x = self.linear(x)

        return x

# DiT

class DiTBase(nn.Module):
    def __init__(self,cfg:DiTConfig,attn_cfg:AttentionConfig,num_timesteps:int,learn_sigma:bool):
        super().__init__()
        self.cfg  = cfg
        if attn_cfg.attention_type == "mha":
            pos_embed = build_2d_sincos_pos_embed(
                d_model=cfg.hidden_size,
                grid_h=cfg.grid_h,
                grid_w=cfg.grid_w,
            )
            self.register_buffer(
                "pos_embed",
                pos_embed.unsqueeze(0),
                persistent=False,
            )
        else:
            self.pos_embed = None
        self.num_timesteps = num_timesteps
        self.learn_sigma = learn_sigma
        self.in_channels = cfg.latent_dim
        self.out_channels = cfg.latent_dim*(2 if learn_sigma  else 1)
        self.effective_num_classes = cfg.num_classes if  cfg.num_classes is not None else 1
        self.x_embedder = nn.Linear(cfg.latent_dim,cfg.hidden_size)
        self.t_embedder = TimestepEmbedder(cfg,num_timesteps)
        self.y_embedder = LabelEmbedder(self.effective_num_classes,cfg.hidden_size,cfg.cfg_dropout)
        self.backbone = DitBackbone(cfg=cfg,attn_cfg=attn_cfg)
        self.final_layer = FinalLayer(cfg,self.out_channels)

    def forward_features(self,z,t,y=None):
        x = self.x_embedder(z)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        c = self._build_condition(z,t,y)
        x = self.forward_backbone(x,c)
        return x,c
    
    def forward_backbone(self, x:torch.Tensor, c:torch.Tensor):
        return self.backbone(x, c)
    
    def _build_condition(self,z,t,y):
        t_emb = self.t_embedder(t)
        if y is None:
            y = torch.full((z.shape[0],),self.effective_num_classes,device=z.device,dtype=torch.long)  #null token index
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        return c

    def forward(self,z:torch.Tensor,t,y = None):
        x,c = self.forward_features(z,t,y)
        x = self.final_layer(x,c)
        if self.learn_sigma:
            eps,v = torch.chunk(x,2,dim=-1)
            return eps,v
        return x
    
class DiT(DiTBase):
    pass 

class LoopedDiT(DiTBase):
    def __init__(self,cfg:LoopedDiTConfig,attn_cfg:AttentionConfig,num_timesteps:int,learn_sigma:bool):
        super().__init__(cfg.dit_config,attn_cfg,num_timesteps,learn_sigma)
        self.loop_cfg = cfg
    def forward_looped(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        record: None | int | list[int] | Literal["all"] = None,
    ):
            history = {}
            if record is None:
                for l in range(self.loop_cfg.loop_steps):
                    x = self.backbone(x,c)
            elif isinstance(record,int):
                for  l in range(self.loop_cfg.loop_steps):
                    x = self.backbone(x,c)
                    if  l+1 == record:
                        history[l+1] = x
            elif isinstance(record,list):
                for l in range(self.loop_cfg.loop_steps):
                    x = self.backbone(x,c)
                    if l+1 in record:
                        history[l+1] = x
            elif record == "all" :
                for l in range(self.loop_cfg.loop_steps):
                    x = self.backbone(x,c)
                    history[l+1] = x
            else:
                raise ValueError(f"Invalid record value: {record!r}, expected None, int, list[int], or 'all'")
            return x,history

    def forward_features(self,z,t,y=None,record = None):
        x = self.x_embedder(z)
        if self.pos_embed is  not None:
            x = x + self.pos_embed
        c = self._build_condition(z,t,y)
        x,history = self.forward_looped(x,c,record)
        return x,c,history
    
    def forward_elastic(self,z,t,y=None,inference_loops:int=2):
        x = self.x_embedder(z)
        if self.pos_embed is  not None:
            x = x + self.pos_embed
        c = self._build_condition(z,t,y)
        for _ in range(inference_loops):
            x = self.backbone(x,c)
        x =  self.final_layer(x,c)
        if self.learn_sigma:
            eps,v = torch.chunk(x,2,dim=-1)
            return eps,v
        return  x
    
    def forward(self, z, t, y=None, record=None):
        x, c, history = self.forward_features(z, t, y, record)
        x = self.final_layer(x, c)
        if self.learn_sigma:
            eps, v = torch.chunk(x, 2, dim=-1)
            out = (eps, v)
        else:
            out = x
        if record is None:
            return out           
        return out,c, history
        
# REPA
class REPA(nn.Module):
    def  __init__(self,dit_dim:int,repr_dim:int):
        super().__init__()
        self.proj = nn.Linear(dit_dim,repr_dim)

    def forward(self,dit_features,target_features):
        z = self.proj(dit_features)
        z = F.normalize(z,dim=-1)
        target_features = F.normalize(target_features,dim=-1)
        loss = 1-(z*target_features).sum(-1).mean()
        return loss
    
#  Dit Factory

#1.Vanilla Dit

def build_dit(cfg):

    variant = cfg["model"]["variant"]
    dit_model_cfg = cfg["model"]["dit"]

    if variant not in DIT_CONFIGS:
        raise ValueError(
            f"Unknown DiT variant {variant}. "
            f"Available: {list(DIT_CONFIGS.keys())}"
        )

    variant_cfg = DIT_CONFIGS[variant]

    dit_cfg = DiTConfig(
        latent_dim=dit_model_cfg["latent_dim"],
        hidden_size=variant_cfg["hidden_size"],
        depth=variant_cfg["depth"],
        num_heads=variant_cfg["num_heads"],
        mlp_ratio=dit_model_cfg["mlp_ratio"],
        grid_h=dit_model_cfg["grid_h"],
        grid_w=dit_model_cfg["grid_w"],
        num_classes=dit_model_cfg["num_classes"],
        cfg_dropout=dit_model_cfg["cfg_dropout"],
        dropout=dit_model_cfg["dropout"],
    )

    attn_cfg = AttentionConfig(
        d_model=dit_cfg.hidden_size,
        n_heads=dit_cfg.num_heads,
        attention_type=cfg["model"]["attention"]["type"],
        dropout=dit_cfg.dropout,
        bias=True,
        grid_h=dit_cfg.grid_h,
        grid_w=dit_cfg.grid_w,
    )

    return DiT(
        dit_cfg,
        attn_cfg,
        num_timesteps=cfg["diffusion"]["timestep"],
        learn_sigma=cfg["diffusion"]["learn_sigma"],
    )

# 2. Looped Dit

def build_looped_dit(cfg):

    variant = cfg["model"]["variant"]
    dit_model_cfg = cfg["model"]["dit"]

    if variant not in LOOPED_DIT_CONFIGS:
        raise ValueError(
            f"Unknown Looped DiT variant {variant}"
        )

    variant_cfg = LOOPED_DIT_CONFIGS[variant]

    dit_cfg = DiTConfig(
        latent_dim=dit_model_cfg["latent_dim"],
        hidden_size=variant_cfg["hidden_size"],
        depth=variant_cfg["depth"],
        num_heads=variant_cfg["num_heads"],
        mlp_ratio=dit_model_cfg["mlp_ratio"],
        grid_h=dit_model_cfg["grid_h"],
        grid_w=dit_model_cfg["grid_w"],
        num_classes=dit_model_cfg["num_classes"],
        cfg_dropout=dit_model_cfg["cfg_dropout"],
        dropout=dit_model_cfg["dropout"],
    )

    loop_cfg = LoopedDiTConfig(
        dit_config=dit_cfg,
        loop_steps=variant_cfg["loop_steps"],
    )

    attn_cfg = AttentionConfig(
        d_model=dit_cfg.hidden_size,
        n_heads=dit_cfg.num_heads,
        attention_type=cfg["model"]["attention"]["type"],
        dropout=dit_cfg.dropout,
        bias=True,
        grid_h=dit_cfg.grid_h,
        grid_w=dit_cfg.grid_w,
    )

    return LoopedDiT(
        loop_cfg,
        attn_cfg,
        num_timesteps=cfg["diffusion"]["timestep"],
        learn_sigma=cfg["diffusion"]["learn_sigma"],
    )
     
DIT_CONFIGS = {
    # Small models (debugging / experiments)
    "DiT-Tiny": {
        "hidden_size": 192,
        "depth": 6,
        "num_heads": 3,
    },

    "DiT-S": {
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
    },


    # Original DiT paper scale

    "DiT-B": {
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
    },

    "DiT-L": {
        "hidden_size": 1024,
        "depth": 24,
        "num_heads": 16,
    },

    "DiT-XL": {
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
    },


    # Large research scale

    "DiT-XXL": {
        "hidden_size": 1536,
        "depth": 48,
        "num_heads": 24,
    },


    "DiT-H": {
        "hidden_size": 1792,
        "depth": 48,
        "num_heads": 28,
    },


    "DiT-G": {
        "hidden_size": 2048,
        "depth": 48,
        "num_heads": 32,
    },


    # Extreme scale (LLM-like)
    "DiT-3B": {
        "hidden_size": 2560,
        "depth": 48,
        "num_heads": 40,
    },


    "DiT-7B": {
        "hidden_size": 4096,
        "depth": 32,
        "num_heads": 32,
    },

}
# Vanilla looped  Dit
LOOPED_DIT_CONFIGS = {
    # Small models
    "Looped-DiT-Tiny": {
        "hidden_size": 192,
        "depth": 6,
        "num_heads": 3,
        "loop_steps": 4,
    },


    "Looped-DiT-S": {
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
        "loop_steps": 4,
    },


    # Original DiT paper scale
    "Looped-DiT-B": {
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "loop_steps": 4,
    },


    "Looped-DiT-L": {
        "hidden_size": 1024,
        "depth": 24,
        "num_heads": 16,
        "loop_steps": 4,
    },


    "Looped-DiT-XL": {
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "loop_steps": 4,
    },


    # Large research scale
    "Looped-DiT-XXL": {
        "hidden_size": 1536,
        "depth": 48,
        "num_heads": 24,
        "loop_steps": 4,
    },


    "Looped-DiT-H": {
        "hidden_size": 1792,
        "depth": 48,
        "num_heads": 28,
        "loop_steps": 4,
    },


    "Looped-DiT-G": {
        "hidden_size": 2048,
        "depth": 48,
        "num_heads": 32,
        "loop_steps": 4,
    },


    # Extreme scale
    "Looped-DiT-3B": {
        "hidden_size": 2560,
        "depth": 48,
        "num_heads": 40,
        "loop_steps": 4,
    },


    "Looped-DiT-7B": {
        "hidden_size": 4096,
        "depth": 32,
        "num_heads": 32,
        "loop_steps": 4,
    },
}
# ---------------------------------------------------------------------------
MODEL_BUILDERS = {
    "vae": build_vae,
    "dit": build_dit,
    "looped_dit": build_looped_dit,
    #"elt": build_elt,
}

def build_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model type '{name}', expected one of {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](cfg)