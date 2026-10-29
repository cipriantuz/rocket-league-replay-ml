"""HMLTv2 — True 2-level Hierarchical Match-Level Transformer for win prediction.

Why this supersedes HMLTv1 (hierarchical_winning.py)
------------------------------------------------------
v1 feeds 23-dim *hand-engineered* match features to a Transformer.  On 31K
replays it ties tabular baselines (LogReg 75.0% / 0.833 AUC) because the 23
features already encode the dominant signal (score differential, time, possession)
that a linear model captures equally well.

v2 consumes RAW per-frame physics so it can learn temporal and tactical patterns
(momentum buildup, rotation discipline, pressure sequences) that the engineered
aggregates flatten away.  The architecture is genuinely hierarchical:

  L1 — Frame Window Encoder
    Input:  (W, 45, 88) — W windows × 45 frames × 88 raw physics features
    Output: (W, d_win)  — one latent vector per 3-second window
    Impl:   1D-CNN stack (Conv1d over the time axis, 88→128→d_win) with GELU +
            residual connections, then masked mean-pool over the 45 time-steps.
    Rationale: CNNs are parameter-efficient for local-temporal patterns (who has
    the ball right now, aerial momentum) vs a full Transformer over 45 frames
    (45 × 88 = 3960 elements/window, expensive per-window).

  [HYBRID path, recommended]
    Concat window_embed (d_win) with the 23-dim hand-engineered match-feature
    vector for the same window → (d_win + 23).  A LayerNorm gate fuses them.
    Motivation: the engineered features carry score / time priors that L1 would
    otherwise need many epochs to reinvent from raw physics.  Giving both gives
    the model the best of both worlds.

  L2 — Match Sequence Transformer
    Input:  (B, T_win, d_win[+23]) — sequence of window embeddings per match
    Output: (B,)                   — raw logit; sigmoid → P(blue wins)
    Impl:   re-uses the HMLT v1 structure (input projection, sinusoidal PE,
            TransformerEncoder, masked mean-pool, MLP head).

Input/Output contract
---------------------
  frame_windows   float32 (B, T_win, 45, 88)  — raw physics per window
  match_feats     float32 (B, T_win, 23) | None — engineered features (hybrid)
  pad_mask        bool    (B, T_win)             — True = padding window
  -> logit        float32 (B,)                  — raw logit for P(blue wins)

Label: 0 = blue wins, 1 = orange wins (same as v1 / MatchSequenceDataset).

Interpretability hook
---------------------
Call forward(..., return_attn=True) to also get the last-layer attention weights
from L2 (shape (B, n_heads, T_win, T_win)).  This enables the "per-window
probability trajectory + attention interpretability" capability claim in the
dissertation even if accuracy gains are modest.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (identical to v1 — reused directly)
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Supports sequences up to max_len positions.  Added to projected input
    embeddings before the L2 Transformer layers.
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# L1 — per-window Frame Encoder (1D-CNN over time axis)
# ---------------------------------------------------------------------------

class _ResConv1dBlock(nn.Module):
    """Residual 1D-Conv block: Conv → BN → GELU → Conv → BN + skip.

    The skip connection uses a 1×1 conv when in_ch != out_ch so the identity
    shortcut always has the right channel count.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2  # 'same' padding for odd kernels
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T)  → (B, out_ch, T)."""
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class FrameWindowEncoder(nn.Module):
    """L1 encoder: maps a raw-frame window (45, n_features) → a latent vector (d_win,).

    Applied independently to every window in a match sequence.  The encoder is
    shared across all windows (weight-tied), keeping parameter count manageable.

    Architecture:
        Conv1d stack: n_features → 128 → d_win (two residual blocks)
        Global mean-pool over the time steps → (d_win,)

    Args:
        n_frames:    Frames per window (default 45 = 3 s @ 15 Hz).
        n_features:  Raw physics features per frame (default 133 for universal).
                     Pass 88 for backward-compat with old 2v2-only checkpoints.
        d_win:       Output embedding dimension.
        kernel_size: Conv kernel over the time axis.
    """

    def __init__(
        self,
        n_frames: int = 45,
        n_features: int = 133,
        d_win: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_features = n_features
        self.d_win = d_win

        # Two residual CNN blocks: 88 → 128 → d_win
        hidden = 128
        self.block1 = _ResConv1dBlock(n_features, hidden, kernel_size)
        self.block2 = _ResConv1dBlock(hidden, d_win, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of frame windows.

        Args:
            x: (B_eff, n_frames, n_features) — B_eff = B × T_win (windows
               flattened from the match dimension before calling this).

        Returns:
            embed: (B_eff, d_win) — one vector per window.
        """
        # Conv1d expects (B, C, L): permute (B_eff, T, F) → (B_eff, F, T)
        h = x.permute(0, 2, 1)          # (B_eff, n_features, n_frames)
        h = self.block1(h)               # (B_eff, 128, n_frames)
        h = self.block2(h)               # (B_eff, d_win, n_frames)
        # Global mean-pool over the time axis
        embed = h.mean(dim=-1)           # (B_eff, d_win)
        return embed


# ---------------------------------------------------------------------------
# L2 — Match Sequence Transformer (mirrors HMLT v1 structure)
# ---------------------------------------------------------------------------

class _L2TransformerWithAttn(nn.Module):
    """Transformer encoder that can optionally return the last-layer attention.

    PyTorch's nn.TransformerEncoder does not expose per-layer attention weights
    by default.  We rebuild a thin wrapper that runs the layers manually so we
    can grab the last layer's attention matrix when requested.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,   # Pre-LN: more stable on small datasets
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """x: (B, T, d_model)  →  (h, attn_weights_or_None)."""
        h = x
        last_attn: Optional[Tensor] = None

        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1

            if is_last and return_attn:
                # Re-run the attention sub-layer manually to grab weights.
                # TransformerEncoderLayer applies: norm → self_attn → residual → ffn
                # We access the internal self_attn module directly.
                h_norm = layer.norm1(h)
                attn_out, last_attn = layer.self_attn(
                    h_norm,
                    h_norm,
                    h_norm,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,  # return (B, n_heads, T, T)
                )
                # Finish the layer manually: add residual, then FFN
                h = h + layer.dropout1(attn_out)
                h2 = layer.norm2(h)
                h = h + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(h2)))))
            else:
                h = layer(h, src_key_padding_mask=src_key_padding_mask)

        h = self.norm(h)
        return h, last_attn


# ---------------------------------------------------------------------------
# HMLTv2 — full 2-level model
# ---------------------------------------------------------------------------

class HMLTv2(nn.Module):
    """2-level Hierarchical Match-Level Transformer for binary win prediction.

    L1: FrameWindowEncoder (CNN) maps raw per-frame physics windows → latent.
    L2: Transformer over the sequence of window embeddings → match outcome.
    Hybrid: optionally concatenates 23-dim engineered match features to the
            L1 embedding, giving the model score/time priors it would otherwise
            learn slowly from physics alone.

    Args:
        n_frames:     Frames per window (45 default = 3 s @ 15 Hz).
        n_features:   Raw physics features per frame (133 universal default;
                      pass 88 only when loading old 2v2-only checkpoints).
        d_win:        L1 output dimension (window embedding size).
        n_match_feats: Dimension of hand-engineered match features (23).
        use_hybrid:   If True, concat match_feats to window_embed before L2.
        d_model:      L2 Transformer hidden dimension.
        n_layers:     Number of L2 TransformerEncoder layers.
        n_heads:      Attention heads in L2 (d_model must be divisible).
        dropout:      Dropout rate throughout the model.
        max_len:      Maximum sequence length for positional encoding.
        l1_kernel:    Conv kernel size for FrameWindowEncoder.
    """

    def __init__(
        self,
        n_frames: int = 45,
        n_features: int = 133,
        d_win: int = 128,
        n_match_feats: int = 23,
        use_hybrid: bool = True,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
        l1_kernel: int = 3,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.use_hybrid = use_hybrid
        self.n_match_feats = n_match_feats
        self.d_win = d_win

        # L1: shared frame-window encoder
        self.frame_encoder = FrameWindowEncoder(
            n_frames=n_frames,
            n_features=n_features,   # 133 universal by default
            d_win=d_win,
            kernel_size=l1_kernel,
        )

        # Fusion dimension: d_win [+ n_match_feats if hybrid]
        d_fused = d_win + (n_match_feats if use_hybrid else 0)

        # LayerNorm gate after fusion (stabilises gradient flow when concatenating
        # features of very different scales — physics vs. normalised aggregates)
        self.fusion_norm = nn.LayerNorm(d_fused)

        # L2: project fused embedding → d_model
        self.input_proj = nn.Linear(d_fused, d_model)

        # Sinusoidal positional encoding over window positions
        self.pos_enc = _SinusoidalPE(d_model=d_model, max_len=max_len, dropout=dropout)

        # L2 Transformer (custom wrapper so we can return last-layer attention)
        self.transformer = _L2TransformerWithAttn(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Classification head: masked-mean-pooled d_model → scalar logit
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_windows: Tensor,
        pad_mask: Tensor,
        match_feats: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            frame_windows: float32 (B, T_win, 45, 88) — raw physics per window.
            pad_mask:      bool    (B, T_win)           — True = padding window.
            match_feats:   float32 (B, T_win, 23) | None — required when use_hybrid=True.
            return_attn:   If True, also return last-layer L2 attention weights
                           of shape (B, n_heads, T_win, T_win).

        Returns:
            logit         float32 (B,) — raw logit; sigmoid → P(blue wins).
            [attn_weights float32 (B, n_heads, T_win, T_win)] — only if return_attn=True.
        """
        B, T_win, n_frames, n_features = frame_windows.shape

        # --- L1: encode each window independently ---
        # Flatten batch × windows → B_eff so the CNN sees one window per "sample"
        windows_flat = frame_windows.view(B * T_win, n_frames, n_features)  # (B*T, 45, 88)
        win_embed_flat = self.frame_encoder(windows_flat)                    # (B*T, d_win)
        win_embed = win_embed_flat.view(B, T_win, self.d_win)                # (B, T, d_win)

        # --- Hybrid fusion: concat engineered features ---
        if self.use_hybrid:
            if match_feats is None:
                raise ValueError(
                    "match_feats must be provided when use_hybrid=True. "
                    "Pass --no-hybrid or supply the 23-dim feature tensor."
                )
            fused = torch.cat([win_embed, match_feats], dim=-1)  # (B, T, d_win+23)
        else:
            fused = win_embed  # (B, T, d_win)

        fused = self.fusion_norm(fused)  # (B, T, d_fused)

        # --- L2: Transformer over window sequence ---
        h = self.input_proj(fused)   # (B, T, d_model)
        h = self.pos_enc(h)          # (B, T, d_model)

        h, attn_weights = self.transformer(
            h,
            src_key_padding_mask=pad_mask,
            return_attn=return_attn,
        )  # h: (B, T, d_model)

        # Masked mean-pool: average only over valid (non-pad) windows
        valid_mask = (~pad_mask).float().unsqueeze(-1)          # (B, T, 1)
        pooled = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        # pooled: (B, d_model)

        logit = self.head(pooled).squeeze(-1)  # (B,)

        if return_attn:
            return logit, attn_weights
        return logit


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
