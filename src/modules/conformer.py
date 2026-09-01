import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class ConformerFeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_model)
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y


class ConformerConvModule(nn.Module):
    def __init__(self, dim_model: int, kernel_size: int, dropout: float):
        super().__init__()
        # Pointwise conv -> GLU
        self.layer_norm = nn.LayerNorm(dim_model)
        self.pw_conv1 = nn.Linear(dim_model, 2 * dim_model)
        self.glu = nn.GLU(dim=-1)
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            in_channels=dim_model, out_channels=dim_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim_model
        )
        # self.bn = nn.BatchNorm1d(dim_model)
        # Use token-wise LayerNorm after depthwise conv to avoid BN issues with padding
        self.ln_after_dw = nn.LayerNorm(dim_model)
        self.activation = nn.SiLU()
        # Pointwise conv back
        self.pw_conv2 = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, subsample_stride: int = 1) -> torch.Tensor:
        # x: (N, T, C)
        y = self.layer_norm(x)
        y = self.pw_conv1(y)
        y = self.glu(y)  # (N, T, C)
        # Depthwise conv expects (N, C, T)
        if subsample_stride and subsample_stride > 1:
            # Use stride in the depthwise conv for more efficient subsampling
            y = F.conv1d(
                y.transpose(1, 2),
                weight=self.dw_conv.weight,
                bias=self.dw_conv.bias,
                stride=subsample_stride,
                padding=self.dw_conv.padding[0],
                dilation=self.dw_conv.dilation[0],
                groups=self.dw_conv.groups
            ).transpose(1, 2)  # (N, T/stride, C)
        else:
            y = self.dw_conv(y.transpose(1, 2)).transpose(1, 2)  # (N, T, C)
        y = self.ln_after_dw(y)
        y = self.activation(y)
        # y = self.dw_conv(y.transpose(1, 2))
        # y = self.bn(y)
        # y = self.activation(y).transpose(1, 2)
        y = self.pw_conv2(y)
        y = self.dropout(y)
        return y


class ConformerBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, conv_kernel_size: int, dropout: float):
        super().__init__()
        # FF modules (scaled residuals)
        self.ff1 = ConformerFeedForward(dim_model, dim_ff, dropout)
        self.ff2 = ConformerFeedForward(dim_model, dim_ff, dropout)
        # MHSA
        self.mha_norm = nn.LayerNorm(dim_model)

        self.mha = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_dropout = nn.Dropout(dropout)
        # Conv module
        self.conv_module = ConformerConvModule(dim_model, conv_kernel_size, dropout)
        # Final norm
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> tuple:
        # FF1 (0.5 scaling)
        x = x + 0.5 * self.ff1(x)
        # MHSA
        y = self.mha_norm(x)
        # Merge key_padding_mask into attn_mask for nn.MultiheadAttention to avoid dtype mismatch warning
        if isinstance(self.mha, nn.MultiheadAttention):
            B, T = y.shape[0], y.shape[1]
            merged_mask = attn_mask
            if src_key_padding_mask is not None:
                kpm_bool = src_key_padding_mask if src_key_padding_mask.dtype is torch.bool else (src_key_padding_mask != 0)
                # Build additive mask (B, T, T): -inf where key token is padded
                mask_dtype = merged_mask.dtype if (merged_mask is not None and merged_mask.dtype.is_floating_point) else y.dtype
                kpm_add = torch.zeros(B, T, T, device=y.device, dtype=mask_dtype)
                kpm_add = kpm_add.masked_fill(kpm_bool[:, None, :], float('-inf'))
                if merged_mask is None:
                    merged_mask = kpm_add
                else:
                    # Align shapes: support (T,T), (B,T,T) or (B*H,T,T)
                    if merged_mask.dim() == 2 and merged_mask.shape == (T, T):
                        merged_mask = merged_mask + kpm_add.mean(dim=0)  # broadcast average over batch
                    elif merged_mask.dim() == 3 and merged_mask.shape[0] == B:
                        merged_mask = merged_mask + kpm_add
                    elif merged_mask.dim() == 3 and merged_mask.shape[0] == B * self.mha.num_heads:
                        # expand kpm_add over heads
                        merged_mask = merged_mask + kpm_add.repeat_interleave(self.mha.num_heads, dim=0)
                    else:
                        # fallback: try to broadcast
                        merged_mask = merged_mask + kpm_add
            # Replace -inf with large negative to avoid NaNs if a row gets fully masked
            if merged_mask is not None:
                neg_large = torch.full_like(merged_mask, -1e9)
                merged_mask = torch.where(torch.isneginf(merged_mask), neg_large, merged_mask)
            attn_out, _ = self.mha(y, y, y, key_padding_mask=None, attn_mask=merged_mask, need_weights=False)
        else:
            attn_out, _ = self.mha(y, y, y, key_padding_mask=src_key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = x + self.mha_dropout(attn_out)

        x = x + self.conv_module(x)
        # FF2 (0.5 scaling)
        x = x + 0.5 * self.ff2(x)
        # Final norm
        x = self.final_norm(x)
        return x, src_key_padding_mask


class ConformerEncoder(nn.Module):
    """
    Conformer-style encoder:
      - input: (batch, time, features)
      - optional src_key_padding_mask: (batch, time) with True = pad
      - output: (batch, time, features)
    """
    def __init__(self, input_dim: int, num_heads: int, num_layers: int, ffn_dim: int, conv_kernel_size: int, dropout: float,
                 attention_causal: bool = False, attention_window_size: int = 0,
                 relative_position_type: str = 't5', rpb_num_buckets: int = 32, rpb_max_distance: int = 128,
                 attention_window_schedule: Optional[list] = None, global_heads: int = 0,
                 rpb_scale: float = 0.0, rpb_per_head: bool = True, attn_mask_mode: str = '3d'):
        super().__init__()
        self.num_heads = num_heads
        self.attention_causal = bool(attention_causal)
        self.window_size = int(attention_window_size) if attention_window_size is not None else 0
        self.use_rpb = (relative_position_type == 't5')
        self.rpb_num_buckets = int(rpb_num_buckets)
        self.rpb_max_distance = int(rpb_max_distance)
        # Optional per-layer window sizes; if not provided, use constant window_size for all layers
        self.window_schedule = attention_window_schedule if attention_window_schedule is not None else []
        self.global_heads = max(0, int(global_heads))
        # RPE controls
        self.rpb_scale = float(rpb_scale)
        self.rpb_per_head = bool(rpb_per_head)
        self.attn_mask_mode = '3d' if str(attn_mask_mode).lower() not in ('2d', '3d') else str(attn_mask_mode).lower()
        if self.use_rpb:
            if self.rpb_per_head:
                self.relative_attention_bias = nn.Embedding(self.rpb_num_buckets, self.num_heads)
            else:
                self.relative_attention_bias = nn.Embedding(self.rpb_num_buckets, 1)
            nn.init.zeros_(self.relative_attention_bias.weight)

        self.layers = nn.ModuleList([])
        for li in range(num_layers):
            self.layers.append(
                ConformerBlock(
                    dim_model=input_dim,
                    num_heads=num_heads,
                    dim_ff=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
            )

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        T5-style relative position bucketing.
        relative_position: (L, S) with values (j - i)
        """
        num_buckets = self.rpb_num_buckets
        max_distance = self.rpb_max_distance
        # Split buckets for negative and positive
        n = relative_position
        sign = (n < 0).to(torch.long)
        n = n.abs()
        # Half the buckets for each sign
        half_buckets = num_buckets // 2
        # Exact buckets for small positions
        small_threshold = 8
        is_small = n < small_threshold
        val_small = n
        # Log-scaled buckets for large positions
        # Map [small_threshold, inf) to [0, half_buckets - small_threshold)
        max_exact = small_threshold
        num_log_buckets = half_buckets - max_exact
        n_clamped = torch.clamp(n, min=1)
        log_scale = (torch.log(n_clamped.float() / max_exact + 1e-6) /
                     torch.log(torch.tensor(max_distance / max_exact + 1.0, device=n.device)))
        val_large = max_exact + (log_scale * num_log_buckets).to(torch.long)
        val_large = torch.clamp(val_large, max=half_buckets - 1)
        bucket = torch.where(is_small, val_small, val_large)
        # Offset by sign: negatives in first half (0..half_buckets-1), positives in second half
        bucket = bucket + sign * half_buckets
        return bucket

    def _build_attn_mask(self, T: int, B: int, device: torch.device, window_size: int, special_unmask_indices: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Build an additive attention mask:
          - shape (T, T) when using shared bias (not used now)
          - shape (B * num_heads, T, T) when using per-head bias (current)
        Combines:
          - -inf where attention is disallowed (causal and/or window)
          - +relative bias values per head
        """
        if not self.attention_causal and (window_size is None or window_size <= 0) and not self.use_rpb and self.global_heads <= 0 and (special_unmask_indices is None or special_unmask_indices.numel() == 0):
            return None
        base = torch.zeros(T, T, device=device, dtype=torch.float32)
        # causal upper-triangular mask
        if self.attention_causal:
            base = base.masked_fill(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1), float('-inf'))
        # local window
        if window_size and window_size > 0:
            idx = torch.arange(T, device=device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)  # (T, T): i - j
            if self.attention_causal:
                allow = (dist >= 0) & (dist <= window_size)
            else:
                allow = (dist.abs() <= window_size)
            base = base.masked_fill(~allow, float('-inf'))
        # Build causal-only base for global heads (ignore windowing)
        base_causal = torch.zeros_like(base)
        if self.attention_causal:
            base_causal = base_causal.masked_fill(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1), float('-inf'))
        # Expand masks according to mode
        if self.attn_mask_mode == '3d':
            base_exp = base.unsqueeze(0).expand(B * self.num_heads, -1, -1).clone()
            base_causal_exp = base_causal.unsqueeze(0).expand(B * self.num_heads, -1, -1)
        else:
            base_exp = base
            base_causal_exp = base_causal
        # relative position bias
        if self.use_rpb:
            rel = torch.arange(T, device=device).unsqueeze(1) - torch.arange(T, device=device).unsqueeze(0)
            buckets = self._relative_position_bucket(rel)
            if self.rpb_per_head:
                bias = self.relative_attention_bias(buckets)  # (T, T, num_heads)
                if self.attn_mask_mode == '3d':
                    bias = bias.permute(2, 0, 1).contiguous()  # (H, T, T)
                    bias = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_heads, T, T)
                    bias = self.rpb_scale * bias
                    bias = bias.masked_fill(base_exp == float('-inf'), 0.0)
                    attn = base_exp + bias
                else:
                    bias2d = bias.mean(dim=2)  # (T, T)
                    bias2d = self.rpb_scale * bias2d
                    bias2d = bias2d.masked_fill(base_exp == float('-inf'), 0.0)
                    attn = base_exp + bias2d
            else:
                bias = self.relative_attention_bias(buckets).squeeze(-1)  # (T, T)
                bias = self.rpb_scale * bias
                bias = bias.masked_fill(base_exp == float('-inf'), 0.0)
                attn = base_exp + bias
        else:
            attn = base_exp
        # Global heads (only applicable in 3D mode): replace masks of first K heads with causal-only (ignore window)
        K = min(self.global_heads, self.num_heads)
        if K > 0 and self.attn_mask_mode == '3d':
            for b in range(B):
                start = b * self.num_heads
                idx = torch.arange(start, start + K, device=device)
                if self.use_rpb:
                    attn[idx] = attn[idx].masked_fill(base_exp[idx] == float('-inf'), 0.0)
                    attn[idx] = attn[idx] + (base_causal_exp[idx] - base_causal_exp[idx].masked_fill(base_causal_exp[idx] != float('-inf'), 0.0))
                    attn[idx] = attn[idx].masked_fill(base_causal_exp[idx] == float('-inf'), float('-inf'))
                else:
                    attn[idx] = base_causal_exp[idx]
        # Global CLS and special indices: allow specified indices to attend to and be attended by all tokens
        indices = []
        if special_unmask_indices is not None and special_unmask_indices.numel() > 0:
            indices.extend(special_unmask_indices.tolist())
        if indices:
            idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            if self.attn_mask_mode == '3d':
                attn[:, idx_tensor, :] = attn[:, idx_tensor, :].masked_fill(attn[:, idx_tensor, :] == float('-inf'), 0.0)
                attn[:, :, idx_tensor] = attn[:, :, idx_tensor].masked_fill(attn[:, :, idx_tensor] == float('-inf'), 0.0)
            else:
                attn[idx_tensor, :] = attn[idx_tensor, :].masked_fill(attn[idx_tensor, :] == float('-inf'), 0.0)
                attn[:, idx_tensor] = attn[:, idx_tensor].masked_fill(attn[:, idx_tensor] == float('-inf'), 0.0)
        return attn

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]

        inserted_indices = None
        T_ext = T
        # Per-layer window growth schedule
        for idx, layer in enumerate(self.layers):
            if self.window_schedule and idx < len(self.window_schedule):
                w = int(self.window_schedule[idx])
            else:
                w = self.window_size
            attn_mask = self._build_attn_mask(T_ext, B, x.device, w, special_unmask_indices=inserted_indices)
            x, src_key_padding_mask = layer(x, src_key_padding_mask=src_key_padding_mask, attn_mask=attn_mask)
            # Update current length after potential subsampling
            T_ext = x.shape[1]

        return x
