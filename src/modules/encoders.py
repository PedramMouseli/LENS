import torch
import torch.nn as nn
import torch.nn.functional as F

class TDSBlock(nn.Module):
    """
    Time-depth separable block with depthwise temporal convolution and channel mixing.
    """
    def __init__(self, channels, kernel_size, stride=1, dropout=0.1):
        super(TDSBlock, self).__init__()
        self.channels = channels
        
        # Temporal Convolution Block
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=channels # This makes it a depthwise convolution
        )
        self.ln1 = nn.LayerNorm(channels, eps=1e-3)
        self.relu1 = nn.ReLU()
        
        # Feed-Forward Block (channel mixing)
        # Implemented as 1x1 Convolutions, which is equivalent to Linear layers
        # but works directly on (N, C, L) format.
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=1)
        )
        self.ln2 = nn.LayerNorm(channels, eps=1e-3)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # We need a residual projection if striding is used
        self.residual_proj = nn.Conv1d(channels, channels, 1, stride=stride) if stride > 1 else None

    def forward(self, x):
        """
        Args:
            x (Tensor): Input of shape `(Batch, Channels, Time)`.
        """
        # --- Temporal Convolution with its own residual ---
        identity = x
        out_conv = self.conv(x)
        
        # Keep the residual addition after normalization and activation to match
        # the ordering used in the emg2qwerty implementation.
        out_conv = self.ln1(out_conv.transpose(1, 2)).transpose(1, 2)
        out_conv = self.relu1(out_conv)

        if self.residual_proj:
            identity = self.residual_proj(identity)
        
        out = out_conv + identity

        # --- Feed-Forward with its own residual ---
        identity = out
        out_ffn = self.ffn(out)
        out = self.ln2((out + out_ffn).transpose(1, 2)).transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout(out)
        
        return out

class TDSConvNetEncoder(nn.Module):
    """
    A stack of TDSBlocks that acts as an encoder and downsampler.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size=9, dropout=0.1, downsample_strides=None):
        super(TDSConvNetEncoder, self).__init__()
        
        if downsample_strides is None:
            # Default to no downsampling if not specified
            downsample_strides = [1] * num_layers
        assert len(downsample_strides) == num_layers, "Length of strides must match number of layers"

        # Initial pointwise convolution to project input_dim to hidden_dim (channels)
        self.initial_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        layers = []
        for i in range(num_layers):
            layers.append(
                TDSBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=downsample_strides[i],
                    dropout=dropout
                )
            )
        self.tds_stack = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape `(Batch, Time, Features)`.
        
        Returns:
            Tensor: Encoded and downsampled tensor of shape `(Batch, NewTime, HiddenDim)`.
        """
        # (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.transpose(1, 2)
        
        out = self.initial_proj(x)
        out = self.tds_stack(out)
        
        # (Batch, Channels, NewTime) -> (Batch, NewTime, Channels)
        return out.transpose(1, 2)

def get_emg_encoder(input_dim, hidden_dim=128, num_layers=4, dropout=0.1):
    """Factory function for the EMG encoder."""
    return TDSConvNetEncoder(input_dim, hidden_dim, num_layers, kernel_size=9, dropout=dropout)

def get_nirs_encoder(input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
    """
    Factory function for the NIRS encoder.
    Typically smaller/shallower than the EMG encoder due to lower signal complexity.
    """
    return TDSConvNetEncoder(input_dim, hidden_dim, num_layers, kernel_size=5, dropout=dropout)