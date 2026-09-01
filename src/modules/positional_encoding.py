# positional_encoding.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, from the "Attention Is All You Need" paper.
    This injects absolute position information into the token sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Create the positional encoding matrix directly in 2D shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register the encoding with a leading batch dimension for broadcasted addition.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        output = x + self.pe[:, :x.size(1)]
        return self.dropout(output)