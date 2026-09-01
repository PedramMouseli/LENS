import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder_projection(nn.Module):
    """
    Project encoder outputs to a shared hidden dimension.
    """

    def __init__(self, emg_dim, nirs_dim, hidden_dim):
        # Using super().__init__() avoids issues in notebooks when modules get reloaded
        # (super(Class, self) can break if Class is re-imported).
        super().__init__()
        self.emg_proj = nn.Linear(emg_dim, hidden_dim)
        self.nirs_proj = nn.Linear(nirs_dim, hidden_dim)
        self.last_attention_weights = None  # for logging/analysis

        # Reserved for a future learned fusion path.
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

        # Reserved for a future post-fusion projection path.
        self.fused_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, emg_seq, nirs_seq):
        """
        Args:
            emg_seq (Tensor): (Batch, Time, EmgDim)
            nirs_seq (Tensor): (Batch, Time, NirsDim)

        Returns:
            Tensor: Projected sequence of shape (Batch, Time, HiddenDim).
        """
        # The current forward path projects the EMG sequence only.
        emg_proj = self.emg_proj(emg_seq)

        fused_representation = nn.SiLU()(emg_proj)

        return fused_representation