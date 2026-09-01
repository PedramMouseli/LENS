import torch
import torch.nn as nn
import numpy as np

class SpecAugment(nn.Module):
    """
    Spectrogram augmentation module.
    Applies frequency and time masking to the input spectrogram.
    Inspired by Park et al. (SpecAugment).
    """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=2, num_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape `(Batch, Time, Features)`.
        
        Returns:
            Tensor: Augmented tensor.
        """
        if not self.training:
            return x

        clone = x.clone()
        num_freq_bins = clone.shape[2]
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            f = np.random.uniform(low=0.0, high=self.freq_mask_param)
            f_zero = int(f)
            f0 = np.random.randint(0, num_freq_bins - f_zero)
            clone[:, :, f0:f0 + f_zero] = 0

        # Apply time masks
        num_time_steps = clone.shape[1]
        for _ in range(self.num_time_masks):
            t = np.random.uniform(low=0.0, high=self.time_mask_param)
            t_zero = int(t)
            if t_zero < num_time_steps:
                t0 = np.random.randint(0, num_time_steps - t_zero)
                clone[:, t0:t0 + t_zero, :] = 0
                
        return clone

class Jitter(nn.Module):
    """
    Applies amplitude jitter by adding Gaussian noise.
    """
    def __init__(self, sigma=0.05):
        super(Jitter, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if not self.training or self.sigma == 0:
            return x
        
        noise = torch.randn_like(x) * self.sigma
        return x + noise