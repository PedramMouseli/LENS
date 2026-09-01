import torch

def perform_masking(features, mask_prob=0.15, mask_len=10):
    """
    Applies masking to the input feature sequence for SSP.
    
    Args:
        features (Tensor): Shape `(Batch, Time, Features)`.
        mask_prob (float): Probability of a token being the start of a mask.
        mask_len (int): The length of each mask span.
    
    Returns:
        tuple: (masked_features, mask)
               masked_features: The input with masks applied.
               mask: A boolean tensor where True indicates a masked position.
    """
    batch_size, seq_len, _ = features.shape
    device = features.device
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    effective_len = max(1, min(mask_len, seq_len))
    # ensure at least one span when seq_len > 0
    num_masks = max(1 if seq_len > 0 else 0, int(seq_len * mask_prob / effective_len))
    max_start = max(1, seq_len - effective_len + 1)

    for i in range(batch_size):
        if num_masks > 0:
            starts = torch.randint(0, max_start, (num_masks,), device=device)
            for s in starts.tolist():
                mask[i, s : s + effective_len] = True

    masked = features.clone()
    masked[mask] = 0
    return masked, mask