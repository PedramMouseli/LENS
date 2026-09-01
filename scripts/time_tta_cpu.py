#!/usr/bin/env python
"""
Time TTA (Test-Time Adaptation) on a single CPU core for one subject.

Reports wall-clock time for:
  - 1  TTA epoch   (single subject)
  - 40 TTA epochs  (single subject)

Usage:
    python scripts/time_tta_cpu.py \
        --weights runs/final_experiment_controls/supervised_model.pt

Uses the neuro_env conda environment (PyTorch on CPU).
"""

import os
import sys
import time
import glob
import hashlib
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Restrict to a single CPU core for reproducible timing
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import MultimodalModel
from src.config import get_config


# ---------------------------------------------------------------------------
# Minimal SSP adaptation dataset (mirrors _SSPAdaptFixedChunkDataset in train.py)
# ---------------------------------------------------------------------------
class SSPAdaptDataset(Dataset):
    """Fixed-chunk dataset for SSP adaptation on a single subject."""
    def __init__(self, feature_path, chunk_len, start_indices):
        feats = torch.load(feature_path, map_location='cpu')
        self.emg = feats['emg']
        self.nirs = feats['nirs']
        self.chunk_len = int(chunk_len)
        self.starts = list(start_indices)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        T = self.emg.shape[0]
        L = self.chunk_len
        s = int(self.starts[idx])
        if T <= L:
            s = 0
        else:
            s = max(0, min(s, T - L))
        e = min(T, s + L)
        return self.emg[s:e, :], self.nirs[s:e, :]


# ---------------------------------------------------------------------------
# TTA adaptation loop (extracted from adapt_and_evaluate in train.py)
# ---------------------------------------------------------------------------
def run_tta_epochs(model_state_dict, config, subject_meta, n_epochs, device):
    """
    Run TTA for `n_epochs` on a single subject and return elapsed wall-clock seconds.
    """
    # 1. Build chunk indices (same logic as train.py)
    feats_tmp = torch.load(subject_meta['feature_path'], map_location='cpu')
    T_full = int(feats_tmp['emg'].shape[0])
    L = int(config['ssp_context_len'])
    max_start = max(0, T_full - L)
    num_train_chunks = int(config.get('adaptation_chunks_per_epoch', 256))

    sid = str(subject_meta.get('id', 'unknown'))
    seed_bytes = hashlib.sha256(sid.encode('utf-8')).digest()
    seed_int = int.from_bytes(seed_bytes[:8], byteorder='little', signed=False) % (2**31 - 1)
    g = torch.Generator()
    g.manual_seed(seed_int)

    if max_start <= 0:
        starts_train = [0] * num_train_chunks
    else:
        candidate_starts = list(range(0, max_start + 1, max(1, L)))
        if len(candidate_starts) < 2:
            starts_train = [0] * num_train_chunks
        else:
            perm = torch.randperm(len(candidate_starts), generator=g).tolist()
            candidate_starts = [candidate_starts[i] for i in perm]
            n_train_unique = min(num_train_chunks, len(candidate_starts))
            train_unique = candidate_starts[:n_train_unique]

            def _fill_to_n(pool, n_needed):
                if n_needed <= len(pool):
                    return pool[:n_needed]
                if len(pool) == 0:
                    return [0] * n_needed
                extra_g = torch.Generator().manual_seed(seed_int + 1)
                extra = torch.randint(low=0, high=len(pool),
                                      size=(n_needed - len(pool),),
                                      generator=extra_g).tolist()
                return pool + [pool[i] for i in extra]

            starts_train = _fill_to_n(train_unique, num_train_chunks)

    dataset = SSPAdaptDataset(subject_meta['feature_path'], L, starts_train)
    loader = DataLoader(dataset,
                        batch_size=config['adaptation_batch_size'],
                        shuffle=True)

    # 2. Instantiate adapted model on CPU
    adapted_model = MultimodalModel(config).to(device)
    # Remap checkpoint keys if the model was saved with a different attribute name
    # (e.g., 'fusion' in the checkpoint vs 'projection' in the current model)
    remapped_sd = {}
    for k, v in model_state_dict.items():
        new_k = k.replace('fusion.', 'projection.') if k.startswith('fusion.') else k
        remapped_sd[new_k] = v
    adapted_model.load_state_dict(remapped_sd, strict=False)

    # 3. Freeze / unfreeze (same as train.py)
    for param in adapted_model.parameters():
        param.requires_grad = False
    for param in adapted_model.emg_encoder.parameters():
        param.requires_grad = True
    for param in adapted_model.nirs_encoder.parameters():
        param.requires_grad = False
    for param in adapted_model.nirs_norm.parameters():
        param.requires_grad = False
    for param in adapted_model.emg_norm.parameters():
        param.requires_grad = True
    for param in adapted_model.local_ssp_head_emg.parameters():
        param.requires_grad = True
    for param in adapted_model.local_ssp_head_nirs.parameters():
        param.requires_grad = True
    for param in adapted_model.global_ssp_head.parameters():
        param.requires_grad = False
    for param in adapted_model.projection.parameters():
        param.requires_grad = True
    if config.get('ssp_use_decoder', False) and hasattr(adapted_model, 'ssp_decoder'):
        for param in adapted_model.ssp_decoder.parameters():
            param.requires_grad = True

    # 4. Optimizer + scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, adapted_model.parameters()),
        lr=config['adaptation_lr']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs))

    emg_dim = config['emg_feature_dim']
    cos_alpha_nirs = float(config.get('ssp_cosine_alpha_nirs', 0.0) or 0.0)

    # 5. Run adaptation and time it
    adapted_model.train()
    torch.manual_seed(42)  # reproducible

    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        for emg_chunk, nirs_chunk in loader:
            emg_chunk = emg_chunk.to(device)
            nirs_chunk = nirs_chunk.to(device)
            optimizer.zero_grad()

            outputs_local = adapted_model(emg_chunk, nirs_chunk, mode='ssp')
            (local_preds, local_targets, emg_mask_ds, nirs_mask_ds,
             emg_mask_hr, nirs_mask_hr, emg_aug, nirs_aug) = outputs_local

            local_preds_emg = local_preds[..., :emg_dim]
            local_preds_nirs = local_preds[..., emg_dim:]
            local_targets_emg = local_targets[..., :emg_dim]
            local_targets_nirs = local_targets[..., emg_dim:]

            loss_local_emg = F.l1_loss(local_preds_emg[emg_mask_ds],
                                       local_targets_emg[emg_mask_ds])
            loss_local_nirs = F.l1_loss(local_preds_nirs[nirs_mask_ds],
                                        local_targets_nirs[nirs_mask_ds])
            loss_local = loss_local_emg + 0.1 * loss_local_nirs

            loss_cos = torch.tensor(0.0, device=device)
            if (cos_alpha_nirs > 0.0) and bool(nirs_mask_ds.any()):
                pn = local_preds_nirs[nirs_mask_ds]
                tn = local_targets_nirs[nirs_mask_ds]
                loss_cos = (1.0 - F.cosine_similarity(pn, tn, dim=-1)).mean() * cos_alpha_nirs

            loss = loss_local + loss_cos
            loss.backward()
            optimizer.step()

        scheduler.step()

    t_end = time.perf_counter()
    return t_end - t_start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Time TTA on a single CPU core.")
    parser.add_argument('--weights', type=str,
                        default='runs/final_experiment_controls/supervised_model.pt',
                        help='Path to supervised model weights (.pt)')
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID to use (default: first available)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup epochs (not timed) to prime caches')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of timed repetitions for averaging')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Torch threads: {torch.get_num_threads()} (interop: {torch.get_num_interop_threads()})")
    print()

    # Load config
    config = get_config()

    # Find one test subject
    project_root = os.path.join(os.path.dirname(__file__), '..')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    label_files = sorted(glob.glob(os.path.join(processed_dir, '*_labels.pt')))

    subject_meta = None
    for lf in label_files:
        sid = os.path.basename(lf).replace('_labels.pt', '')
        if args.subject and sid != args.subject:
            continue
        fp = os.path.join(processed_dir, f'{sid}_features_logspec.pt')
        if os.path.exists(fp):
            subject_meta = {'id': sid, 'feature_path': fp, 'label_path': lf}
            break

    if subject_meta is None:
        print("ERROR: No matching subject found.")
        sys.exit(1)

    print(f"Subject: {subject_meta['id']}")

    # Check subject data size
    feats = torch.load(subject_meta['feature_path'], map_location='cpu')
    T = feats['emg'].shape[0]
    print(f"  EMG shape:  {feats['emg'].shape}  ({T/50:.0f}s at 50 Hz)")
    print(f"  NIRS shape: {feats['nirs'].shape}")
    del feats
    print()

    # Load model weights
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    print(f"  Model parameters: {sum(p.numel() for p in MultimodalModel(config).parameters()):,}")
    print()

    # Warmup run (primes JIT, memory allocators, etc.)
    if args.warmup > 0:
        print(f"Warmup: {args.warmup} epoch(s)...")
        _ = run_tta_epochs(state_dict, config, subject_meta, args.warmup, device)
        print("  Warmup complete.\n")

    # --- Time 1 TTA epoch ---
    print(f"Timing 1 TTA epoch ({args.repeats} repeats)...")
    times_1 = []
    for r in range(args.repeats):
        t = run_tta_epochs(state_dict, config, subject_meta, 1, device)
        times_1.append(t)
        print(f"  Run {r+1}: {t:.3f}s")

    mean_1 = sum(times_1) / len(times_1)
    std_1 = (sum((t - mean_1)**2 for t in times_1) / len(times_1)) ** 0.5
    print(f"  → 1 epoch:  {mean_1:.3f} ± {std_1:.3f}s")
    print()

    # --- Time 40 TTA epochs ---
    print(f"Timing 40 TTA epochs ({args.repeats} repeats)...")
    times_40 = []
    for r in range(args.repeats):
        t = run_tta_epochs(state_dict, config, subject_meta, 40, device)
        times_40.append(t)
        print(f"  Run {r+1}: {t:.3f}s")

    mean_40 = sum(times_40) / len(times_40)
    std_40 = (sum((t - mean_40)**2 for t in times_40) / len(times_40)) ** 0.5
    print(f"  → 40 epochs: {mean_40:.3f} ± {std_40:.3f}s")
    print()

    # Summary
    print("=" * 55)
    print("        TTA TIMING SUMMARY  (Apple M2 Ultra, 1 CPU core)")
    print("=" * 55)
    print(f"  Subject:            {subject_meta['id']}")
    print(f"  Sequence length:    {T} steps ({T/50:.0f}s)")
    print(f"  Chunks per epoch:   {config.get('adaptation_chunks_per_epoch', 16)}")
    print(f"  Batch size:         {config['adaptation_batch_size']}")
    print(f"  Context length:     {config['ssp_context_len']} steps ({config['ssp_context_len']/50:.0f}s)")
    print(f"  Repeats:            {args.repeats}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  1  TTA epoch:       {mean_1:.3f} ± {std_1:.3f}s")
    print(f"  40 TTA epochs:      {mean_40:.3f} ± {std_40:.3f}s")
    print(f"  Per-epoch avg:      {mean_40/40:.3f}s")
    print("=" * 55)


if __name__ == '__main__':
    main()
