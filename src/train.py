# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# --- Suppress TensorFlow/oneDNN warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations

# --- Path Helper ---
# This allows the script to be run from anywhere and still find the project's modules.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import math
import copy
import signal
import hashlib

# --- Import custom modules using absolute paths from the project root ---
from src.model import MultimodalModel
from src.data_loader import PainDataset, collate_fn_supervised
from src.config import get_config
from src.utils import (setup_ddp, cleanup_ddp, TensorboardLogger, EarlyStopping,
                   calculate_regression_metrics, 
                   print_metrics, set_seed, compute_label_transform_stats, inverse_transform_labels,
                   beta_nll_from_params, gamma_nll_from_params, mix2_beta_nll_from_params, mix2_gamma_nll_from_params,
                   params_to_means)

def ssp_epoch(rank, epoch, model, dataloader, optimizer, device, config, is_train=True):
    """Handles a single epoch for Hierarchical SSP (training or validation)."""
    if is_train:
        model.train()
        desc = f"SSP Train Epoch {epoch+1}"
    else:
        model.eval()
        desc = f"SSP Val Epoch {epoch+1}"

    if isinstance(dataloader.sampler, DistributedSampler) and is_train:
        dataloader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    total_loss_emg = 0.0
    total_loss_nirs = 0.0
    total_loss_cos_emg = 0.0
    total_loss_cos_nirs = 0.0
    total_loss_cons = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=desc, disable=(rank != 0))
    context = torch.enable_grad() if is_train else torch.no_grad()

    emg_dim = config['emg_feature_dim']

    with context:
        for emg_chunk, nirs_chunk in pbar:
            emg_chunk, nirs_chunk = emg_chunk.to(device), nirs_chunk.to(device)
            if is_train:
                optimizer.zero_grad()



            # The forward pass now takes the UNMASKED data and handles all logic internally
            outputs_local = model(emg_chunk, nirs_chunk, mode='ssp')
            # Backward/forward compatibility: model may optionally return fused latent at the end.
            if isinstance(outputs_local, (tuple, list)) and len(outputs_local) >= 9:
                (local_preds, local_targets, emg_mask_ds, nirs_mask_ds,
                 emg_mask_hr, nirs_mask_hr, emg_aug, nirs_aug, fused_latent_1) = outputs_local[:9]
            else:
                (local_preds, local_targets, emg_mask_ds, nirs_mask_ds,
                 emg_mask_hr, nirs_mask_hr, emg_aug, nirs_aug) = outputs_local
                fused_latent_1 = None
            
            # --- Local Loss Calculation ---
            local_preds_emg = local_preds[..., :emg_dim]
            local_preds_nirs = local_preds[..., emg_dim:]
            local_targets_emg = local_targets[..., :emg_dim]
            local_targets_nirs = local_targets[..., emg_dim:]


            loss_local_emg = nn.SmoothL1Loss()(local_preds_emg[emg_mask_ds], local_targets_emg[emg_mask_ds])
            loss_local_nirs = nn.SmoothL1Loss()(local_preds_nirs[nirs_mask_ds], local_targets_nirs[nirs_mask_ds])

            loss_local = loss_local_emg + 0.1 * loss_local_nirs

            # --- Optional cosine similarity term on masked positions (directional alignment) ---
            cos_alpha_nirs = float(config.get('ssp_cosine_alpha_nirs', 0.0) or 0.0)
            loss_cos = torch.tensor(0.0, device=device)

            loss_cos_nirs = torch.tensor(0.0, device=device)

            if (cos_alpha_nirs > 0.0) and bool(nirs_mask_ds.any()):
                pn = local_preds_nirs[nirs_mask_ds]
                tn = local_targets_nirs[nirs_mask_ds]
                loss_cos_nirs = (1.0 - F.cosine_similarity(pn, tn, dim=-1)).mean() * cos_alpha_nirs
                loss_cos = loss_cos + loss_cos_nirs

            loss = loss_local + loss_cos
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=10.0) # A max_norm of 1.0 to 10.0 is common
                optimizer.step()
            
            total_loss += loss.item() * emg_chunk.size(0)
            total_loss_emg += loss_local_emg.item()
            total_loss_nirs += loss_local_nirs.item()
            total_loss_cos_nirs += loss_cos_nirs.item()
            num_batches += 1
    
    total_loss_tensor = torch.tensor(total_loss).to(device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    # Correctly average loss over the entire dataset across all GPUs
    if hasattr(dataloader, "dataset"):
        dataset_size = len(dataloader.dataset)
    else:
        dataset_size = len(dataloader.sampler.data_source)
    avg_loss = total_loss_tensor.item() / dataset_size if dataset_size > 0 else 0.0

    # On rank 0, stash per-modality SSP losses and (for attention fusion) mean weights into config.
    # IMPORTANT: we run ssp_epoch() for both train and val each epoch; keep them separate so
    # TensorBoard doesn't accidentally log val stats under train tags.
    if rank == 0 and num_batches > 0:
        prefix = "train" if is_train else "val"
        config[f'ssp_last_{prefix}_loss_local_emg'] = total_loss_emg / num_batches
        config[f'ssp_last_{prefix}_loss_local_nirs'] = total_loss_nirs / num_batches
        config[f'ssp_last_{prefix}_loss_cosine_emg'] = total_loss_cos_emg / num_batches
        config[f'ssp_last_{prefix}_loss_cosine_nirs'] = total_loss_cos_nirs / num_batches
        config[f'ssp_last_{prefix}_loss_consistency'] = total_loss_cons / num_batches
        fusion_type = str(config.get('fusion_params', {}).get('type', 'attention')).lower()
        if fusion_type == 'attention':
            fusion_mod = getattr(model.module, 'fusion', None) if hasattr(model, 'module') else getattr(model, 'fusion', None)
            attn_w = getattr(fusion_mod, 'last_attention_weights', None) if fusion_mod is not None else None
            if attn_w is not None:
                config[f'ssp_last_{prefix}_attn_w_emg'] = attn_w[..., 0].mean().item()
                config[f'ssp_last_{prefix}_attn_w_nirs'] = attn_w[..., 1].mean().item()

    return avg_loss


def supervised_train_epoch(rank, epoch, model, dataloader, optimizer, device, config, logger=None):
    """
    Handles a training epoch for the final supervised task (Phase 3).
    """
    model.train()
    desc = f"Supervised Fine-tune Epoch {epoch+1}"
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    
    total_loss_epoch = 0
    pbar = tqdm(dataloader, desc=desc, disable=(rank != 0))

    # Track param means for distributional heads
    reg_type_global = config.get('regression_loss', 'smooth_l1')
    param_sums = {}
    num_samples = 0
    for batch in pbar:
        emg_full, nirs_full, phases_full = batch['emg'].to(device), batch['nirs'].to(device), batch['task_phase_ids'].to(device)

        main_labels = batch['post_mvc_pain'].to(device)

        optimizer.zero_grad()

        # --- Build inputs: either concatenated windows or full sequence ---
        fs = config['target_sample_rate']
        win_len = int(config.get('trial_window_seconds', 60) * fs)
        sep_len = int(config.get('window_separator_seconds', 0) * fs)
        num_trials_before = config.get('num_trials_before_mvc_for_final', 3)
        mvc_post = int(config.get('mvc_post_seconds', 120) * fs)

        concat_emg_list, concat_nirs_list, concat_phases_list, lengths_list = [], [], [], []
        for i in range(emg_full.size(0)):
            L = (phases_full[i] != -1).sum().item() if 'lengths' in batch else emg_full.shape[1] # fallback
            phases_i = phases_full[i]
            # Find MVC start/end (phase == 2)
            mask_mvc = (phases_i[:L] == 2)
            if mask_mvc.any():
                mvc_start = torch.where(mask_mvc)[0][0].item()
                mvc_end = torch.where(mask_mvc)[0][-1].item()
                mvc_window_start = mvc_start
                mvc_window_end = min(L, mvc_end + mvc_post)
            else:
                mvc_window_start, mvc_window_end = None, None

            # Trials indices (assumed available): use batch['intermediate_pain_indices']
            idxs = batch['intermediate_pain_indices'][i]
            if not config.get('use_trial_windowing', True):
                # Use whole sequence: from start (or a reasonable start) to min(mvc_end+post, end)
                s = 0
                e = emg_full.shape[1]
                if mvc_window_start is not None:
                    e = min(e, mvc_window_end)
                concat_emg_list.append(emg_full[i, s:e, :])
                concat_nirs_list.append(nirs_full[i, s:e, :])
                concat_phases_list.append(phases_full[i, s:e])
                lengths_list.append(e - s)
            else:
                # Windowed last N trials + MVC
                valid_trials = [t for t in range(len(idxs)) if idxs[t].item() > 0]
                if mvc_window_start is not None:
                    valid_trials = [t for t in valid_trials if idxs[t].item() <= mvc_window_start]
                chosen = valid_trials[-num_trials_before:] if len(valid_trials) > 0 else []

                windows_emg, windows_nirs, windows_phases = [], [], []
                total_len = 0
                for idx, t in enumerate(chosen):
                    idx_t = idxs[t].item()
                    s = max(0, idx_t - win_len)
                    e = idx_t
                    windows_emg.append(emg_full[i, s:e, :])
                    windows_nirs.append(nirs_full[i, s:e, :])
                    windows_phases.append(phases_full[i, s:e])
                    total_len += (e - s)
                    if sep_len > 0:
                        windows_emg.append(torch.zeros(sep_len, emg_full.shape[-1], device=device))
                        windows_nirs.append(torch.zeros(sep_len, nirs_full.shape[-1], device=device))
                        windows_phases.append(torch.zeros(sep_len, dtype=phases_full.dtype, device=device))
                        total_len += sep_len
                if mvc_window_start is not None and mvc_window_end > mvc_window_start:
                    windows_emg.append(emg_full[i, mvc_window_start:mvc_window_end, :])
                    windows_nirs.append(nirs_full[i, mvc_window_start:mvc_window_end, :])
                    windows_phases.append(phases_full[i, mvc_window_start:mvc_window_end])
                    total_len += (mvc_window_end - mvc_window_start)

                if total_len == 0:
                    tail = min(win_len, emg_full.shape[1])
                    windows_emg.append(emg_full[i, -tail:, :])
                    windows_nirs.append(nirs_full[i, -tail:, :])
                    windows_phases.append(phases_full[i, -tail:])
                    total_len = tail

                concat_emg_list.append(torch.cat(windows_emg, dim=0))
                concat_nirs_list.append(torch.cat(windows_nirs, dim=0))
                concat_phases_list.append(torch.cat(windows_phases, dim=0))
                lengths_list.append(total_len)

        max_len_cat = max(lengths_list)
        emg = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_emg_list])
        nirs = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_nirs_list])
        task_phases = torch.stack([F.pad(x, (0, max_len_cat - x.shape[0])) for x in concat_phases_list])
        task_phase_embeds = F.one_hot(task_phases, num_classes=config['num_task_phases']).float()
        seg_ids = None
        if config.get('use_segment_embeddings', False):
            seg_ids_list = []
            for x in concat_phases_list:
                seg = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                # Assign increasing segment ids per window: we do not have boundaries now; set all to 1
                seg.fill_(1)
                seg_ids_list.append(F.pad(seg, (0, max_len_cat - x.shape[0])))
            seg_ids = torch.stack(seg_ids_list)
        lengths_tensor = torch.tensor(lengths_list, device=device, dtype=torch.long)
        padding_mask = torch.arange(max_len_cat, device=device).unsqueeze(0).expand(len(lengths_list), -1) >= lengths_tensor.unsqueeze(1)

        emg_aug = model.module.augment_emg(emg)
        nirs_aug = model.module.nirs_jitter(nirs)

        dev_ids = batch.get('device_id', None)
        dev_ids = dev_ids.to(device) if isinstance(dev_ids, torch.Tensor) else None
        reg_type = config.get('regression_loss', 'smooth_l1')
        params_or_pred = model(emg_aug, nirs_aug, task_phase_embedding=task_phase_embeds, mode='main_regression', padding_mask=padding_mask, segment_ids=seg_ids, device_ids=dev_ids)
        if reg_type == 'smooth_l1':
            loss = nn.SmoothL1Loss()(params_or_pred, main_labels)
            # loss = nn.MSELoss()(params_or_pred, main_labels)
        elif reg_type == 'beta_nll':
            loss = beta_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
        elif reg_type == 'gamma_nll':
            loss = gamma_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
        elif reg_type == 'beta2_nll':
            loss = mix2_beta_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
        elif reg_type == 'gamma2_nll':
            loss = mix2_gamma_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
        else:
            loss = nn.SmoothL1Loss()(params_or_pred, main_labels)
        # Accumulate parameter means for logging
        if reg_type != 'smooth_l1':
            from src.utils import extract_distribution_param_means
            stats = extract_distribution_param_means(params_or_pred.detach(), reg_type)
            bs = params_or_pred.size(0)
            for k, v in stats.items():
                param_sums[k] = param_sums.get(k, 0.0) + float(v) * bs
            num_samples += bs

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
        optimizer.step()

        total_loss_epoch += loss.item() * emg.size(0)

    total_loss_tensor = torch.tensor(total_loss_epoch).to(device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    # Correctly average loss over the entire dataset across all GPUs
    if hasattr(dataloader, "dataset"):
        dataset_size = len(dataloader.dataset)
    else:
        dataset_size = len(dataloader.sampler.data_source)
    avg_loss = total_loss_tensor.item() / dataset_size
    # Log parameter means at epoch end on rank 0
    if rank == 0 and logger is not None and reg_type_global != 'smooth_l1' and num_samples > 0:
        for k, s in param_sums.items():
            logger.log_scalar(f"Supervised/Train_Param_{k}", s / num_samples, epoch)
    return avg_loss


@torch.no_grad()
def evaluate(rank, model, dataloader, device, config, save_predictions=False):
    model.eval()
    all_preds_epoch, all_labels_epoch = [], []
    # Additional accumulators for hybrid mode
    all_preds_reg_epoch, all_labels_reg_epoch = [], []
    all_logits_epoch, all_class_labels_epoch = [], []

    pbar = tqdm(dataloader, desc="Evaluation", disable=(rank != 0 and dataloader.sampler is None))

    # Accumulate NLL when using distributional regression
    total_nll_accum = 0.0
    total_samples = 0
    reg_type = config.get('regression_loss', 'smooth_l1')
    # Track parameter means over eval
    param_sums = {}
    # Per-sample transformed distribution parameters (for saving)
    per_sample_params = {}  # name -> list of floats aligned with dataset order
    # Metadata accumulators
    meta_subject_ids_all = []
    meta_device_strs_all = []
    meta_class_labels_all = []

    for batch in pbar:
        emg_full, nirs_full, phases_full = batch['emg'].to(device), batch['nirs'].to(device), batch['task_phase_ids'].to(device)

        use_windowing = config.get('use_trial_windowing', True)
        fs = config['target_sample_rate']
        win_len = int(config.get('trial_window_seconds', 60) * fs)
        sep_len = int(config.get('window_separator_seconds', 0) * fs)
        num_trials_before = config.get('num_trials_before_mvc_for_final', 3)
        mvc_post = int(config.get('mvc_post_seconds', 120) * fs)

        main_labels = batch['post_mvc_pain'].to(device)

        if use_windowing:
            concat_emg_list, concat_nirs_list, concat_phases_list, lengths_list = [], [], [], []
            intermediate_indices_list = batch['intermediate_pain_indices']
            B = emg_full.size(0)
            for i in range(B):
                phases_i = phases_full[i]
                L = batch['lengths'][i].item() if 'lengths' in batch else emg_full.shape[1]
                mask_mvc = (phases_i[:L] == 2)
                if mask_mvc.any():
                    mvc_start = torch.where(mask_mvc)[0][0].item()
                    mvc_end = torch.where(mask_mvc)[0][-1].item()
                    mvc_window_start = mvc_start
                    mvc_window_end = min(L, mvc_end + mvc_post)
                else:
                    mvc_window_start, mvc_window_end = None, None

                idxs = intermediate_indices_list[i]
                valid_trials = [t for t in range(len(idxs)) if idxs[t].item() > 0]
                if mvc_window_start is not None:
                    valid_trials = [t for t in valid_trials if idxs[t].item() <= mvc_window_start]
                chosen = valid_trials[-num_trials_before:] if len(valid_trials) > 0 else []

                windows_emg, windows_nirs, windows_phases = [], [], []
                total_len = 0
                for t in chosen:
                    idx_t = idxs[t].item()
                    s = max(0, idx_t - win_len)
                    e = idx_t
                    windows_emg.append(emg_full[i, s:e, :])
                    windows_nirs.append(nirs_full[i, s:e, :])
                    windows_phases.append(phases_full[i, s:e])
                    total_len += (e - s)
                    if sep_len > 0:
                        windows_emg.append(torch.zeros(sep_len, emg_full.shape[-1], device=device))
                        windows_nirs.append(torch.zeros(sep_len, nirs_full.shape[-1], device=device))
                        windows_phases.append(torch.zeros(sep_len, dtype=phases_full.dtype, device=device))
                        total_len += sep_len

                if mvc_window_start is not None and mvc_window_end is not None and mvc_window_end > mvc_window_start:
                    windows_emg.append(emg_full[i, mvc_window_start:mvc_window_end, :])
                    windows_nirs.append(nirs_full[i, mvc_window_start:mvc_window_end, :])
                    windows_phases.append(phases_full[i, mvc_window_start:mvc_window_end])
                    total_len += (mvc_window_end - mvc_window_start)

                if total_len == 0:
                    tail = min(win_len, emg_full.shape[1])
                    windows_emg.append(emg_full[i, -tail:, :])
                    windows_nirs.append(nirs_full[i, -tail:, :])
                    windows_phases.append(phases_full[i, -tail:])
                    total_len = tail

                concat_emg_list.append(torch.cat(windows_emg, dim=0))
                concat_nirs_list.append(torch.cat(windows_nirs, dim=0))
                concat_phases_list.append(torch.cat(windows_phases, dim=0))
                lengths_list.append(total_len)

            max_len_cat = max(lengths_list)
            emg = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_emg_list])
            nirs = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_nirs_list])
            task_phases = torch.stack([F.pad(x, (0, max_len_cat - x.shape[0])) for x in concat_phases_list])
            task_phase_embeds = F.one_hot(task_phases, num_classes=config['num_task_phases']).float()
            seg_ids = None
            if config.get('use_segment_embeddings', False):
                seg_ids_list = []
                for x in concat_phases_list:
                    seg = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                    seg.fill_(1)
                    seg_ids_list.append(F.pad(seg, (0, max_len_cat - x.shape[0])))
                seg_ids = torch.stack(seg_ids_list)
            lengths_tensor = torch.tensor(lengths_list, device=device, dtype=torch.long)
            padding_mask = torch.arange(max_len_cat, device=device).unsqueeze(0).expand(len(lengths_list), -1) >= lengths_tensor.unsqueeze(1)
        else:
            # Match supervised_train_epoch behavior: truncate each sequence at MVC end + post
            B = emg_full.size(0)
            emg_list, nirs_list, phases_list, lengths = [], [], [], []
            for i in range(B):
                phases_i = phases_full[i]
                L = batch['lengths'][i].item() if 'lengths' in batch else emg_full.shape[1]
                mask_mvc = (phases_i[:L] == 2)
                s = 0
                e = L
                if mask_mvc.any():
                    mvc_start = torch.where(mask_mvc)[0][0].item()
                    mvc_end = torch.where(mask_mvc)[0][-1].item()
                    mvc_window_end = min(L, mvc_end + mvc_post)
                    e = mvc_window_end
                emg_list.append(emg_full[i, s:e, :])
                nirs_list.append(nirs_full[i, s:e, :])
                phases_list.append(phases_full[i, s:e])
                lengths.append(e - s)

            max_len = max(lengths)
            emg = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in emg_list])
            nirs = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in nirs_list])
            task_phases = torch.stack([F.pad(x, (0, max_len - x.shape[0])) for x in phases_list])
            task_phase_embeds = F.one_hot(task_phases, num_classes=config['num_task_phases']).float()
            lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
            padding_mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lengths), -1) >= lengths_tensor.unsqueeze(1)
            seg_ids = None
            if config.get('use_segment_embeddings', False):
                seg_ids = torch.ones(emg.size(0), emg.size(1), dtype=torch.long, device=device)

        # capture metadata
        if 'subject_ids' in batch: meta_subject_ids_all.extend(batch['subject_ids'])
        if 'device_strs' in batch: meta_device_strs_all.extend(batch['device_strs'])
        if 'class_label' in batch: meta_class_labels_all.extend(batch['class_label'].cpu().tolist())

        dev_ids = batch.get('device_id', None)
        dev_ids = dev_ids.to(device) if isinstance(dev_ids, torch.Tensor) else None

        params_or_pred = model(emg, nirs, task_phase_embedding=task_phase_embeds, mode='main_regression', padding_mask=padding_mask, segment_ids=seg_ids, device_ids=dev_ids)
        if reg_type != 'smooth_l1':
            if reg_type == 'beta_nll':
                batch_nll = beta_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
            elif reg_type == 'gamma_nll':
                batch_nll = gamma_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
            elif reg_type == 'beta2_nll':
                batch_nll = mix2_beta_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
            elif reg_type == 'gamma2_nll':
                batch_nll = mix2_gamma_nll_from_params(params_or_pred, main_labels, eps=config.get('distribution_eps', 1e-6))
            else:
                batch_nll = torch.tensor(0.0, device=device)
            total_nll_accum += batch_nll.item() * emg.size(0)
            total_samples += emg.size(0)
            # Accumulate parameter means for logging
            from src.utils import extract_distribution_param_means
            stats = extract_distribution_param_means(params_or_pred.detach(), reg_type)
            bs = params_or_pred.size(0)
            for k, v in stats.items():
                param_sums[k] = param_sums.get(k, 0.0) + float(v) * bs
            # Store per-sample parameters for saving
            from src.utils import extract_distribution_params_per_sample
            dps = extract_distribution_params_per_sample(params_or_pred.detach(), reg_type)
            for name, tensor_vals in dps.items():
                lst = per_sample_params.get(name, [])
                lst.extend(tensor_vals.detach().cpu().tolist())
                per_sample_params[name] = lst

        # Keep tensors on the GPU for the gather operation
        if reg_type == 'smooth_l1':
            all_preds_epoch.append(params_or_pred)
        else:
            mean_preds = params_to_means(params_or_pred, reg_type)
            all_preds_epoch.append(mean_preds)

        all_labels_epoch.append(main_labels)

    # --- Aggregate and Calculate Metrics on Rank 0 ---
    if isinstance(dataloader.sampler, DistributedSampler):
        gathered_preds = [torch.zeros_like(torch.cat(all_preds_epoch)) for _ in range(dist.get_world_size())] if rank == 0 else None
        gathered_labels = [torch.zeros_like(torch.cat(all_labels_epoch)) for _ in range(dist.get_world_size())] if rank == 0 else None

        dist.gather(torch.cat(all_preds_epoch), gather_list=gathered_preds, dst=0)
        dist.gather(torch.cat(all_labels_epoch), gather_list=gathered_labels, dst=0)
    
        # Move to CPU only on rank 0 after gathering is complete
        if rank == 0:
            final_preds = torch.cat(gathered_preds).cpu()[:len(dataloader.dataset)]
            final_labels = torch.cat(gathered_labels).cpu()[:len(dataloader.dataset)]
        else:
            # Other ranks don't need the final tensors
            final_preds, final_labels = None, None
    else:
        final_preds = torch.cat(all_preds_epoch).cpu()
        final_labels = torch.cat(all_labels_epoch).cpu()

    # Inverse-transform labels
    stats = config.get('label_transform_stats')
    final_preds_raw = inverse_transform_labels(final_preds, stats, 'main')
    final_labels_raw = inverse_transform_labels(final_labels, stats, 'main')

    if rank == 0:
        
        metrics = calculate_regression_metrics(final_preds_raw, final_labels_raw)
        # Include NLL when using distributional regression
        if reg_type != 'smooth_l1' and total_samples > 0:
            metrics['nll'] = total_nll_accum / total_samples
            # Include parameter means
            for k, s in param_sums.items():
                metrics[f'param_mean_{k}'] = s / total_samples
        
        if save_predictions:
            log_dir = config.get('log_dir', 'runs/default')
            os.makedirs(log_dir, exist_ok=True)
            fold_tag = f"_fold{config.get('fold_idx', 0)}"
            pred_log_path = os.path.join(log_dir, f"test_predictions{fold_tag}.pt")
            # In multi-GPU eval, metadata lists may be incomplete; save empty lists in that case
            if isinstance(dataloader.sampler, DistributedSampler):
                subj_ids_save, dev_strs_save, class_labels_save = [], [], []
            else:
                subj_ids_save, dev_strs_save, class_labels_save = meta_subject_ids_all, meta_device_strs_all, meta_class_labels_all

            torch.save({'predictions': final_preds_raw, 'labels': final_labels_raw,
                'subject_ids': subj_ids_save, 'device_strs': dev_strs_save, 'class_labels': class_labels_save,
                'distribution_params': per_sample_params if reg_type!='smooth_l1' else None,
                'fold_idx': config.get('fold_idx', 0)}, pred_log_path)
            print(f"Saved final predictions and labels to {pred_log_path}")
        return metrics
    return None


def adapt_and_evaluate(rank, model, subjects, device, config, *, return_outputs: bool = False, save_prefix: str = "adapted_test_predictions", run_adaptation: bool = True):
    """
    Performs test-time adaptation on the encoders and SSP heads for each subject and evaluates.
    Gradients are ENABLED for the adaptation loop and DISABLED for the final evaluation pass.
    """
    # This function runs only on the main process (rank 0)
    if rank != 0: 
        return None
    
    all_adapted_preds, all_labels = [], []
    # For hybrid evaluation
    all_adapted_preds_reg, all_labels_reg = [], []
    all_class_logits, all_labels_cls = [], []
    meta_subject_ids_all, meta_device_strs_all, meta_class_labels_all = [], [], []
    
    pbar = tqdm(subjects, desc="Adapting and Evaluating Subjects", disable=(rank != 0))
    
    # Per-sample transformed distribution parameters across all subjects
    per_sample_params = {}  # name -> list of floats
    for i, subject_data in enumerate(pbar):
        # Create DataLoaders for this single subject
        # Build a repeat-chunk dataset so adaptation_batch_size and multiple updates per epoch are effective
        class _SSPAdaptFixedChunkDataset(Dataset):
            def __init__(self, meta, cfg, start_indices):
                feats = torch.load(meta['feature_path'])
                self.emg = feats['emg']
                self.nirs = feats['nirs']
                self.chunk_len = int(cfg['ssp_context_len'])
                self.starts = list(start_indices)
            def __len__(self):
                return len(self.starts)
            def __getitem__(self, idx):
                T = self.emg.shape[0]
                L = self.chunk_len
                s = int(self.starts[idx])
                # Guardrails in case of edge conditions
                if T <= L:
                    s = 0
                else:
                    s = max(0, min(s, T - L))
                e = min(T, s + L)
                return self.emg[s:e, :], self.nirs[s:e, :]

        # Build fixed train/monitor chunk indices for adaptation early stopping (still same subject).
        feats_tmp = torch.load(subject_data['feature_path'])
        T_full = int(feats_tmp['emg'].shape[0])
        L = int(config['ssp_context_len'])
        max_start = max(0, T_full - L)
        num_train_chunks = int(config.get('adaptation_chunks_per_epoch', 256))
        num_mon_chunks = int(config.get('adaptation_es_monitor_chunks', 16))
        # Stable per-subject seed (independent of Python hash randomization)
        sid = str(subject_data.get('id', 'unknown'))
        seed_bytes = hashlib.sha256(sid.encode('utf-8')).digest()
        seed_int = int.from_bytes(seed_bytes[:8], byteorder='little', signed=False) % (2**31 - 1)
        # seed_int = i
        g = torch.Generator()
        g.manual_seed(seed_int)
        if max_start <= 0:
            starts_train = [0] * num_train_chunks
            starts_mon = [0] * num_mon_chunks
        else:
            # Choose NON-OVERLAPPING windows (by construction): starts spaced by chunk length L.
            # This guarantees no time overlap between train and monitor chunks.
            candidate_starts = list(range(0, max_start + 1, max(1, L)))
            if len(candidate_starts) < 2:
                # Subject too short for disjoint windows; fall back (cannot guarantee no overlap).
                starts_train = [0] * num_train_chunks
                starts_mon = [0] * num_mon_chunks
            else:
                perm = torch.randperm(len(candidate_starts), generator=g).tolist()
                candidate_starts = [candidate_starts[i] for i in perm]

                # Allocate disjoint unique starts to train then monitor
                n_train_unique = min(num_train_chunks, len(candidate_starts))
                train_unique = candidate_starts[:n_train_unique]
                remaining = candidate_starts[n_train_unique:]
                n_mon_unique = min(num_mon_chunks, len(remaining))
                mon_unique = remaining[:n_mon_unique]

                # If we need more chunks than unique windows, sample WITH replacement within each split
                def _fill_to_n(pool, n_needed):
                    if n_needed <= len(pool):
                        return pool[:n_needed]
                    if len(pool) == 0:
                        return [0] * n_needed
                    extra = torch.randint(low=0, high=len(pool), size=(n_needed - len(pool),), generator=g).tolist()
                    return pool + [pool[i] for i in extra]

                starts_train = _fill_to_n(train_unique, num_train_chunks)
                starts_mon = _fill_to_n(mon_unique, num_mon_chunks)

        ssp_adapt_dataset = _SSPAdaptFixedChunkDataset(subject_data, config, starts_train)
        ssp_adapt_loader = DataLoader(ssp_adapt_dataset, batch_size=config['adaptation_batch_size'], shuffle=True)
        sup_eval_dataset = PainDataset([subject_data], config, mode='test')
        sup_eval_loader = DataLoader(sup_eval_dataset, batch_size=1, collate_fn=collate_fn_supervised)

        # --- Adaptation Loop (Requires Gradients) ---
        # 1. Instantiate the model for adaptation, using MultimodalModel directly
        adapted_model = MultimodalModel(config).to(device)
        adapted_model.load_state_dict(model.state_dict()) # Load the best generic model weights
        
        # 2. Freeze all parameters first
        for param in adapted_model.parameters():
            param.requires_grad = False
            
        # 3. Unfreeze only the layers to be adapted: encoders and SSP heads
        for param in adapted_model.emg_encoder.parameters():
            param.requires_grad = True
        for param in adapted_model.nirs_encoder.parameters():
            param.requires_grad = False
        for param in adapted_model.nirs_norm.parameters():
            param.requires_grad = False
        for param in adapted_model.emg_norm.parameters():
            param.requires_grad = True
        # for param in adapted_model.local_ssp_head.parameters():
        #     param.requires_grad = True
        for param in adapted_model.local_ssp_head_emg.parameters():
            param.requires_grad = True
        for param in adapted_model.local_ssp_head_nirs.parameters():
            param.requires_grad = True
        for param in adapted_model.global_ssp_head.parameters():
            param.requires_grad = False
        for param in adapted_model.fusion.parameters():
            param.requires_grad = True
        if config.get('ssp_use_decoder', False):
            for param in adapted_model.ssp_decoder.parameters():
                param.requires_grad = True
            
        # 4. Create an optimizer for only the unfrozen (trainable) parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=config['adaptation_lr'])
        adaptation_epochs = max(0, int(config.get('adaptation_epochs', 0)))
        scheduler = None
        if adaptation_epochs > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adaptation_epochs)
        
        # 5. Run the adaptation training loop (optional)
        adapted_model.train() # Set to training mode for adaptation
        emg_dim = config['emg_feature_dim']

        if run_adaptation and adaptation_epochs > 0:
            for epoch in range(adaptation_epochs):
                # This is a simplified, non-distributed training loop for the single subject
                train_loss_sum = 0.0
                train_batches = 0

                for emg_chunk, nirs_chunk in ssp_adapt_loader:
                    emg_chunk, nirs_chunk = emg_chunk.to(device), nirs_chunk.to(device)
                    optimizer.zero_grad()

                    outputs_local = adapted_model(emg_chunk, nirs_chunk, mode='ssp')

                    (local_preds, local_targets, emg_mask_ds, nirs_mask_ds,
                        emg_mask_hr, nirs_mask_hr, emg_aug, nirs_aug) = outputs_local

                    local_preds_emg = local_preds[..., :emg_dim]
                    local_preds_nirs = local_preds[..., emg_dim:]
                    local_targets_emg = local_targets[..., :emg_dim]
                    local_targets_nirs = local_targets[..., emg_dim:]

                    loss_local_emg = F.l1_loss(local_preds_emg[emg_mask_ds], local_targets_emg[emg_mask_ds])
                    loss_local_nirs = F.l1_loss(local_preds_nirs[nirs_mask_ds], local_targets_nirs[nirs_mask_ds])
                    loss_local = loss_local_emg + 0.1 * loss_local_nirs
                    # loss_local = loss_local_emg

                    cos_alpha_nirs = float(config.get('ssp_cosine_alpha_nirs', 0.0) or 0.0)
                    loss_cos = torch.tensor(0.0, device=device)

                    if (cos_alpha_nirs > 0.0) and bool(nirs_mask_ds.any()):
                        pn = local_preds_nirs[nirs_mask_ds]
                        tn = local_targets_nirs[nirs_mask_ds]
                        loss_cos = loss_cos + (1.0 - F.cosine_similarity(pn, tn, dim=-1)).mean() * cos_alpha_nirs


                    loss = loss_local + loss_cos

                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.detach().item())
                    train_batches += 1

                # Step LR scheduler once per epoch
                if scheduler is not None:
                    scheduler.step()
        
        # --- Final Evaluation (No Gradients) ---
        adapted_model.eval() # Set to evaluation mode
        with torch.no_grad():
            batch = next(iter(sup_eval_loader))
            emg_full = batch['emg'].to(device)
            nirs_full = batch['nirs'].to(device)

            main_labels = batch['post_mvc_pain']
            phases_full = batch['task_phase_ids'].to(device)

            # Build inputs per subject according to use_trial_windowing
            fs = config['target_sample_rate']
            win_len = int(config.get('trial_window_seconds', 60) * fs)
            sep_len = int(config.get('window_separator_seconds', 0) * fs)
            num_trials_before = int(config.get('num_trials_before_mvc_for_final', 3))
            mvc_post = int(config.get('mvc_post_seconds', 120) * fs)

            use_windowing = config.get('use_trial_windowing', True)
            if use_windowing:
                concat_emg_list, concat_nirs_list, concat_phases_list, lengths_list = [], [], [], []
                intermediate_indices_list = batch['intermediate_pain_indices']
                B = emg_full.size(0)
                for i in range(B):
                    phases_i = phases_full[i]
                    L = batch['lengths'][i].item() if 'lengths' in batch else emg_full.shape[1]
                    mask_mvc = (phases_i[:L] == 2)
                    if mask_mvc.any():
                        mvc_start = torch.where(mask_mvc)[0][0].item()
                        mvc_end = torch.where(mask_mvc)[0][-1].item()
                        mvc_window_start = mvc_start
                        mvc_window_end = min(L, mvc_end + mvc_post)
                    else:
                        mvc_window_start, mvc_window_end = None, None

                    idxs = intermediate_indices_list[i]
                    valid_trials = [t for t in range(len(idxs)) if idxs[t].item() > 0]
                    if mvc_window_start is not None:
                        valid_trials = [t for t in valid_trials if idxs[t].item() <= mvc_window_start]
                    chosen = valid_trials[-num_trials_before:] if len(valid_trials) > 0 else []

                    windows_emg, windows_nirs, windows_phases = [], [], []
                    total_len = 0
                    for t in chosen:
                        idx_t = idxs[t].item()
                        s = max(0, idx_t - win_len)
                        e = idx_t
                        windows_emg.append(emg_full[i, s:e, :])
                        windows_nirs.append(nirs_full[i, s:e, :])
                        windows_phases.append(phases_full[i, s:e])
                        total_len += (e - s)
                        if sep_len > 0:
                            windows_emg.append(torch.zeros(sep_len, emg_full.shape[-1], device=device))
                            windows_nirs.append(torch.zeros(sep_len, nirs_full.shape[-1], device=device))
                            windows_phases.append(torch.zeros(sep_len, dtype=phases_full.dtype, device=device))
                            total_len += sep_len

                    if mvc_window_start is not None and mvc_window_end is not None and mvc_window_end > mvc_window_start:
                        windows_emg.append(emg_full[i, mvc_window_start:mvc_window_end, :])
                        windows_nirs.append(nirs_full[i, mvc_window_start:mvc_window_end, :])
                        windows_phases.append(phases_full[i, mvc_window_start:mvc_window_end])
                        total_len += (mvc_window_end - mvc_window_start)

                    if total_len == 0:
                        tail = min(win_len, emg_full.shape[1])
                        windows_emg.append(emg_full[i, -tail:, :])
                        windows_nirs.append(nirs_full[i, -tail:, :])
                        windows_phases.append(phases_full[i, -tail:])
                        total_len = tail

                    concat_emg_list.append(torch.cat(windows_emg, dim=0))
                    concat_nirs_list.append(torch.cat(windows_nirs, dim=0))
                    concat_phases_list.append(torch.cat(windows_phases, dim=0))
                    lengths_list.append(total_len)

                max_len_cat = max(lengths_list)
                emg = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_emg_list])
                nirs = torch.stack([F.pad(x, (0, 0, 0, max_len_cat - x.shape[0])) for x in concat_nirs_list])
                task_phases = torch.stack([F.pad(x, (0, max_len_cat - x.shape[0])) for x in concat_phases_list])
                task_phase_embeds = F.one_hot(task_phases, num_classes=config['num_task_phases']).float()
                lengths_tensor = torch.tensor(lengths_list, device=device, dtype=torch.long)
                padding_mask = torch.arange(max_len_cat, device=device).unsqueeze(0).expand(len(lengths_list), -1) >= lengths_tensor.unsqueeze(1)
                seg_ids = None
                if config.get('use_segment_embeddings', False):
                    seg_ids = torch.ones(emg.size(0), max_len_cat, dtype=torch.long, device=device)
            else:
                # Non-windowed: truncate to MVC end + post to match supervised train
                B = emg_full.size(0)
                emg_list, nirs_list, phases_list, lengths = [], [], [], []
                for i in range(B):
                    phases_i = phases_full[i]
                    L = batch['lengths'][i].item() if 'lengths' in batch else emg_full.shape[1]
                    mask_mvc = (phases_i[:L] == 2)
                    s = 0
                    e = L
                    if mask_mvc.any():
                        mvc_start = torch.where(mask_mvc)[0][0].item()
                        mvc_end = torch.where(mask_mvc)[0][-1].item()
                        mvc_window_end = min(L, mvc_end + mvc_post)
                        e = mvc_window_end
                    emg_list.append(emg_full[i, s:e, :])
                    nirs_list.append(nirs_full[i, s:e, :])
                    phases_list.append(phases_full[i, s:e])
                    lengths.append(e - s)

                max_len = max(lengths)
                emg = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in emg_list])
                nirs = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in nirs_list])
                task_phases = torch.stack([F.pad(x, (0, max_len - x.shape[0])) for x in phases_list])
                task_phase_embeds = F.one_hot(task_phases, num_classes=config['num_task_phases']).float()
                lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
                padding_mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lengths), -1) >= lengths_tensor.unsqueeze(1)
                seg_ids = None
                if config.get('use_segment_embeddings', False):
                    seg_ids = torch.ones(emg.size(0), emg.size(1), dtype=torch.long, device=device)

            dev_ids = batch.get('device_id', None)
            dev_ids = dev_ids.to(device) if isinstance(dev_ids, torch.Tensor) else None

            reg_type = config.get('regression_loss', 'smooth_l1')
            params_or_pred = adapted_model(emg, nirs, task_phase_embedding=task_phase_embeds, mode='main_regression', padding_mask=padding_mask, segment_ids=seg_ids, device_ids=dev_ids)
            pred = params_or_pred if reg_type == 'smooth_l1' else params_to_means(params_or_pred, reg_type)
            # Collect per-sample distribution params for saving
            if reg_type != 'smooth_l1':
                from src.utils import extract_distribution_params_per_sample
                dps = extract_distribution_params_per_sample(params_or_pred.detach(), reg_type)
                for name, tensor_vals in dps.items():
                    lst = per_sample_params.get(name, [])
                    lst.extend(tensor_vals.detach().cpu().tolist())
                    per_sample_params[name] = lst

            all_adapted_preds.append(pred.cpu())
            all_labels.append(main_labels)
            # Capture metadata for this subject
            try:
                sid = subject_data.get('id', '')
                meta_subject_ids_all.append(sid)
                lbl = torch.load(subject_data['label_path'])
                dev = lbl.get('device', '')
                meta_device_strs_all.append(str(dev))
                cl = lbl.get('class_label', None)
                meta_class_labels_all.append(int(cl) if cl is not None else None)
            except Exception:
                meta_subject_ids_all.append('')
                meta_device_strs_all.append('')
                meta_class_labels_all.append(None)

    # After all subjects are adapted and predicted:
    final_preds = torch.cat(all_adapted_preds)
    final_labels = torch.cat(all_labels)

    stats = config.get('label_transform_stats')
    final_preds_raw = inverse_transform_labels(final_preds, stats, 'main')
    final_labels_raw = inverse_transform_labels(final_labels, stats, 'main')

    # Calculate full metrics
    metrics = calculate_regression_metrics(final_preds_raw, final_labels_raw)
    
    # Save predictions
    log_dir = config.get('log_dir', 'runs/default')
    os.makedirs(log_dir, exist_ok=True)
    fold_tag = f"_fold{config.get('fold_idx', 0)}"
    pred_log_path = os.path.join(log_dir, f"{save_prefix}{fold_tag}.pt")

    torch.save({'predictions': final_preds_raw, 'labels': final_labels_raw,
        'subject_ids': meta_subject_ids_all, 'device_strs': meta_device_strs_all, 'class_labels': meta_class_labels_all,
        'fold_idx': config.get('fold_idx', 0)}, pred_log_path)
    print(f"Saved adapted predictions and labels to {pred_log_path}")
    
    # Save per-subject outputs including distribution params (rank 0 only)
    log_dir = config.get('log_dir', 'runs/default')
    os.makedirs(log_dir, exist_ok=True)
    fold_tag = f"_fold{config.get('fold_idx', 0)}"
    pred_log_path = os.path.join(log_dir, f"{save_prefix}{fold_tag}.pt")
    # subject ids list and device/class labels were collected in meta_*_all
    torch.save({
        'predictions': final_preds_raw, 'labels': final_labels_raw,
        'subject_ids': meta_subject_ids_all, 'device_strs': meta_device_strs_all, 'class_labels': meta_class_labels_all,
        'distribution_params': per_sample_params if config.get('regression_loss','smooth_l1')!='smooth_l1' else None,
        'fold_idx': config.get('fold_idx', 0)
    }, pred_log_path)
    print(f"Saved adapted final predictions and labels to {pred_log_path}")
    if return_outputs:
        outputs = {
            'subject_ids': meta_subject_ids_all,
            'device_strs': meta_device_strs_all,
        }
        
        outputs.update({'preds_raw': final_preds_raw, 'labels_raw': final_labels_raw})
        return metrics, outputs
    return metrics


def _normalize_adaptation_epoch_candidates(config):
    candidates = config.get('adaptation_epoch_selection_candidates', [config.get('adaptation_epochs', 0)])
    if candidates is None:
        candidates = [config.get('adaptation_epochs', 0)]
    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]
    normalized = sorted({int(v) for v in candidates if int(v) >= 0})
    if not normalized:
        raise ValueError("adaptation_epoch_selection_candidates must contain at least one non-negative integer.")
    return normalized


def _get_adaptation_epoch_selection_score(metrics):
    keys = ['spearman_rho_left', 'spearman_rho_right']
    vals = [float(metrics[k]) for k in keys if k in metrics]
    if not vals:
        raise ValueError(f"Could not find Spearman metrics in results: {list(metrics.keys())}")
    return float(sum(vals) / len(vals))


def _split_transfer_validation_subjects(test_subjects, config, base_seed, fold_idx):
    num_val = int(config.get('adaptation_transfer_validation_num_subjects', 0) or 0)
    if num_val <= 0:
        raise ValueError(
            "adaptation_transfer_validation_num_subjects must be > 0 when using "
            "adaptation_epoch_selection_source='transfer_validation_set'."
        )
    if num_val >= len(test_subjects):
        raise ValueError(
            f"Transfer validation needs fewer subjects than the test set size. "
            f"Requested {num_val}, available {len(test_subjects)}."
        )

    seed_offset = int(config.get('adaptation_transfer_validation_seed_offset', 31415))
    rng = np.random.default_rng(int(base_seed) + 7000 * int(fold_idx) + seed_offset)
    chosen = sorted(int(i) for i in rng.choice(len(test_subjects), size=num_val, replace=False))
    val_subjects = [test_subjects[i] for i in chosen]
    chosen_set = set(chosen)
    eval_subjects = [s for i, s in enumerate(test_subjects) if i not in chosen_set]
    return val_subjects, eval_subjects


def _build_adaptation_epoch_selection_cv_folds(subjects, config, base_seed):
    n_subjects = len(subjects)
    n_folds = int(config.get('adaptation_epoch_selection_cv_folds', 5) or 0)
    if n_folds < 2:
        raise ValueError("adaptation_epoch_selection_cv_folds must be at least 2.")
    if n_folds > n_subjects:
        raise ValueError(
            f"adaptation_epoch_selection_cv_folds={n_folds} exceeds the number of available "
            f"subjects ({n_subjects})."
        )

    seed_offset = int(config.get('adaptation_epoch_selection_cv_seed_offset', 27182))
    split_seed = int(base_seed) + 11000 * int(config.get('fold_idx', 0)) + seed_offset
    indices = np.arange(n_subjects)

    strat_labels = None

    device_labels = []
    for s in subjects:
        lp = torch.load(s['label_path'])
        device_labels.append(str(lp.get('device', 'unknown')))
    counts = pd.Series(device_labels).value_counts()
    if len(counts) >= 2 and int(counts.min()) >= n_folds:
        strat_labels = device_labels

    if strat_labels is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        fold_indices = list(splitter.split(indices, strat_labels))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        fold_indices = list(splitter.split(indices))

    return [([subjects[i] for i in train_idx], [subjects[i] for i in val_idx]) for train_idx, val_idx in fold_indices]

def select_optimal_adaptation_epochs(rank, model, subjects, device, config):
    if rank != 0:
        return int(config.get('adaptation_epochs', 0))
    if not subjects:
        raise ValueError("No validation subjects available for adaptation epoch selection.")

    candidates = _normalize_adaptation_epoch_candidates(config)
    best_epochs = None
    best_score = -float('inf')
    selection_rows = []

    for epochs in candidates:
        candidate_config = copy.deepcopy(config)
        candidate_config['adaptation_epochs'] = int(epochs)
        metrics, _ = adapt_and_evaluate(
            rank, model, subjects, device, candidate_config,
            return_outputs=True,
            save_prefix=f"adapted_validation_predictions_epoch_{int(epochs)}",
            run_adaptation=(int(epochs) > 0),
        )
        score = _get_adaptation_epoch_selection_score(metrics)
        selection_rows.append({'epochs': int(epochs), 'score': float(score), 'metrics': metrics})
        if (score > best_score) or (math.isclose(score, best_score) and (best_epochs is None or int(epochs) < best_epochs)):
            best_score = float(score)
            best_epochs = int(epochs)

        print(f"Epoch {epochs} score: {score:.4f}")

    log_dir = config.get('log_dir', 'runs/default')
    os.makedirs(log_dir, exist_ok=True)
    fold_tag = f"_fold{config.get('fold_idx', 0)}"
    selection_path = os.path.join(log_dir, f"adaptation_epoch_selection{fold_tag}.pt")
    torch.save({
        'selected_adaptation_epochs': best_epochs,
        'best_score_mean_spearman_rho': best_score,
        'candidates': candidates,
        'results': selection_rows,
        'validation_subject_ids': [s.get('id', '') for s in subjects],
        'validation_source': config.get('adaptation_epoch_selection_source', 'transfer_validation_set'),
    }, selection_path)
    print(f"Saved adaptation epoch selection results to {selection_path}")

    return int(best_epochs)

def main_worker(rank, world_size, args):
    # Seed strategy (DDP-safe):
    # - Use a shared base seed for operations that must be identical across ranks (e.g., random_split).
    # - Use per-rank seeds for stochastic augmentations/masking during training.
    base_seed = int(getattr(args, "seed", 42))
    set_seed(base_seed)
    if world_size > 1:
        # If launched with torchrun, it provides MASTER_ADDR/MASTER_PORT via env.
        # Do NOT override those with our CLI flag, otherwise ranks can hang waiting
        # for a TCPStore that was never started on the overridden port.
        if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            setup_ddp(rank, world_size, master_port=None)
        else:
            setup_ddp(rank, world_size, args.master_port)

    # Graceful shutdown on SIGINT/SIGTERM inside each rank to avoid TCPStore/NCCL spam.
    def _handle_term(signum, frame):
        try:
            if world_size > 1:
                cleanup_ddp()
        finally:
            # Hard-exit to ensure all ranks stop promptly.
            os._exit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
    device = torch.device(f"cuda:{rank}")
    
    # --- Create Log Directory ---
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
    if world_size > 1:
        # Make barrier explicit about which device to use (NCCL).
        dist.barrier(device_ids=[rank])

    logger = TensorboardLogger(log_dir=args.log_dir) if rank == 0 else None
    config = get_config()
    config['fold_idx'] = args.fold_idx
    config['log_dir'] = args.log_dir
    # Optional CLI override for adaptation epochs (useful for Phase 4 sweeps).
    if getattr(args, 'adaptation_epochs', None) is not None:
        config['adaptation_epochs'] = int(args.adaptation_epochs)
    
    # --- Data Loading & Splitting (only if needed for the current phase) ---
    if args.run_supervised or args.run_final_evaluation:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        processed_data_dir = os.path.join(project_root, "data/processed")
        all_label_files = glob.glob(os.path.join(processed_data_dir, "*_labels.pt"))
        if not all_label_files:
            if rank == 0: raise FileNotFoundError(f"No processed label files found in {processed_data_dir}.")
            return
        
        all_subjects_metadata = []
        for label_path in all_label_files:
            subject_id = os.path.basename(label_path).replace("_labels.pt", "")
            feature_path = os.path.join(processed_data_dir, f"{subject_id}_features_logspec.pt")
            if os.path.exists(feature_path):
                all_subjects_metadata.append({'id': subject_id, 'feature_path': feature_path, 'label_path': label_path})
        
        all_subjects_metadata.sort(key=lambda x: x['id'])
        
        if args.use_separate_test_set:
            train_participants_df = pd.read_csv(args.participants_train)
            test_participants_df = pd.read_csv(args.participants_test)
            train_val_ids = set(train_participants_df['sub_id'])
            test_ids = set(test_participants_df['sub_id'])
            train_val_subjects = [s for s in all_subjects_metadata if s['id'] in train_val_ids]
            test_subjects = [s for s in all_subjects_metadata if s['id'] in test_ids]
        else:
            # Build candidate subject pool from participants files if provided; otherwise use all
            combined_ids = None
            class_label_map = {}
            if args.participants_train and os.path.exists(args.participants_train):
                df_tr = pd.read_csv(args.participants_train)
                ids_tr = set(df_tr['sub_id'])
                combined_ids = ids_tr if combined_ids is None else combined_ids.union(ids_tr)
                if 'class_label' in df_tr.columns:
                    class_label_map.update({row['sub_id']: int(row['class_label']) for _, row in df_tr.iterrows()})
            if args.participants_test and os.path.exists(args.participants_test):
                df_te = pd.read_csv(args.participants_test)
                ids_te = set(df_te['sub_id'])
                combined_ids = ids_te if combined_ids is None else combined_ids.union(ids_te)
                if 'class_label' in df_te.columns:
                    class_label_map.update({row['sub_id']: int(row['class_label']) for _, row in df_te.iterrows()})

            subject_pool = all_subjects_metadata if combined_ids is None else [s for s in all_subjects_metadata if s['id'] in combined_ids]

            y_labels = []
            for s in subject_pool:
                sid = s['id']
                if sid in class_label_map:
                    y_labels.append(class_label_map[sid])
                else:
                    lp = torch.load(s['label_path'])
                    y_labels.append(int(lp.get('class_label', -1)))

            device_labels = []
            for s in subject_pool:
                lp = torch.load(s['label_path'])
                device_labels.append(lp.get('device', 'unknown'))


            unique_devices = list(set(device_labels))
            if len(unique_devices) >= 2:
                skf_dev = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
                splits = list(skf_dev.split(np.arange(len(subject_pool)), device_labels))

            else:
                kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
                splits = list(kf.split(np.arange(len(subject_pool))))

            train_val_idx, test_idx = splits[args.fold_idx]
            train_val_subjects = [subject_pool[i] for i in train_val_idx]
            test_subjects = [subject_pool[i] for i in test_idx]


        # Inner validation split stratified by device when possible
        train_val_devices = []
        for s in train_val_subjects:

            lp = torch.load(s['label_path'])
            train_val_devices.append(lp.get('device', 'unknown'))

        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=0.15,
            stratify=train_val_devices, random_state=42)

        # Optional: limit supervised TRAIN subject count (keep validation/test unchanged).
        # This happens BEFORE computing label transform stats so scaling matches the training subset.
        if getattr(args, "sup_subject_limit", None) is not None:
            k = int(args.sup_subject_limit)
            if k > 0 and len(train_subjects) > k:
                off = int(getattr(args, "sup_subject_limit_seed_offset", 17))
                rng = np.random.default_rng(int(base_seed) + 9000 * int(args.fold_idx) + off)
                chosen = rng.choice(len(train_subjects), size=k, replace=False)
                # Keep deterministic order after sampling
                chosen = sorted(int(i) for i in chosen)
                train_subjects = [train_subjects[i] for i in chosen]

        # Compute label stats only if using log-transform normalization
        if config.get('use_label_log_transform', True):
            config['label_transform_stats'] = compute_label_transform_stats(train_subjects)
        else:
            config['label_transform_stats'] = None
        if rank == 0:
            print(f"Dataset split: {len(train_subjects)} train, {len(val_subjects)} validation, {len(test_subjects)} test")

    # --- Phase 1: Self-Supervised Pre-training ---
    if args.run_ssp:
        if rank == 0: print("--- Starting SSP Phase ---")

        ssp_build_mode = str(config.get('ssp_build_mode', 'ssp'))
        model = MultimodalModel(config, build_mode=ssp_build_mode).to(device)
        model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # Data loading for SSP needs to happen inside this block
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        processed_data_dir = os.path.join(project_root, "data/processed")
        all_feature_files = glob.glob(os.path.join(processed_data_dir, f"*_features_logspec.pt"))
        if not bool(getattr(args, "ssp_on_all_data", False)):
            if not args.participants_train or not os.path.exists(args.participants_train):
                raise ValueError(
                    "--participants_train must be provided for SSP unless --ssp_on_all_data is used."
                )
            train_participants_df = pd.read_csv(args.participants_train)
            train_ids = set(train_participants_df['sub_id'])
            all_feature_files = [
                fp for fp in all_feature_files
                if os.path.basename(fp).replace(f"_features_logspec.pt", "") in train_ids
            ]
        if not all_feature_files:
            if rank == 0:
                raise FileNotFoundError(f"No SSP feature files found in {processed_data_dir} for the requested subject set.")
            return
        ssp_data_source = [{'feature_path': fp, 'label_path': fp.replace(f'_features_logspec.pt', '_labels.pt')} for fp in all_feature_files]

        # Optional: limit the number of SSP subjects before building the dataset.
        # Validation remains monitor-only; training still uses the resulting full SSP dataset.
        if getattr(args, "ssp_subject_limit", None) is not None:
            k = int(args.ssp_subject_limit)
            if k > 0 and len(ssp_data_source) > k:
                g_lim = torch.Generator()
                off = int(getattr(args, "ssp_subject_limit_seed_offset", 12345))
                g_lim.manual_seed(int(base_seed) + off)
                perm = torch.randperm(len(ssp_data_source), generator=g_lim).tolist()
                keep = sorted(perm[:k])
                ssp_data_source = [ssp_data_source[i] for i in keep]


        ssp_full_dset = PainDataset(ssp_data_source, config, mode='ssp')
        # ssp_train_d, ssp_val_d = random_split(ssp_full_dset, [int(0.95 * len(ssp_full_dset)), len(ssp_full_dset) - int(0.95 * len(ssp_full_dset))])
        # Deterministic split (DDP-safe): use an explicit generator so the split is invariant to
        # RNG consumption from model initialization / RNG-advance experiments.
        # ssp_split_seed = int(config.get("ssp_split_seed", base_seed + 1000 * int(config.get("fold_idx", 0))))
        g_ssp = torch.Generator()
        g_ssp.manual_seed(base_seed)
        n_total = len(ssp_full_dset)
        n_train = int(0.95 * n_total)
        n_val = n_total - n_train
        ssp_train_d, ssp_val_d = random_split(ssp_full_dset, [n_train, n_val], generator=g_ssp)
        if rank == 0:
            print(f"SSP val data: {ssp_val_d.indices}")
        # Train on the FULL SSP dataset (reduced by ssp_subject_limit if set).
        ssp_train_sampler = DistributedSampler(ssp_full_dset, rank=rank, num_replicas=world_size, shuffle=True)
        ssp_val_sampler = DistributedSampler(ssp_val_d, rank=rank, num_replicas=world_size, shuffle=False)
        ssp_train_loader = DataLoader(ssp_full_dset, batch_size=config['ssp_batch_size'], sampler=ssp_train_sampler)
        ssp_val_loader = DataLoader(ssp_val_d, batch_size=config['ssp_batch_size'], sampler=ssp_val_sampler)

        # After the split is fixed and identical across ranks, switch to per-rank randomness
        # for masking/jitter/specaugment during SSP training.
        set_seed(base_seed + int(rank))
        # _advance_rng(int(config.get("rng_advance_cuda_before_ssp_training", 0)), device=device)

        if rank == 0: print(f"SSP data split: {len(ssp_full_dset)} total, {len(ssp_val_d)} validation (monitor-only)")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_ddp.parameters()), lr=config['ssp_lr'])
        
        # optimizer = optim.AdamW(model_ddp.parameters(), lr=config['ssp_lr'])
        scheduler_ssp = CosineAnnealingLR(optimizer, T_max=config['ssp_epochs'])
        for epoch in tqdm(range(config['ssp_epochs']), desc="SSP Training", disable=rank!=0):
            train_loss = ssp_epoch(rank, epoch, model_ddp, ssp_train_loader, optimizer, device, config, is_train=True)

            val_loss = ssp_epoch(rank, epoch, model_ddp, ssp_val_loader, None, device, config, is_train=False)

            if rank == 0:
                print(f"SSP Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if logger:
                    logger.log_scalar("SSP/Train_Loss", train_loss, epoch)
                    logger.log_scalar("SSP/Val_Loss", val_loss, epoch)
                    # Optional per-modality SSP losses (train vs val kept separate)
                    if 'ssp_last_train_loss_local_emg' in config:
                        logger.log_scalar("SSP/Train_Loss_EMG", config['ssp_last_train_loss_local_emg'], epoch)
                    if 'ssp_last_train_loss_local_nirs' in config:
                        logger.log_scalar("SSP/Train_Loss_NIRS", config['ssp_last_train_loss_local_nirs'], epoch)
                    if 'ssp_last_train_loss_cosine_nirs' in config:
                        logger.log_scalar("SSP/Train_Loss_Cosine_NIRS", config['ssp_last_train_loss_cosine_nirs'], epoch)

                    if 'ssp_last_val_loss_local_emg' in config:
                        logger.log_scalar("SSP/Val_Loss_EMG", config['ssp_last_val_loss_local_emg'], epoch)
                    if 'ssp_last_val_loss_local_nirs' in config:
                        logger.log_scalar("SSP/Val_Loss_NIRS", config['ssp_last_val_loss_local_nirs'], epoch)
                    if 'ssp_last_val_loss_cosine_emg' in config:
                        logger.log_scalar("SSP/Val_Loss_Cosine_NIRS", config['ssp_last_val_loss_cosine_nirs'], epoch)

                    logger.log_scalar("LR/SSP", scheduler_ssp.get_last_lr()[0], epoch)

            scheduler_ssp.step()

        if rank == 0:

            torch.save(model_ddp.module.state_dict(), args.ssp_weights_path)
            print(f"SSP model saved to {args.ssp_weights_path}")

    # --- Phase 2: Supervised Fine-tuning and Evaluation ---
    elif args.run_supervised:
        if rank == 0: print("\n--- Starting Supervised Fine-tuning Phase ---")
        model = MultimodalModel(config).to(device)
        if (not args.skip_ssp) and os.path.exists(args.ssp_weights_path):
            sd = torch.load(args.ssp_weights_path, map_location=device)
            model_sd = model.state_dict()
            # Drop head params or any mismatched shapes
            filtered = {}
            for k, v in sd.items():
                if k.startswith('main_regressor_head') or k.startswith('transformer') or k.startswith('downsampler_conv1') or k.startswith('downsampler_conv2') or k.startswith('cpc'):
                # if k.startswith('main_regressor_head'):
                    continue
                if k in model_sd and model_sd[k].shape == v.shape:
                    filtered[k] = v
            missing = set(model_sd.keys()) - set(filtered.keys())
            if rank == 0:
                print(f"Loaded {len(filtered)}/{len(model_sd)} keys from {args.ssp_weights_path}. Skipped {len(missing)} (e.g., changed heads).")
            model.load_state_dict(filtered, strict=False)

        else:
            if rank == 0:
                print("No state dict found, training from scratch")
        model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=False)

        train_dataset = PainDataset(train_subjects, config, mode='train')

        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['sup_batch_size'], sampler=train_sampler, collate_fn=collate_fn_supervised)
        val_dataset = PainDataset(val_subjects, config, mode='validation')
        # Validation does not need to be distributed, can run on rank 0
        val_loader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], collate_fn=collate_fn_supervised)
        
        # --- Optimizer: (optional) layer-wise discriminative LR for pretrained encoders ---
        lr_main = float(config.get('sup_lr_main', 5e-5))
        lr_enc_base = float(config.get('sup_lr_encoder', lr_main))

        enc_group_dicts = []
        enc_param_ids = set()

        enc_params = []
        if hasattr(model_ddp.module, 'emg_encoder'):
            enc_params += list(model_ddp.module.emg_encoder.parameters())
        if hasattr(model_ddp.module, 'nirs_encoder'):
            enc_params += list(model_ddp.module.nirs_encoder.parameters())
        enc_params = [p for p in enc_params if p.requires_grad]
        if enc_params:
            enc_group_dicts.append({'params': enc_params, 'lr': lr_enc_base, 'name': 'encoders'})
            enc_param_ids.update(id(p) for p in enc_params)

        other_params = [p for p in model_ddp.parameters() if p.requires_grad and id(p) not in enc_param_ids]
        optimizer_sup = optim.AdamW(
            [{'params': other_params, 'lr': lr_main, 'name': 'main'}] + enc_group_dicts
        )

        scheduler_sup = CosineAnnealingLR(optimizer_sup, T_max=config['sup_epochs'])

        sup_early_stopper = EarlyStopping(patience=config['sup_patience'], verbose=(rank==0), path=args.sup_weights_path)
        for epoch in tqdm(range(config['sup_epochs']), desc="Supervised Fine-tuning", disable=rank!=0):

            train_loss = supervised_train_epoch(rank, epoch, model_ddp, train_loader, optimizer_sup, device, config, logger=logger if rank==0 else None)
            stop_tensor = torch.zeros(1, device=device, dtype=torch.int32)
            if rank == 0:
                val_metrics = evaluate(rank, model_ddp.module, val_loader, device, config)

                reg_type = config.get('regression_loss', 'smooth_l1')
                if reg_type == 'smooth_l1':
                    val_loss = val_metrics['smooth_l1']
                else:
                    val_loss = val_metrics.get('nll', float('inf'))

                loss_name = 'SmoothL1' if config.get('regression_loss','smooth_l1')=='smooth_l1' else 'NLL'
                print(f"Epoch {epoch+1}, Supervised Train Loss: {train_loss:.4f}, Val Loss ({loss_name}): {val_loss:.4f}")

                if logger:
                    logger.log_scalar("Supervised/Train_Loss", train_loss, epoch)
                    logger.log_scalar(f"Supervised/Val_{loss_name}", val_loss, epoch)
                    logger.log_scalar("LR/Main", scheduler_sup.get_last_lr()[0], epoch)
                    # Log distribution parameter means if present
                    if config.get('mode','regression')=='regression' and reg_type != 'smooth_l1':
                        for key, value in val_metrics.items():
                            if key.startswith('param_mean_'):
                                logger.log_scalar(f"Supervised/Val_{key.replace('param_mean_','Param_')}", value, epoch)
                
                if 0 <= epoch:
                    sup_early_stopper(val_loss, model_ddp.module)
                
                if sup_early_stopper.early_stop:
                    stop_tensor.fill_(1)
            if world_size > 1:
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
            
            scheduler_sup.step()

    elif args.run_final_evaluation:
        # --- Final Evaluation ---
        if rank == 0: print("\n--- Starting Final Evaluation on Test Set ---")
        best_model = MultimodalModel(config).to(device)
        best_model.load_state_dict(torch.load(args.sup_weights_path, map_location=device), strict=False)
        test_dataset = PainDataset(test_subjects, config, mode='test')
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if world_size > 1 else None
        test_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], sampler=test_sampler, collate_fn=collate_fn_supervised)
        
        if args.adaptation:
            adaptation_eval_subjects = list(test_subjects)
            adaptation_val_subjects = list(val_subjects) if val_subjects is not None else []
            adaptation_selection_subjects = list(adaptation_val_subjects)
            adaptation_val_source = str(config.get('adaptation_epoch_selection_source', 'transfer_validation_set'))
            if adaptation_val_source == 'transfer_validation_set':
                adaptation_val_subjects, adaptation_eval_subjects = _split_transfer_validation_subjects(
                    test_subjects, config, base_seed, config.get('fold_idx', 0)
                )
                adaptation_selection_subjects = list(adaptation_val_subjects)
                if rank == 0:
                    print(
                        "Using transfer validation subjects for adaptation epoch selection: "
                        f"{len(adaptation_val_subjects)} validation, {len(adaptation_eval_subjects)} held-out test"
                    )
            elif adaptation_val_source == 'trainval':
                adaptation_selection_subjects = list(train_subjects) + list(val_subjects)
                if rank == 0:
                    print(
                        "Using full training-domain subject pool for adaptation epoch selection: "
                        f"{len(adaptation_selection_subjects)} subjects"
                    )
            else:
                raise ValueError(
                    "adaptation_epoch_selection_source must be "
                    "'transfer_validation_set' or 'trainval'."
                )

            adaptation_config = copy.deepcopy(config)
            if bool(config.get('adaptation_epoch_selection_enabled', False)):
                if rank == 0:
                    print("Selecting the optimal number of adaptation epochs from validation performance.")

                selected_adaptation_epochs = select_optimal_adaptation_epochs(
                    rank, best_model, adaptation_selection_subjects, device, adaptation_config
                )
                adaptation_config['adaptation_epochs'] = int(selected_adaptation_epochs)
                adaptation_config['selected_adaptation_epochs'] = int(selected_adaptation_epochs)
                if rank == 0:
                    print(f"Selected adaptation_epochs={selected_adaptation_epochs}")
            else:
                adaptation_config['selected_adaptation_epochs'] = int(adaptation_config.get('adaptation_epochs', 0))

            final_metrics = adapt_and_evaluate(
                rank, best_model, adaptation_eval_subjects, device, adaptation_config,
                return_outputs=False,
                save_prefix="adapted_test_predictions",
                run_adaptation=True,
            )
        else:
            model_to_eval = DDP(best_model, device_ids=[rank]) if world_size > 1 else best_model
            final_metrics = evaluate(rank, model_to_eval, test_loader, device, config, save_predictions=True)

        if rank == 0 and final_metrics is not None:
            print_metrics(final_metrics, header="Final Test Set Results")
            if logger:
                for key, value in final_metrics.items():
                    logger.log_scalar(f"Results/Final_{key.upper()}", value, 0)
                logger.close()

    if world_size > 1: cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a multimodal model for pain prediction.")
    parser.add_argument('--skip_ssp', action='store_true', help="Train from scratch without SSP.")
    parser.add_argument('--ssp_on_all_data', action='store_true', help="Use test data in SSP phase.")
    parser.add_argument('--adaptation', action='store_true', help="Enable test-time adaptation.")
    parser.add_argument('--adaptation_epochs', type=int, default=None,
                        help="Override config['adaptation_epochs'] for test-time adaptation (Phase 4).")
    parser.add_argument('--ssp_subject_limit', type=int, default=None,
                        help="Optional: limit number of SUBJECTS used for SSP training (random subset; validation unchanged).")
    parser.add_argument('--ssp_subject_limit_seed_offset', type=int, default=12345,
                        help="Seed offset used ONLY for SSP subject subsampling when --ssp_subject_limit is set (default: 12345).")
    parser.add_argument('--sup_subject_limit', type=int, default=None,
                        help="Optional: limit number of SUBJECTS used for supervised training (random subset; validation unchanged).")
    parser.add_argument('--sup_subject_limit_seed_offset', type=int, default=17,
                        help="Seed offset used ONLY for supervised subject subsampling when --sup_subject_limit is set (default: 17).")
    parser.add_argument('--log_dir', type=str, default='runs/experiment1', help="Base directory for logs and models for a specific phase.")
    
    # --- Phase Control Arguments ---
    phase_group = parser.add_mutually_exclusive_group(required=True)
    phase_group.add_argument('--run_ssp', action='store_true', help="Run only the SSP phase.")
    phase_group.add_argument('--run_supervised', action='store_true', help="Run only the supervised fine-tuning and evaluation phase.")
    phase_group.add_argument('--run_final_evaluation', action='store_true', help="Run only the final evaluation phase.")
    # --- Weight Path Arguments ---
    parser.add_argument('--ssp_weights_path', type=str, default='runs/ssp_model.pt', help="Path to save/load SSP weights.")
    parser.add_argument('--sup_weights_path', type=str, default='runs/supervised_model.pt', help="Path to save/load supervised weights.")
    # Data Splitting Arguments
    parser.add_argument('--use_separate_test_set', action='store_true', help="Use a pre-defined train/test split instead of K-Fold CV.")
    parser.add_argument('--participants_train', type=str, default=None, help="Path to training participants CSV. Required with --use_separate_test_set.")
    parser.add_argument('--participants_test', type=str, default=None, help="Path to test participants CSV. Required with --use_separate_test_set.")
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for CV. Ignored if --use_separate_test_set.")
    parser.add_argument('--fold_idx', type=int, default=0, help="The specific fold to run for CV.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--master_port', type=int, default=12355, help="Master port for DDP.")
    args = parser.parse_args()
    
    # If launched via torchrun, use env-provided ranks and skip mp.spawn.
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        main_worker(local_rank, world_size, args)
    else:
        world_size = torch.cuda.device_count()
        if world_size > 1:
            try:
                mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
            except KeyboardInterrupt:
                # Best-effort: terminate the whole process group so children don't keep running.
                try:
                    os.killpg(os.getpgid(0), signal.SIGTERM)
                except Exception:
                    pass
                raise
        else:
            main_worker(0, 1, args)