# utils.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, r2_score, balanced_accuracy_score
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import os
import random
import math
import warnings


try:
    from scipy.stats import SpearmanRConstantInputWarning as _SpearmanConstWarn  # type: ignore
except Exception:  # pragma: no cover
    try:
        from scipy.stats._stats_py import SpearmanRConstantInputWarning as _SpearmanConstWarn  # type: ignore
    except Exception:  # pragma: no cover
        class _SpearmanConstWarn(RuntimeWarning):
            pass

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def setup_ddp(rank, world_size, master_port=None):
    """Initializes the distributed process group."""
    # Only set defaults if they are not already provided (e.g. by torchrun).
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if master_port:
        os.environ['MASTER_PORT'] = str(master_port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(12355)
    # Ensure each rank uses a well-defined CUDA device; this also silences
    # warnings about guessing device_id in init_process_group/barrier.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    def close(self):
        self.writer.close()

def calculate_regression_metrics(preds, labels):
    """Calculate regression losses plus Pearson and Spearman correlations."""
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    mse = np.mean((preds_np - labels_np)**2)
    mae = np.mean(np.abs(preds_np - labels_np))
    smooth_l1 = nn.SmoothL1Loss()(preds, labels)

    def safe_corr(x, y):
        # Use nan-aware std and guard against near-constant or NaN-only inputs
        std_x = np.nanstd(x)
        std_y = np.nanstd(y)
        if np.isnan(std_x) or np.isnan(std_y) or std_x < 1e-8 or std_y < 1e-8:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    corr_left = safe_corr(preds_np[:, 0], labels_np[:, 0])
    corr_right = safe_corr(preds_np[:, 1], labels_np[:, 1])

    def safe_spearman(x, y):
        """
        Spearman correlation that:
          - suppresses constant-input warnings
          - treats NaN / undefined results as 0
        We deliberately do NOT short-circuit on small std here, to avoid
        incorrectly forcing valid correlations to 0.
        """
        try:
            x = np.asarray(x).reshape(-1)
            y = np.asarray(y).reshape(-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _SpearmanConstWarn)
                # Use nan-aware policy if available; fall back otherwise.
                try:
                    v = stats.spearmanr(x, y, nan_policy="omit")[0]
                except TypeError:
                    v = stats.spearmanr(x, y)[0]
            return 0.0 if v is None or np.isnan(v) else float(v)
        except Exception:
            return 0.0

    rho_left = safe_spearman(preds_np[:, 0], labels_np[:, 0])
    rho_right = safe_spearman(preds_np[:, 1], labels_np[:, 1])

    return {
        "mse": float(mse),
        "mae": float(mae),
        "smooth_l1": smooth_l1,
        "correlation_left": corr_left,
        "correlation_right": corr_right,
        "spearman_rho_left": rho_left,
        "spearman_rho_right": rho_right,
    }

def print_metrics(metrics_dict, header="Metrics"):
    """Prints a dictionary of metrics in a formatted way."""
    print(f"--- {header} ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")
    print("-" * (len(header) + 8))

def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed to use.
    """

    # 1. Set the seed for Python's built-in random module
    random.seed(seed)

    # 2. Set the seed for NumPy
    np.random.seed(seed)

    # 3. Set the seed for PyTorch on CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.

    # 4. Configure PyTorch to be deterministic
    # This may slow down training, but is crucial for reproducibility.
    # It ensures that the same algorithm is chosen for CUDA convolutions every time.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Set an environment variable for other libraries that might use it
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_label_transform_stats(subjects, epsilon=1e-6):
    stats = {'epsilon': epsilon}
    main_vals, inter_vals = [], []

    for meta in subjects:
        labels = torch.load(meta['label_path'])
        
        # Main labels
        main_pain = labels['post_mvc_pain'].float().reshape(-1, 2)
        main_vals.append(main_pain)

        # Intermediate labels
        inter_pain = labels.get('intermediate_pain')
        if inter_pain is not None and inter_pain.numel() > 0:
            inter_vals.append(inter_pain.float().reshape(-1, 2))

    def get_stats(vals, key):
        if not vals: return
        tensor = torch.cat(vals, dim=0) / 100.0
        log_tensor = torch.log1p(tensor + epsilon)
        stats[key] = {
            'mean': log_tensor.mean(dim=0),
            'std': log_tensor.std(dim=0, unbiased=False).clamp(min=1e-6),
        }

    get_stats(main_vals, 'main')
    get_stats(inter_vals, 'intermediate')

    return stats

def transform_labels(values, stats, key):
    values = values.float()
    if not stats or key not in stats:
        return values / 100.0

    epsilon = stats['epsilon']
    scaled = values / 100.0
    log_vals = torch.log1p(scaled + epsilon)
    mean = stats[key]['mean'].to(log_vals.device)
    std = stats[key]['std'].to(log_vals.device)
    return (log_vals - mean) / std

def inverse_transform_labels(values, stats, key):
    values = values.float()
    if not stats or key not in stats:
        return values * 100.0
        
    epsilon = stats['epsilon']
    mean = stats[key]['mean'].to(values.device)
    std = stats[key]['std'].to(values.device)
    log_vals = values * std + mean
    scaled = torch.expm1(log_vals) - epsilon
    scaled = torch.clamp(scaled, min=0.0)
    return scaled * 100.0

# --- Distributional regression utilities ---
def softplus(x, beta=1.0):
    return torch.nn.functional.softplus(x, beta=beta)

def beta_nll_from_params(params, targets, eps=1e-6):
    """
    params: (N, 4) => [aL, bL, aR, bR] raw
    targets: (N, 2) in [0,1]
    returns scalar NLL averaged over batch and sides
    """
    t = torch.clamp(targets, eps, 1.0 - eps)
    aL = softplus(params[:, 0]) + 1e-3
    bL = softplus(params[:, 1]) + 1e-3
    aR = softplus(params[:, 2]) + 1e-3
    bR = softplus(params[:, 3]) + 1e-3

    def side_nll(y, a, b):
        return - ( (a - 1.0) * torch.log(y) + (b - 1.0) * torch.log(1.0 - y) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)) )

    nll_left = side_nll(t[:, 0], aL, bL)
    nll_right = side_nll(t[:, 1], aR, bR)
    return (nll_left + nll_right).mean()

def gamma_nll_from_params(params, targets, eps=1e-6):
    """
    params: (N, 4) => [kL, rL, kR, rR] raw (shape k>0, rate r>0)
    targets: (N,2) positive
    """
    y = torch.clamp(targets, eps, None)
    kL = softplus(params[:, 0]) + 1e-3
    rL = softplus(params[:, 1]) + 1e-3
    kR = softplus(params[:, 2]) + 1e-3
    rR = softplus(params[:, 3]) + 1e-3

    def side_nll(y, k, r):
        # log pdf = k*log(r) - lgamma(k) + (k-1)*log(y) - r*y
        return -(k * torch.log(r) - torch.lgamma(k) + (k - 1.0) * torch.log(y) - r * y)

    return (side_nll(y[:, 0], kL, rL) + side_nll(y[:, 1], kR, rR)).mean()


def mix2_beta_nll_from_params(params, targets, eps=1e-6):
    """
    params: (N, 10) ...
    targets: (N,2) ...
    kl_weight: The lambda hyperparameter for the regularization strength.
    """
    # --- Parameter extraction (same as before) ---
    t = torch.clamp(targets, eps, 1.0 - eps)
    wL = torch.sigmoid(params[:, 0])

    a1L = softplus(params[:, 1]) + 1e-3; b1L = softplus(params[:, 2]) + 1e-3
    a2L = softplus(params[:, 3]) + 1e-3; b2L = softplus(params[:, 4]) + 1e-3
    wR = torch.sigmoid(params[:, 5])

    a1R = softplus(params[:, 6]) + 1e-3; b1R = softplus(params[:, 7]) + 1e-3
    a2R = softplus(params[:, 8]) + 1e-3; b2R = softplus(params[:, 9]) + 1e-3

    # --- NLL Calculation (same as before) ---
    def side_log_mix_pdf(y, w, a1, b1, a2, b2):
        logB1 = torch.lgamma(a1) + torch.lgamma(b1) - torch.lgamma(a1 + b1)
        logB2 = torch.lgamma(a2) + torch.lgamma(b2) - torch.lgamma(a2 + b2)
        logp1 = -logB1 + (a1 - 1.0) * torch.log(y) + (b1 - 1.0) * torch.log(1.0 - y)
        logp2 = -logB2 + (a2 - 1.0) * torch.log(y) + (b2 - 1.0) * torch.log(1.0 - y)
        m1 = torch.log(w + eps) + logp1
        m2 = torch.log(1.0 - w + eps) + logp2
        m = torch.maximum(m1, m2)
        return m + torch.log(torch.exp(m1 - m) + torch.exp(m2 - m))

    logpL = side_log_mix_pdf(t[:, 0], wL, a1L, b1L, a2L, b2L)
    logpR = side_log_mix_pdf(t[:, 1], wR, a1R, b1R, a2R, b2R)
    total_loss = (-(logpL + logpR)).mean()

    return total_loss

def mix2_gamma_nll_from_params(params, targets, eps=1e-6):
    """
    params: (N, 10) per sample [wL, k1L, r1L, k2L, r2L, wR, k1R, r1R, k2R, r2R] raw
    targets: (N,2) positive
    """
    y = torch.clamp(targets, eps, None)
    wL = torch.sigmoid(params[:, 0])
    k1L = softplus(params[:, 1]) + 1e-3; r1L = softplus(params[:, 2]) + 1e-3
    k2L = softplus(params[:, 3]) + 1e-3; r2L = softplus(params[:, 4]) + 1e-3
    wR = torch.sigmoid(params[:, 5])
    k1R = softplus(params[:, 6]) + 1e-3; r1R = softplus(params[:, 7]) + 1e-3
    k2R = softplus(params[:, 8]) + 1e-3; r2R = softplus(params[:, 9]) + 1e-3

    def side_log_gamma_pdf(y, k, r):
        return k * torch.log(r) - torch.lgamma(k) + (k - 1.0) * torch.log(y) - r * y

    def side_log_mix_pdf(y, w, k1, r1, k2, r2):
        logp1 = side_log_gamma_pdf(y, k1, r1)
        logp2 = side_log_gamma_pdf(y, k2, r2)
        m1 = torch.log(w + eps) + logp1
        m2 = torch.log(1.0 - w + eps) + logp2
        m = torch.maximum(m1, m2)
        return m + torch.log(torch.exp(m1 - m) + torch.exp(m2 - m))

    logpL = side_log_mix_pdf(y[:, 0], wL, k1L, r1L, k2L, r2L)
    logpR = side_log_mix_pdf(y[:, 1], wR, k1R, r1R, k2R, r2R)
    return (-(logpL + logpR)).mean()

def params_to_means(params, loss_type):
    """Convert parameter outputs to mean predictions in [0,1] for metrics.
    params: tensor [N, D]
    returns tensor [N, 2] means for left/right
    """
    if loss_type == 'beta_nll':
        aL = softplus(params[:, 0]) + 1e-3
        bL = softplus(params[:, 1]) + 1e-3
        aR = softplus(params[:, 2]) + 1e-3
        bR = softplus(params[:, 3]) + 1e-3
        mL = aL / (aL + bL)
        mR = aR / (aR + bR)
        return torch.stack([mL, mR], dim=1)
    if loss_type == 'gamma_nll':
        kL = softplus(params[:, 0]) + 1e-3
        rL = softplus(params[:, 1]) + 1e-3
        kR = softplus(params[:, 2]) + 1e-3
        rR = softplus(params[:, 3]) + 1e-3
        mL = kL / rL
        mR = kR / rR
        return torch.stack([mL, mR], dim=1)
    if loss_type == 'beta2_nll':
        wL = torch.sigmoid(params[:, 0]); a1L = softplus(params[:, 1]) + 1e-3; b1L = softplus(params[:, 2]) + 1e-3; a2L = softplus(params[:, 3]) + 1e-3; b2L = softplus(params[:, 4]) + 1e-3
        wR = torch.sigmoid(params[:, 5]); a1R = softplus(params[:, 6]) + 1e-3; b1R = softplus(params[:, 7]) + 1e-3; a2R = softplus(params[:, 8]) + 1e-3; b2R = softplus(params[:, 9]) + 1e-3
        mL = wL * (a1L / (a1L + b1L)) + (1.0 - wL) * (a2L / (a2L + b2L))
        mR = wR * (a1R / (a1R + b1R)) + (1.0 - wR) * (a2R / (a2R + b2R))

        return torch.stack([mL, mR], dim=1)
    if loss_type == 'gamma2_nll':
        wL = torch.sigmoid(params[:, 0]); k1L = softplus(params[:, 1]) + 1e-3; r1L = softplus(params[:, 2]) + 1e-3; k2L = softplus(params[:, 3]) + 1e-3; r2L = softplus(params[:, 4]) + 1e-3
        wR = torch.sigmoid(params[:, 5]); k1R = softplus(params[:, 6]) + 1e-3; r1R = softplus(params[:, 7]) + 1e-3; k2R = softplus(params[:, 8]) + 1e-3; r2R = softplus(params[:, 9]) + 1e-3
        mL = wL * (k1L / r1L) + (1.0 - wL) * (k2L / r2L)
        mR = wR * (k1R / r1R) + (1.0 - wR) * (k2R / r2R)
        return torch.stack([mL, mR], dim=1)
    raise ValueError(f"Unknown loss_type {loss_type}")

def extract_distribution_param_means(params: torch.Tensor, loss_type: str, eps: float = 1e-6) -> dict:
    """
    Compute interpretable distribution parameter means from raw model outputs for logging.
    Returns a dict of scalar floats keyed by parameter names.
    Supported loss_type: 'beta_nll', 'gamma_nll', 'beta2_nll', 'gamma2_nll'
    """
    out = {}
    with torch.no_grad():
        if loss_type == 'beta_nll':
            aL = softplus(params[:, 0]) + 1e-3
            bL = softplus(params[:, 1]) + 1e-3
            aR = softplus(params[:, 2]) + 1e-3
            bR = softplus(params[:, 3]) + 1e-3
            out['aL'] = float(aL.mean().item())
            out['bL'] = float(bL.mean().item())
            out['aR'] = float(aR.mean().item())
            out['bR'] = float(bR.mean().item())
        elif loss_type == 'gamma_nll':
            kL = softplus(params[:, 0]) + 1e-3
            rL = softplus(params[:, 1]) + 1e-3
            kR = softplus(params[:, 2]) + 1e-3
            rR = softplus(params[:, 3]) + 1e-3
            out['kL'] = float(kL.mean().item())
            out['rL'] = float(rL.mean().item())
            out['kR'] = float(kR.mean().item())
            out['rR'] = float(rR.mean().item())
        elif loss_type == 'beta2_nll':
            wL = torch.sigmoid(params[:, 0])
            a1L = softplus(params[:, 1]) + 1e-3; b1L = softplus(params[:, 2]) + 1e-3
            a2L = softplus(params[:, 3]) + 1e-3; b2L = softplus(params[:, 4]) + 1e-3
            wR = torch.sigmoid(params[:, 5])
            a1R = softplus(params[:, 6]) + 1e-3; b1R = softplus(params[:, 7]) + 1e-3
            a2R = softplus(params[:, 8]) + 1e-3; b2R = softplus(params[:, 9]) + 1e-3
            out.update({
                'wL': float(wL.mean().item()), 'a1L': float(a1L.mean().item()), 'b1L': float(b1L.mean().item()),
                'a2L': float(a2L.mean().item()), 'b2L': float(b2L.mean().item()),
                'wR': float(wR.mean().item()), 'a1R': float(a1R.mean().item()), 'b1R': float(b1R.mean().item()),
                'a2R': float(a2R.mean().item()), 'b2R': float(b2R.mean().item()),
            })
        elif loss_type == 'gamma2_nll':
            wL = torch.sigmoid(params[:, 0])
            k1L = softplus(params[:, 1]) + 1e-3; r1L = softplus(params[:, 2]) + 1e-3
            k2L = softplus(params[:, 3]) + 1e-3; r2L = softplus(params[:, 4]) + 1e-3
            wR = torch.sigmoid(params[:, 5])
            k1R = softplus(params[:, 6]) + 1e-3; r1R = softplus(params[:, 7]) + 1e-3
            k2R = softplus(params[:, 8]) + 1e-3; r2R = softplus(params[:, 9]) + 1e-3
            out.update({
                'wL': float(wL.mean().item()), 'k1L': float(k1L.mean().item()), 'r1L': float(r1L.mean().item()),
                'k2L': float(k2L.mean().item()), 'r2L': float(r2L.mean().item()),
                'wR': float(wR.mean().item()), 'k1R': float(k1R.mean().item()), 'r1R': float(r1R.mean().item()),
                'k2R': float(k2R.mean().item()), 'r2R': float(r2R.mean().item()),
            })
    return out

def extract_distribution_params_per_sample(params: torch.Tensor, loss_type: str, eps: float = 1e-6) -> dict:
    """
    Transform raw model outputs (per-sample) into interpretable distribution parameters.
    Returns a dict mapping parameter names to tensors of shape [N].
    """
    if loss_type == 'beta_nll':
        aL = softplus(params[:, 0]) + 1e-3
        bL = softplus(params[:, 1]) + 1e-3
        aR = softplus(params[:, 2]) + 1e-3
        bR = softplus(params[:, 3]) + 1e-3
        return {'aL': aL, 'bL': bL, 'aR': aR, 'bR': bR}
    if loss_type == 'gamma_nll':
        kL = softplus(params[:, 0]) + 1e-3
        rL = softplus(params[:, 1]) + 1e-3
        kR = softplus(params[:, 2]) + 1e-3
        rR = softplus(params[:, 3]) + 1e-3
        return {'kL': kL, 'rL': rL, 'kR': kR, 'rR': rR}
    if loss_type == 'beta2_nll':
        wL = torch.sigmoid(params[:, 0])
        a1L = softplus(params[:, 1]) + 1e-3; b1L = softplus(params[:, 2]) + 1e-3
        a2L = softplus(params[:, 3]) + 1e-3; b2L = softplus(params[:, 4]) + 1e-3
        wR = torch.sigmoid(params[:, 5])
        a1R = softplus(params[:, 6]) + 1e-3; b1R = softplus(params[:, 7]) + 1e-3
        a2R = softplus(params[:, 8]) + 1e-3; b2R = softplus(params[:, 9]) + 1e-3
        return {
            'wL': wL, 'a1L': a1L, 'b1L': b1L, 'a2L': a2L, 'b2L': b2L,
            'wR': wR, 'a1R': a1R, 'b1R': b1R, 'a2R': a2R, 'b2R': b2R
        }
    if loss_type == 'gamma2_nll':
        wL = torch.sigmoid(params[:, 0])
        k1L = softplus(params[:, 1]) + 1e-3; r1L = softplus(params[:, 2]) + 1e-3
        k2L = softplus(params[:, 3]) + 1e-3; r2L = softplus(params[:, 4]) + 1e-3
        wR = torch.sigmoid(params[:, 5])
        k1R = softplus(params[:, 6]) + 1e-3; r1R = softplus(params[:, 7]) + 1e-3
        k2R = softplus(params[:, 8]) + 1e-3; r2R = softplus(params[:, 9]) + 1e-3
        return {
            'wL': wL, 'k1L': k1L, 'r1L': r1L, 'k2L': k2L, 'r2L': r2L,
            'wR': wR, 'k1R': k1R, 'r1R': r1R, 'k2R': k2R, 'r2R': r2R
        }
    return {}