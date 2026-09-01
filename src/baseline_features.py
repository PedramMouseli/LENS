import os
import h5py
import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import linregress

def compute_sample_entropy(x, order=2, r=0.2):
    """Computes sample entropy with Chebyshev distance."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 50:
        return 0.0
    if N > 2000:
        step = int(np.ceil(N / 2000))
        x = x[::step]
    std_x = np.std(x)
    if std_x == 0:
        return 0.0
    r_val = r * std_x
    
    emb2 = np.lib.stride_tricks.sliding_window_view(x, 2)
    emb3 = np.lib.stride_tricks.sliding_window_view(x, 3)
    
    d2 = np.max(np.abs(emb2[:, None, :] - emb2[None, :, :]), axis=-1)
    c2 = np.sum(d2 <= r_val) - len(emb2)
    
    d3 = np.max(np.abs(emb3[:, None, :] - emb3[None, :, :]), axis=-1)
    c3 = np.sum(d3 <= r_val) - len(emb3)
    
    if c2 == 0 or c3 == 0:
        return 0.0
    return -np.log(c3 / c2)

def compute_spectral_features(x, fs):
    """Computes MNF, MDF, high-frequency power ratio, and Dimitrov FInsm5."""
    nperseg = min(len(x), int(fs * 0.5))
    f, psd = welch(x, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0, 0.0, 0.0, 0.0

    mnf = np.sum(f * psd) / total_power
    cum_power = np.cumsum(psd)
    mdf_idx = np.where(cum_power >= total_power / 2.0)[0]
    mdf = f[mdf_idx[0]] if len(mdf_idx) > 0 else 0.0

    # High-frequency power ratio: P(150-400 Hz) / P(10-500 Hz)
    emg_band = (f >= 10) & (f <= 500)
    hf_band = (f >= 150) & (f <= 400)
    emg_power = np.sum(psd[emg_band])
    hf_ratio = np.sum(psd[hf_band]) / (emg_power + 1e-30)

    # Dimitrov fatigue index FInsm5 = M_{-1} / M_5 over 5-500 Hz
    # Dimitrov et al., J Electromyogr Kinesiol (2006)
    dim_band = (f >= 5) & (f <= 500)
    f_dim = f[dim_band]
    psd_dim = psd[dim_band]
    m_neg1 = np.sum(psd_dim * f_dim**(-1))
    m_5 = np.sum(psd_dim * f_dim**5)
    dimitrov_index = m_neg1 / (m_5 + 1e-30)

    return mnf, mdf, hf_ratio, dimitrov_index

def compute_dwt_powers(x, wavelet='db5'):
    """Computes power across 9 DWT decomposition levels (levels 4 to 12)."""
    coeffs = pywt.wavedec(x, wavelet)
    powers = []
    for i in range(min(4, len(coeffs)), min(13, len(coeffs))):
        powers.append(np.sum(coeffs[i]**2))
    while len(powers) < 9:
        powers.append(0.0)
    return np.array(powers, dtype=np.float64)

def extract_subject_features(hdf5_path):
    """
    Extracts pre-registered PLS features and classical EMG/NIRS features
    for left and right sides from a single raw HDF5 subject file.
    
    Returns:
        dict: Features and labels for 'left' and 'right' sides.
    """
    with h5py.File(hdf5_path, 'r') as f:
        emg = f['emg'][:]
        nirs = f['nirs'][:]
        device_type = f['device_type'][()].decode('utf-8')
        labels_grp = f['labels']
        post_mvc_pain = labels_grp['post_mvc_pain'][:]  # [left, right]
        indices = labels_grp['intermediate_pain_indices'][:]
        task_phase_ids = labels_grp['task_phase_ids'][:]

    device_number = 1 if device_type == 'A' else 2
    sf_emg = 2048.0 if device_type == 'A' else 2000.0
    sf_nirs = 50.0

    # NIRS TSI indices: 12 (left), 13 (right)
    nirs_l = nirs[:, 12]
    nirs_r = nirs[:, 13]
    emg_l = emg[:, 0]
    emg_r = emg[:, 1]

    # Baseline StO2 (first minute before trial 1 clench start)
    first_clench_start_nirs = indices[0] - int(60 * sf_nirs)
    base_nirs_l = np.median(nirs_l[:max(1, first_clench_start_nirs)])
    base_nirs_r = np.median(nirs_r[:max(1, first_clench_start_nirs)])

    # MVC segment
    mvc_nirs_indices = np.where(task_phase_ids == 2)[0]
    if len(mvc_nirs_indices) > 0:
        mvc_t_start = mvc_nirs_indices[0] / sf_nirs
        mvc_t_end = mvc_nirs_indices[-1] / sf_nirs
        emg_mvc_l = emg_l[int(mvc_t_start * sf_emg):int(mvc_t_end * sf_emg)]
        emg_mvc_r = emg_r[int(mvc_t_start * sf_emg):int(mvc_t_end * sf_emg)]
    else:
        emg_mvc_l, emg_mvc_r = emg_l[-int(5*sf_emg):], emg_r[-int(5*sf_emg):]

    def _extract_side_features(emg_sig, nirs_sig, base_nirs, emg_mvc):
        # Baseline-normalize NIRS (subtract pre-task baseline median)
        nirs_sig = nirs_sig - base_nirs

        task_medians = np.zeros(15)
        rest_medians = np.zeros(15)
        samp_entropies = np.zeros(15)
        dwt_powers = np.zeros((15, 9))
        rms_vals = np.zeros(15)
        mdf_vals = np.zeros(15)
        mnf_vals = np.zeros(15)
        hf_ratios = np.zeros(15)
        dimitrov_indices = np.zeros(15)
        zcr_vals = np.zeros(15)

        for i in range(15):
            idx = indices[i]
            # Trial timestamps
            t_clench_start = (idx - 60 * sf_nirs) / sf_nirs
            t_clench_end = (idx - 30 * sf_nirs) / sf_nirs

            # NIRS clench & rest (with 5s trim)
            n_clench_start = int(idx - 60 * sf_nirs + 5 * sf_nirs)
            n_clench_end = int(idx - 30 * sf_nirs - 5 * sf_nirs)
            n_rest_start = int(idx - 30 * sf_nirs + 5 * sf_nirs)
            n_rest_end = int(idx - 5 * sf_nirs)

            task_medians[i] = np.median(nirs_sig[n_clench_start:n_clench_end])
            rest_medians[i] = np.median(nirs_sig[n_rest_start:n_rest_end])

            # EMG clench (with 5s trim)
            e_start = int((t_clench_start + 5.0) * sf_emg)
            e_end = int((t_clench_end - 5.0) * sf_emg)
            clench_emg = emg_sig[e_start:e_end]

            # Time domain & entropy
            rms_vals[i] = np.sqrt(np.mean(clench_emg**2))
            zcr_vals[i] = np.sum(np.diff(clench_emg > 0) != 0) / (len(clench_emg) / sf_emg)
            samp_entropies[i] = compute_sample_entropy(clench_emg)

            # Spectral & Wavelet
            mnf_vals[i], mdf_vals[i], hf_ratios[i], dimitrov_indices[i] = compute_spectral_features(clench_emg, sf_emg)
            dwt_powers[i, :] = compute_dwt_powers(clench_emg)

        # Feature trends across 15 trials
        x_trials = np.arange(15)
        diff_nirs = task_medians - rest_medians
        nirs_slope = linregress(x_trials, diff_nirs).slope
        entropy_slope = linregress(x_trials / 100.0, samp_entropies).slope
        wt_slopes = [linregress(x_trials / 100.0, dwt_powers[:, k]).slope for k in range(9)]
        mdf_slope = linregress(x_trials, mdf_vals).slope
        rms_slope = linregress(x_trials, rms_vals).slope

        # --- Pre-registered PLS 5 Features ---
        pls_feat = np.array([
            np.abs(samp_entropies[14] - samp_entropies[0]),
            np.sign(nirs_slope),
            np.sign(wt_slopes[8]),  # Level 1 detail slope sign
            np.sign(entropy_slope),
            np.mean(task_medians[3:6])  # already baseline-normalized
        ], dtype=np.float32)

        # --- Spectral Shift Baseline Features ---
        spec_shift_feat = np.array([
            mdf_slope,
            mdf_vals[14] - mdf_vals[0],
            np.mean(mdf_vals),
            compute_spectral_features(emg_mvc, sf_emg)[1] if len(emg_mvc) > 0 else np.mean(mdf_vals)
        ], dtype=np.float32)

        # --- Full Classical Feature Set (18 Features) ---
        classical_feat = np.concatenate([
            [np.mean(rms_vals), rms_slope, np.sqrt(np.mean(emg_mvc**2)) if len(emg_mvc) > 0 else np.mean(rms_vals), np.mean(zcr_vals)],
            [np.mean(mnf_vals), np.mean(mdf_vals), mdf_slope, np.mean(hf_ratios), np.mean(dimitrov_indices)],
            [np.mean(samp_entropies), entropy_slope],
            wt_slopes,
            [base_nirs, nirs_slope, np.mean(task_medians[3:6])]  # base_nirs is raw absolute; medians already normalized
        ]).astype(np.float32)

        return pls_feat, spec_shift_feat, classical_feat, device_number

    pls_l, spec_l, class_l, device_l = _extract_side_features(emg_l, nirs_l, base_nirs_l, emg_mvc_l)
    pls_r, spec_r, class_r, device_r = _extract_side_features(emg_r, nirs_r, base_nirs_r, emg_mvc_r)

    return {
        'left': {
            'pls_features': pls_l,
            'spectral_features': spec_l,
            'classical_features': class_l,
            'target': post_mvc_pain[0]
        },
        'right': {
            'pls_features': pls_r,
            'spectral_features': spec_r,
            'classical_features': class_r,
            'target': post_mvc_pain[1]
        },
        'device': device_l
    }
