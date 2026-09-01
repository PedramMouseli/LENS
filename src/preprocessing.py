import numpy as np
import torch
from scipy.signal import butter, filtfilt, spectrogram, resample
from scipy.linalg import logm
import h5py
import argparse
import os

# --- Configuration Constants ---
# Common sampling rate to standardize all EMG data
COMMON_EMG_SAMPLE_RATE = 2000
NIRS_SAMPLE_RATE = 50

# --- Filtering ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

# --- Resampling ---
def resample_emg(emg_data, original_fs, target_fs):
    """Resamples EMG data to a common sampling rate using scipy."""
    if original_fs == target_fs:
        return emg_data
    
    num_samples = int(emg_data.shape[0] * target_fs / original_fs)
    resampled_data = resample(emg_data, num_samples, axis=0)
    return resampled_data

# --- EMG Feature Extraction ---
def extract_log_spectrogram(emg_data, fs, window_size_ms=32, stride_ms=20):
    """Computes log-scaled spectrograms for EMG channels."""
    nperseg = int(window_size_ms / 1000 * fs)
    noverlap = nperseg - int(stride_ms / 1000 * fs)
    
    log_specs = []
    for i in range(emg_data.shape[1]): # Iterate over channels (left/right)
        f, t, Sxx = spectrogram(emg_data[:, i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        log_spec = np.log(Sxx + 1e-10) # Add epsilon for numerical stability
        log_specs.append(log_spec)
        
    return np.concatenate(log_specs, axis=0).T # Shape: [Time, FreqBins * Channels]

# --- NIRS Feature Extraction ---
def extract_nirs_features(nirs_data, fs=NIRS_SAMPLE_RATE, add_derivatives=True):
    """Normalizes NIRS against a baseline and optionally adds derivatives."""
    baseline_end_idx = 2 * 60 * fs
    if nirs_data.shape[0] < baseline_end_idx:
        print("Warning: NIRS data is shorter than the expected 2-minute baseline.")
        baseline_end_idx = nirs_data.shape[0] // 4 # Use first quarter as baseline

    baseline = np.mean(nirs_data[:baseline_end_idx], axis=0)
    normalized_nirs = nirs_data - baseline
    
    if not add_derivatives:
        return normalized_nirs
        
    derivatives = np.gradient(normalized_nirs, axis=0)
    return np.concatenate([normalized_nirs, derivatives], axis=1)

# --- Main Processing Function ---
def process_and_save(args):
    """
    Main function to load raw data, process it into feature sets, and save them.
    """
    # --- 1. Load Data ---
    try:
        with h5py.File(args.input_path, 'r') as f:
            emg = f['emg'][:]      # Expects shape [Time, 2]
            nirs = f['nirs'][:]    # Expects shape [Time, 6]
            device_type = f['device_type'][()].decode('utf-8') # Load and decode device type
            
            # Load all labels from the HDF5 file
            labels_grp = f['labels']
            post_mvc_pain = labels_grp['post_mvc_pain'][:]
            # fatigue removed
            intermediate_pain = labels_grp['intermediate_pain'][:]
            # fatigue removed
            intermediate_pain_indices = labels_grp['intermediate_pain_indices'][:]
            task_phase_ids = labels_grp['task_phase_ids'][:]
            class_label = labels_grp['class_label'][()] if 'class_label' in labels_grp else None

    except Exception as e:
        print(f"Error loading data from {args.input_path}: {e}")
        return

    # --- 2. Determine EMG Sampling Rate from stored device type ---
    if device_type == 'A':
        original_emg_fs = 2048
    elif device_type == 'B':
        original_emg_fs = 2000
    else:
        raise ValueError(f"Unknown EMG device type '{device_type}' in file {args.input_path}")
    
    # --- 3. Process EMG ---
    emg_resampled = resample_emg(emg, original_fs=original_emg_fs, target_fs=COMMON_EMG_SAMPLE_RATE)
    emg_filtered = bandpass_filter(emg_resampled, lowcut=40, highcut=850, fs=COMMON_EMG_SAMPLE_RATE)
    
    # Extract log-spectrogram features
    emg_features = extract_log_spectrogram(emg_filtered, fs=COMMON_EMG_SAMPLE_RATE)


    # --- 4. Process NIRS ---
    nirs_features = extract_nirs_features(nirs, fs=NIRS_SAMPLE_RATE, add_derivatives=True)

    # --- 5. Align Features and Labels ---
    # The feature extraction stride determines the downsampling factor
    stride_s = 20 / 1000  # Stride in seconds from config
    nirs_fs = NIRS_SAMPLE_RATE
    # Note: This downsampling factor assumes NIRS_SAMPLE_RATE (50Hz) and a feature stride of 20ms.
    # This results in a downsampling factor of 1 (50 * 0.020), meaning features are generated at 50Hz.
    # If your feature extraction parameters change, this calculation must be updated.
    downsampling_factor = int(nirs_fs * stride_s)
    
    # Downsample task_phase_ids by taking the first ID in each window frame
    task_phase_ids_downsampled = task_phase_ids[::downsampling_factor]

    # Align all sequences to the minimum length after feature extraction
    min_len = min(
        emg_features.shape[0],
        nirs_features.shape[0],
        len(task_phase_ids_downsampled)
    )
    
    features_extracted = {
        'emg': torch.from_numpy(emg_features[:min_len]).float(),
        'nirs': torch.from_numpy(nirs_features[:min_len]).float()
    }

    # Prepare labels for saving, ensuring indices are also aligned
    labels_to_save = {
        'post_mvc_pain': torch.from_numpy(post_mvc_pain).float(),
        'intermediate_pain': torch.from_numpy(intermediate_pain).float(),
        'intermediate_pain_indices': torch.from_numpy(intermediate_pain_indices // downsampling_factor).long(),
        'task_phase_ids': torch.from_numpy(task_phase_ids_downsampled[:min_len]).long(),
        'device': device_type
    }
    if class_label is not None:
        labels_to_save['class_label'] = torch.tensor(class_label, dtype=torch.long)

    # --- 6. Save to Output Directory ---
    subject_id = os.path.splitext(os.path.basename(args.input_path))[0]
    
    output_path = os.path.join(args.output_dir, f"{subject_id}_features_logspec.pt")
    torch.save(features_extracted, output_path)
    
    # Save the corresponding label file
    label_path = os.path.join(args.output_dir, f"{subject_id}_labels.pt")
    torch.save(labels_to_save, label_path)

    print(f"Successfully processed and saved features and labels for {subject_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess EMG and NIRS data for pain prediction.")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the raw HDF5 data file for a single subject.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the processed feature tensors.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_and_save(args)