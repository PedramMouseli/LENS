# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torch.nn.functional as F
from src.utils import transform_labels

class PainDataset(Dataset):
    """
    PyTorch Dataset for the pain prediction task.
    Handles different modes for SSP and supervised training.
    """
    def __init__(self, data_list, config, mode='ssp'):
        """
        Args:
            data_list (list): A list of dictionaries, where each dict contains
                              paths to processed features and labels for a subject.
            config (dict): Global configuration dictionary.
            mode (str): 'ssp', 'train', 'validation', or 'test'.
        """
        self.data_list = data_list
        self.config = config
        self.mode = mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_data = self.data_list[idx]
        
        # Load pre-computed features
        features = torch.load(subject_data['feature_path'])
        emg_seq = features['emg']
        nirs_seq = features['nirs']
        
        if self.mode == 'ssp':
            # For SSP, we sample a random chunk from the full sequence
            chunk_len = self.config['ssp_context_len']
            start_idx = random.randint(0, emg_seq.shape[0] - chunk_len)
            emg_chunk = emg_seq[start_idx : start_idx + chunk_len]
            nirs_chunk = nirs_seq[start_idx : start_idx + chunk_len]
            return emg_chunk, nirs_chunk
        
        else: # For supervised 'train', 'validation', or 'test'
            # Load labels
            labels = torch.load(subject_data['label_path'])
            stats = self.config.get('label_transform_stats')

            post_mvc_pain = labels['post_mvc_pain'].float()
            post_mvc_pain = transform_labels(post_mvc_pain, stats, 'main')

            intermediate_pain = labels.get('intermediate_pain', torch.empty(0)).float()
            if intermediate_pain.numel() > 0:
                intermediate_pain = transform_labels(intermediate_pain, stats, 'intermediate')
            
            # Load pre-computed task phase IDs from the label file
            task_phase_ids = labels.get('task_phase_ids', torch.empty(0))
            
            sample = {
                'emg': emg_seq,
                'nirs': nirs_seq,
                'task_phase_ids': task_phase_ids,
                'post_mvc_pain': post_mvc_pain,
                'intermediate_pain': intermediate_pain,
                'intermediate_pain_indices': labels.get('intermediate_pain_indices', torch.empty(0)),
                'device_id': torch.tensor(0, dtype=torch.long), # default unknown
                'subject_id': subject_data.get('id', ''),
            }
            # Map device string to small integer id within [0, num_devices-1]
            # If num_devices == 2 (A/B), map: unknown->0, A->0, B->1
            # If num_devices >= 3, map: unknown->0, A->1, B->2
            dev = labels.get('device', None)
            num_devices = int(self.config.get('num_devices', 3))
            if dev is not None:
                dev_str = str(dev) if not isinstance(dev, torch.Tensor) else str(dev)
                dev_u = dev_str.strip().upper()
                sample['device_str'] = dev_str
                if num_devices <= 2:
                    # Binary device embedding: bucket A with unknown as 0, B as 1
                    if dev_u.endswith('B') or dev_u == 'B':
                        sample['device_id'] = torch.tensor(1, dtype=torch.long)
                    else:
                        sample['device_id'] = torch.tensor(0, dtype=torch.long)
                else:
                    # Ternary: 0 unknown, 1 A, 2 B
                    if dev_u.endswith('A') or dev_u == 'A':
                        sample['device_id'] = torch.tensor(1, dtype=torch.long)
                    elif dev_u.endswith('B') or dev_u == 'B':
                        sample['device_id'] = torch.tensor(2, dtype=torch.long)
                    else:
                        sample['device_id'] = torch.tensor(0, dtype=torch.long)
            # Optional classification label (e.g., TMD vs Control)
            class_label = labels.get('class_label', None)
            if class_label is not None:
                # Expect a scalar or 1D tensor
                if isinstance(class_label, torch.Tensor):
                    sample['class_label'] = class_label.long()
                else:
                    sample['class_label'] = torch.tensor(class_label).long()
            return sample

    def get_task_phase_ids(self, seq_len):
        """
        Deprecated helper retained for older data paths.

        Supervised runs load `task_phase_ids` from the preprocessed label files.
        """
        fs = self.config['target_sample_rate']
        phase_ids = torch.zeros(seq_len, dtype=torch.long)
        phase_ids[:2 * 60 * fs] = 0
        return phase_ids

def collate_fn_supervised(batch):
    """
    Custom collate function for supervised training to handle variable length sequences.
    It will pad all sequences in a batch to the length of the longest sequence.
    """
    # Find max length in batch
    max_len = max([s['emg'].shape[0] for s in batch])
    batch_size = len(batch)
    
    padded_emg = []
    padded_nirs = []
    padded_task_phases = []
    post_mvc_labels = []
    seq_lengths = []
    # Store intermediate labels and their indices
    intermediate_labels = []
    intermediate_indices = []
    class_labels = []
    device_ids = []
    subject_ids = []
    device_strs = []
    
    for item in batch:
        emg, nirs, phases = item['emg'], item['nirs'], item['task_phase_ids']
        pad_len = max_len - emg.shape[0]
        seq_lengths.append(emg.shape[0])
        
        padded_emg.append(F.pad(emg, (0, 0, 0, pad_len), 'constant', 0))
        padded_nirs.append(F.pad(nirs, (0, 0, 0, pad_len), 'constant', 0))
        padded_task_phases.append(F.pad(phases, (0, pad_len), 'constant', 0)) # Pad with 0 (rest)
        
        post_mvc_labels.append(item['post_mvc_pain'])
        intermediate_labels.append(item['intermediate_pain'])
        intermediate_indices.append(item['intermediate_pain_indices'])
        if 'class_label' in item:
            class_labels.append(item['class_label'])
        if 'device_id' in item:
            device_ids.append(item['device_id'])
        if 'subject_id' in item:
            subject_ids.append(item['subject_id'])
        if 'device_str' in item:
            device_strs.append(item['device_str'])

    lengths = torch.tensor(seq_lengths, dtype=torch.long)
    padding_mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    
    out = {
        'emg': torch.stack(padded_emg),
        'nirs': torch.stack(padded_nirs),
        'task_phase_ids': torch.stack(padded_task_phases),
        'post_mvc_pain': torch.stack(post_mvc_labels),
        'intermediate_pain': intermediate_labels, # Keep as list of tensors
        'intermediate_pain_indices': intermediate_indices, # Keep as list of tensors
        'padding_mask': padding_mask,
        'lengths': lengths,
    }
    if len(class_labels) == len(batch) and len(class_labels) > 0:
        out['class_label'] = torch.stack([cl if isinstance(cl, torch.Tensor) else torch.tensor(cl) for cl in class_labels])
    if len(device_ids) == len(batch) and len(device_ids) > 0:
        out['device_id'] = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d) for d in device_ids])
    # keep metadata lists as-is
    out['subject_ids'] = subject_ids
    out['device_strs'] = device_strs
    return out