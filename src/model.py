# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Use absolute imports starting from the project root ('src')
from src.modules.encoders import TDSConvNetEncoder
from src.modules.projection import encoder_projection
from src.modules.positional_encoding import PositionalEncoding
from src.augmentations import SpecAugment, Jitter
from src.ssp_utils import perform_masking

class MultimodalModel(nn.Module):
    """
    Hierarchical multimodal model for SSP pretraining and supervised pain prediction.
    The pipeline is local encoders -> projection -> temporal downsampling -> Conformer.
    """
    def __init__(self, config, build_mode: str = "full"):
        super(MultimodalModel, self).__init__()
        self.config = config
        self.build_mode = str(build_mode or "full").lower()

        # --- Augmentation Modules (applied in the training loop) ---
        self.emg_spec_augment = SpecAugment(**config['emg_spec_augment_params'])
        self.nirs_jitter = Jitter(**config['nirs_jitter_params'])

        # CLS token dimension must match the global sequence model input.
        transformer_input_dim = config['conformer_params']['input_dim'] + config['num_task_phases']
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_input_dim))

        # --- Initial normalization ---
        self.emg_norm = nn.LayerNorm(config['emg_feature_dim'])
        self.nirs_norm = nn.LayerNorm(config['nirs_feature_dim'])

        # --- Modality-specific local encoders ---
        self.emg_encoder = TDSConvNetEncoder(
            input_dim=config['emg_feature_dim'],
            **config['emg_encoder_params']
        )
        self.nirs_encoder = TDSConvNetEncoder(
            input_dim=config['nirs_feature_dim'],
            **config['nirs_encoder_params']
        )
        
        # --- Encoder projection ---
        emg_enc_dim = config['emg_encoder_params']['hidden_dim']
        nirs_enc_dim = config['nirs_encoder_params']['hidden_dim']

        projection_hidden_dim = config['encoder_projection_hidden_dim']
        self.projection = encoder_projection(
            emg_dim=emg_enc_dim,
            nirs_dim=nirs_enc_dim,
            hidden_dim=projection_hidden_dim,
        )

        # Expose projection output dimension for downstream modules
        self.projection_out_dim = projection_hidden_dim

        # --- Local SSP heads ---
        projection_dim = self.projection_out_dim
        emg_dim, nirs_dim = config['emg_feature_dim'], config['nirs_feature_dim']

        self.local_ssp_head_emg = nn.Linear(projection_dim, emg_dim)
        self.local_ssp_head_nirs = nn.Linear(projection_dim, nirs_dim)

        # If we're only running SSP, avoid initializing the rest of the architecture.
        # This keeps SSP behavior invariant to changes in unrelated frozen modules.
        if self.build_mode == "ssp":
            return
        
        # --- Temporal downsampler ---
        ds_stride = int(math.prod([int(config.get('global_downsampler_first_stride', 5)), int(config.get('global_downsampler_second_stride', 5))]))
        # Split the total stride across two convolutions.
        first_stride = int(config.get('global_downsampler_first_stride', 5))
        second_stride = int(config.get('global_downsampler_second_stride', max(1, ds_stride // max(1, first_stride))))
        # Clamp invalid stride values instead of failing at construction time.
        if first_stride < 1: first_stride = 1
        if second_stride < 1: second_stride = 1
        total_stride = max(1, first_stride * second_stride)

        self.downsampler_conv1 = nn.Conv1d(
            in_channels=self.projection_out_dim,
            out_channels=config['conformer_params']['input_dim'],
            kernel_size=int(config['global_downsampler_first_kernel']),
            stride=first_stride
        )
        self.downsampler_conv2 = nn.Conv1d(
            in_channels=config['conformer_params']['input_dim'],
            out_channels=config['conformer_params']['input_dim'],
            kernel_size=int(config['global_downsampler_second_kernel']),
            stride=second_stride
        )
        self.downsampler_proj = None
        self.downsampler_norm = nn.LayerNorm(config['conformer_params']['input_dim'], eps=1e-3)
        self.downsampler_relu = nn.ReLU()
        
        # --- Positional encoding ---
        transformer_input_dim = config['conformer_params']['input_dim'] + config['num_task_phases']
        cparams = config.get('conformer_params', {})

        self.pos_encoder = PositionalEncoding(
            d_model=transformer_input_dim,
            dropout=config['conformer_params']['positional_encoding_dropout']
        )

        # --- Global sequence model ---
        from src.modules.conformer import ConformerEncoder
        cparams = config.get('conformer_params', {})
        self.transformer = ConformerEncoder(
            input_dim=transformer_input_dim,
            num_heads=int(cparams.get('num_heads', 4)),
            num_layers=int(cparams.get('num_layers', 4)),
            ffn_dim=int(cparams.get('ffn_dim', 512)),
            conv_kernel_size=int(cparams.get('conv_kernel_size', 15)),
            dropout=float(cparams.get('dropout', 0.3)),
            attention_causal=bool(cparams.get('attention_causal', False)),
            attention_window_size=int(cparams.get('attention_window_size', 0) or 0),
            relative_position_type=str(cparams.get('relative_position_type', 't5')),
            rpb_num_buckets=int(cparams.get('rpb_num_buckets', 32)),
            rpb_max_distance=int(cparams.get('rpb_max_distance', 128)),
            attention_window_schedule=cparams.get('attention_window_schedule', None),
            global_heads=int(cparams.get('global_heads', 0)),
            rpb_scale=float(cparams.get('rpb_scale', 0.0)),
            rpb_per_head=bool(cparams.get('rpb_per_head', True)),
            attn_mask_mode=str(cparams.get('attn_mask_mode', '3d'))
        )
        
        # --- Final regression head ---
        # Main regression head for post-MVC pain
        reg_loss = config.get('regression_loss', 'smooth_l1')
        if reg_loss == 'smooth_l1':
            out_dim = 2
        elif reg_loss in ('beta_nll', 'gamma_nll'):
            out_dim = 4
        elif reg_loss in ('beta2_nll', 'gamma2_nll'):
            out_dim = 10
        else:
            out_dim = 2
        self.main_regressor_head = nn.Sequential(
            nn.Linear(transformer_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, out_dim)
        )

        # --- Global SSP head ---
        projection_dim = self.projection_out_dim
        downsampled_dim = config['conformer_params']['input_dim']
        transformer_input_dim = config['conformer_params']['input_dim'] + config['num_task_phases']
        self.global_ssp_head = nn.Linear(transformer_input_dim, downsampled_dim)

    def augment_emg(self, emg_seq):
        return self.emg_spec_augment(emg_seq)

    def downsample_mask(self, m, ds):
        if ds <= 1: return m
        return F.max_pool1d(m.float().unsqueeze(1), kernel_size=ds, stride=ds, ceil_mode=True).squeeze(1).bool()

    def downsample_feats(self,x, ds):
        if ds <= 1: return x
        return F.avg_pool1d(x.transpose(1,2), kernel_size=ds, stride=ds, ceil_mode=True).transpose(1,2)

    def _projection_encoded_sequences(self, emg_encoded, nirs_encoded):
        """
        Trim both modality streams to a shared time axis, then project them.

        Returns:
            Tensor: projected sequence of shape (B, T_min, C_proj).
        """
        # Ensure sequence lengths match before fusion
        min_len = min(emg_encoded.shape[1], nirs_encoded.shape[1])
        emg_encoded = emg_encoded[:, :min_len, :]
        nirs_encoded = nirs_encoded[:, :min_len, :]

        projection_seq = self.projection(emg_encoded, nirs_encoded)
        return projection_seq

    def _encode_and_projection(self, emg_seq, nirs_seq):
        """Normalize, encode, and project the EMG and NIRS sequences."""
        emg_norm = self.emg_norm(emg_seq)
        nirs_norm = self.nirs_norm(nirs_seq)
        emg_encoded = self.emg_encoder(emg_norm)
        nirs_encoded = self.nirs_encoder(nirs_norm)
        projection_seq = self._projection_encoded_sequences(emg_encoded, nirs_encoded)
        return projection_seq

    def _downsample_sequence(self, projection_seq):
        """Apply the temporal downsampling block to a projected sequence."""
        x = projection_seq.transpose(1, 2)  # (N, C, L)
        x = self.downsampler_conv1(x)
        x = self.downsampler_relu(x)
        x = self.downsampler_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.downsampler_conv2(x)
        x = self.downsampler_relu(x)
        x = self.downsampler_norm(x.transpose(1, 2)).transpose(1, 2)
        return x.transpose(1, 2) # Return as (N, L_new, C_new)

    def forward(self, emg_seq, nirs_seq, task_phase_embedding=None, mode='main_regression', padding_mask=None, segment_ids=None, ssp_local_mask_params=None, device_ids=None):
        if mode == 'ssp':
            # --- Hierarchical SSP Logic ---
            
            # Step 1: Augment original data to create ground truth targets
            with torch.no_grad(): # Augmentation is not a differentiable operation
                emg_aug = self.augment_emg(emg_seq)
                nirs_aug = self.nirs_jitter(nirs_seq)
            
            # Step 2: Create locally masked inputs from the augmented data
            local_mask_params_emg = ssp_local_mask_params if ssp_local_mask_params is not None else self.config['ssp_local_mask_params_emg']
            local_mask_params_nirs = ssp_local_mask_params if ssp_local_mask_params is not None else self.config['ssp_local_mask_params_nirs']
            masked_emg_local, emg_mask_local = perform_masking(emg_aug, **local_mask_params_emg)
            masked_nirs_local, nirs_mask_local = perform_masking(nirs_aug, **local_mask_params_nirs)

            # Step 3: Forward pass for predictions
            projection_seq_from_masked = self._encode_and_projection(
                masked_emg_local, masked_nirs_local
            )
            # local_preds = self.local_ssp_head(fused_seq_from_masked)
            local_preds = self.local_ssp_head_emg(projection_seq_from_masked)
            local_preds_nirs = self.local_ssp_head_nirs(projection_seq_from_masked)
            local_preds = torch.cat([local_preds, local_preds_nirs], dim=-1)

            ds = math.prod(self.config['emg_encoder_params']['downsample_strides'])

            emg_mask_local_ds = self.downsample_mask(emg_mask_local, ds)
            nirs_mask_local_ds = self.downsample_mask(nirs_mask_local, ds)

            emg_mask_local_ds = emg_mask_local_ds[:, :local_preds.shape[1]]
            nirs_mask_local_ds = nirs_mask_local_ds[:, :local_preds.shape[1]]
            
            # The target for local loss is the original (but augmented) full feature set
            local_target_feats = torch.cat([emg_aug, nirs_aug], dim=-1)
            local_target_feats = self.downsample_feats(local_target_feats, ds)
            local_target_feats = local_target_feats[:, :local_preds.shape[1]]

            return (local_preds, local_target_feats, emg_mask_local_ds, nirs_mask_local_ds,
                    emg_mask_local, nirs_mask_local, emg_aug, nirs_aug)


        elif mode == 'main_regression':
            mask = padding_mask.bool() if padding_mask is not None else None
            projection_seq = self._encode_and_projection(emg_seq, nirs_seq)
            if mask is not None:
                ds_factor = math.prod(self.config['emg_encoder_params']['downsample_strides'])
                mask = self.downsample_mask(mask, ds_factor)
                mask = mask[:, :projection_seq.shape[1]]

            downsampled_seq = self._downsample_sequence(projection_seq)
            if mask is not None:
                mask = self.downsample_mask(mask, int(math.prod([int(self.config.get('global_downsampler_first_stride', 5)), int(self.config.get('global_downsampler_second_stride', 5))])))
                mask = mask[:, :downsampled_seq.shape[1]]
            
            # Downsample task phase embeddings...
            # NOTE: The downsampling factor for task phases must match the TOTAL downsampling of the signal.
            global_stride = int(math.prod([int(self.config.get('global_downsampler_first_stride', 5)), int(self.config.get('global_downsampler_second_stride', 5))]))
            enc_stride = math.prod(self.config['emg_encoder_params']['downsample_strides'])
            eff_global = max(1, global_stride)
            total_stride = enc_stride * eff_global
            downsampled_one_hot_phases = F.max_pool1d(
                task_phase_embedding.transpose(1, 2).float(), # (N, C, L) for pooling
                kernel_size=total_stride,
                stride=total_stride
            ).transpose(1, 2) # (N, L_new, C)
            
            # Ensure sequence lengths match after all downsampling
            min_len = min(downsampled_seq.shape[1], downsampled_one_hot_phases.shape[1])
            downsampled_seq = downsampled_seq[:, :min_len, :]
            downsampled_one_hot_phases = downsampled_one_hot_phases[:, :min_len, :]
            if mask is not None:
                mask = mask[:, :min_len]

            # Concatenate features and task phases...
            seq_for_transformer = torch.cat([downsampled_seq, downsampled_one_hot_phases], dim=-1)

            # Optionally add segment embeddings (downsampled to match)
            if self.use_segment_embeddings and segment_ids is not None:
                # Downsample segment ids using max-pool over integers encoded as floats
                seg_ids_float = segment_ids.unsqueeze(1).float()  # (N, 1, L)
                global_stride = int(math.prod([int(self.config.get('global_downsampler_first_stride', 5)), int(self.config.get('global_downsampler_second_stride', 5))]))
                enc_stride = math.prod(self.config['emg_encoder_params']['downsample_strides'])
                eff_global = max(1, global_stride)
                total_stride = enc_stride * eff_global
                seg_ids_ds = F.max_pool1d(seg_ids_float, kernel_size=total_stride, stride=total_stride).squeeze(1).long()  # (N, L_ds)
                seg_ids_ds = seg_ids_ds[:, :seq_for_transformer.shape[1]]
                seg_emb = self.segment_embedding(seg_ids_ds)  # (N, L_ds, D)
            else:
                seg_emb = None

            # Optionally add device embedding as residual (broadcast to sequence length)
            if self.use_device_embedding and device_ids is not None:
                dev_emb = self.device_embedding(device_ids)  # (N, D)
                dev_emb = dev_emb.unsqueeze(1).expand(-1, seq_for_transformer.shape[1], -1)
                seq_for_transformer = seq_for_transformer + dev_emb

            # --- CLS Token Logic for Supervised Task ---
            batch_size = seq_for_transformer.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

            # prepend CLS token
            seq_for_transformer = torch.cat([cls_tokens, seq_for_transformer], dim=1)
            if mask is not None:
                # prepend CLS token to the mask
                mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1)

            # If segment embeddings are used, prepend CLS segment id embedding (0)
            if seg_emb is not None:
                cls_seg = self.segment_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
                seg_emb = torch.cat([cls_seg, seg_emb], dim=1)
                seq_for_transformer = seq_for_transformer + seg_emb
            seq_for_transformer_pos = self.pos_encoder(seq_for_transformer)

            aggregated_seq = self.transformer(seq_for_transformer_pos,
                                              src_key_padding_mask=mask)

            # --- Select the CLS token's output for regression ---
            # The CLS token is at the first position (index 0)
            cls_output = aggregated_seq[:, 0, :] # Shape: [Batch, Features]
            
            main_out = self.main_regressor_head(cls_output)
            return main_out

        else:
            raise ValueError(f"Unknown mode: {mode}")