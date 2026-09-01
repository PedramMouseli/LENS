def get_config():
    """
    Return the experiment configuration dictionary.
    """
    
    # --- Base Feature Dimensions ---
    # These should match the output of your preprocessing.py script.
    emg_feature_dim = 66 
    # Example: 3 NIRS signals (O2Hb, HHb, TSI) + their 3 derivatives
    nirs_feature_dim = 28

    config = {
        # --- Data & General Settings ---
        'target_sample_rate': 50,  # Hz. All features are resampled to this rate.
        'num_task_phases': 3,      # Number of task-phase labels used for one-hot embeddings.
        
        # --- Label transform control & distributional regression ---
        # If False: no log-normalization; dataset will scale labels to [0,1] only
        'use_label_log_transform': True,
        'distribution_eps': 1e-6,
        # Regression loss: 'smooth_l1', 'beta_nll', 'gamma_nll', 'beta2_nll', 'gamma2_nll'
        'regression_loss': 'smooth_l1',
        
        # --- Self-Supervised Pre-training (SSP) ---
        'ssp_epochs': 2000,
        'ssp_lr': 5e-5,
        'ssp_batch_size': 8,
        # Use longer chunks for hierarchical SSP to give the transformer meaningful context
        'ssp_context_len': 20 * 50,
        'ssp_local_mask_params_emg': {
            'mask_prob': 0.5,
            'mask_len': 10,
        },
        'ssp_local_mask_params_nirs': {
            'mask_prob': 0.45,
            'mask_len': 150,
        },

        'ssp_down_loss_alpha': 1.0,      # weight of downsampled loss

        'ssp_build_mode': 'ssp',
        'ssp_cosine_alpha_nirs': 1.0,

        # --- Supervised Fine-tuning ---
        'sup_epochs': 300,
        'sup_lr_main': 5e-5,  # Learning rate for the main, final prediction task
        'sup_lr_encoder': 5e-5,   # Learning rate for the encoder weights

        'sup_batch_size': 8,

        'use_trial_windowing': True, # toggle between windowed-concat vs full sequence
        'trial_window_seconds': 60, # per-trial window length ending at trial index
        'num_trials_before_mvc_for_final': 7, # how many trials before MVC to include
        'mvc_post_seconds': 120, # extend window after MVC end
        'window_separator_seconds': 5, # optional small zero-gap between concatenated windows
        
        # --- Evaluation & Adaptation ---
        'eval_batch_size': 4,
        'adaptation_epochs': 40,
        # Optional supervised selection of adaptation epochs before the final
        # adapt_and_evaluate run.
        'adaptation_epoch_selection_enabled': True,
        # Which validation subjects to use when selecting adaptation epochs:
        # - 'transfer_validation_set': draw validation subjects from the test split and
        #   remove them from the final evaluation subjects
        # - 'trainval': use the full training-domain subject pool
        #   (train + validation subjects) and choose the epoch count from the
        #   aggregate performance on the validation set
        'adaptation_epoch_selection_source': 'transfer_validation_set',
        # Candidate adaptation epoch counts to search over.
        'adaptation_epoch_selection_candidates': [1,10,20,40],
        # Number of subjects to hold out from the test set when
        # adaptation_epoch_selection_source == 'transfer_validation_set'.
        'adaptation_transfer_validation_num_subjects': 6,
        'adaptation_transfer_validation_seed_offset': 0,
        'adaptation_lr': 1e-5, # Very low LR for test-time adaptation of encoders
        'adaptation_batch_size': 4,
        'adaptation_chunks_per_epoch': 16,
        
        # --- Model Architecture ---
        # Augmentations
        'emg_spec_augment_params': {
            'freq_mask_param': 2,
            'time_mask_param': 10, # 500ms
            'num_freq_masks': 6,
            'num_time_masks': 10,
        },
        'nirs_jitter_params': {
            'sigma': 0.03,
        },
        
        # EMG Encoder (TDSConvNet)
        'emg_encoder_params': {
            'hidden_dim': 128,
            'num_layers': 5,
            'kernel_size': 11,
            'dropout': 0.3,
            # Stage 1: (8x)
            'downsample_strides': [2, 1, 2, 1, 2], 
        },

        # NIRS Encoder (TDSConvNet)
        'nirs_encoder_params': {
            'hidden_dim': 64,
            'num_layers': 3,
            'kernel_size': 7,
            'dropout': 0.3,
            # Stage 1: Also downsample by 8x
            'downsample_strides': [2, 2, 2],
        },
        
        # Encoder Projection Layer
        'encoder_projection_hidden_dim': 128,
        
        # Temporal Downsampler
        'global_downsampler_first_stride': 5,
        'global_downsampler_second_stride': 5,
        # 'global_downsampler_stride': 25,
        'global_downsampler_kernel': 10, # Non-overlapping windows
        'global_downsampler_first_kernel': 9,
        'global_downsampler_second_kernel': 9,
        
        # Conformer
        'conformer_params': {
            'input_dim': 125,
            'positional_encoding_dropout': 0.3,
            'num_heads': 4,
            'num_layers': 4,
            'ffn_dim': 512,
            'conv_kernel_size': 15,
            'dropout': 0.3,
            # Attention options
            'attention_causal': False,
            # Set window_size to 0 or None to disable local/windowed masking
            'attention_window_size': 10,
            'attention_window_schedule': [8, 8, 16, 32],
            # Relative positional bias: 'none' | 't5' 
            'relative_position_type': 't5',
            'rpb_num_buckets': 16,
            'rpb_max_distance': 256,
            'rpb_scale': 1.0,
            'rpb_per_head': False,
            'attn_mask_mode': '2d',
            'global_heads': 1,
        },
        
        'sup_patience': 50,      # Patience for supervised early stopping
    }
    
    # --- Set dynamic feature dimensions ---
    config['emg_feature_dim'] = emg_feature_dim
    config['nirs_feature_dim'] = nirs_feature_dim
    
    return config

if __name__ == '__main__':
    training_config = get_config()
    
    print("--- Configuration summary ---")
    print(f"EMG Feature Dimension: {training_config['emg_feature_dim']}")
    print(f"NIRS Feature Dimension: {training_config['nirs_feature_dim']}")
    print(f"SSP Context Length (steps): {training_config['ssp_context_len']}")
    print(f"Main Supervised LR: {training_config['sup_lr_main']}")
    