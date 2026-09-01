import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import time

# Assuming data_processing.py is in the same directory or accessible
from data_processing import muscle_data

def load_and_prepare_ratings(ratings_path, participants_path):
    """
    Loads pain ratings from CSV for left/right masseter,
    and splits it into intermediate and post-mvc ratings.
    """
    ratings_df = pd.read_csv(ratings_path)
    participants_df = pd.read_csv(participants_path)
    
    all_subject_ratings = {}

    # Optional class label per subject (classification task)
    class_label_map = {}
    if 'class_label' in participants_df.columns:
        for _, row in participants_df.iterrows():
            class_label_map[row['sub_id']] = row['class_label']

    for subject_id in participants_df['sub_id']:
        if subject_id not in ratings_df['participant_id'].values:
            print(f"Warning: Subject {subject_id} not found in ratings file. Skipping.")
            continue

        subject_ratings_raw = ratings_df[ratings_df['participant_id'] == subject_id]
        
        pain_rc = np.zeros(16)
        pain_lc = np.zeros(16)
        # Remove fatigue ratings per new requirement
        
        # Post-MVC pain (Trial 16 equivalent, index 0)
        pain_rc[0] = int(subject_ratings_raw['Right Cheek'].iloc[0])
        pain_lc[0] = int(subject_ratings_raw['Left Cheek'].iloc[0])
        # (fatigue removed)

        # Intermediate pain ratings (Trials 1-15, indices 1-15)
        for i in range(2, 31, 2): # Odd numbers for fatigue, Even for pain, up to 30
            pain_col_rc = f"Right Cheek.{i}"
            pain_col_lc = f"Left Cheek.{i}"
            trial_idx = (i + 1) // 2
            
            if pain_col_rc in subject_ratings_raw.columns and pain_col_lc in subject_ratings_raw.columns:
                 pain_rc[trial_idx] = int(subject_ratings_raw[pain_col_rc].iloc[0])
                 pain_lc[trial_idx] = int(subject_ratings_raw[pain_col_lc].iloc[0])
            # (fatigue removed)

        # Per user request: index 0-14 are intermediate, index 15 is post_mvc_pain.
        # post_mvc_pain is shape (2,) for [left, right]
        post_mvc_pain = np.array([pain_lc[15], pain_rc[15]], dtype=np.float32)
        # intermediate_pain is shape (15, 2) for [trial, side]
        intermediate_pain = np.stack([pain_lc[0:15], pain_rc[0:15]], axis=1).astype(np.float32)

        entry = {
            'post_mvc_pain': post_mvc_pain,
            'intermediate_pain': intermediate_pain,
            # fatigue removed
        }
        if subject_id in class_label_map:
            entry['class_label'] = int(class_label_map[subject_id])
        all_subject_ratings[subject_id] = entry
        
    return all_subject_ratings

def process_subject(sub_params, subject_ratings, data_folder, raw_output_dir):
    """
    Processes a single subject's data and saves it to an HDF5 file.
    """
    sub_id = sub_params['sub_id']
    sub_num = sub_id # Use sub_id as the number for consistency
    
    if sub_id not in subject_ratings:
        print(f"Skipping {sub_id}: No pain ratings found.")
        return

    print(f'Processing {sub_num} (Subject ID: {sub_id})')
    
    # --- 1. Load parameters from metadata ---
    offset = sub_params['offset']
    mvc_events = np.fromstring(sub_params['mvc_events'], dtype=int, sep=',')
    clench_array = np.fromstring(sub_params['clench_order'], dtype=int, sep=',')
    scaling_coef = sub_params['scaling_coef']
    add_events = np.fromstring(sub_params['add_event'], dtype=float, sep=',') if not pd.isna(sub_params['add_event']) else None
    rm_events = np.fromstring(sub_params['rm_event'], dtype=float, sep=',') if not pd.isna(sub_params['rm_event']) else None
    stress_included = sub_params['stress_included']
    
    # --- 2. Load and process signals using muscle_data class ---
    nirs_path = os.path.join(data_folder, 'NIRS_excel', f'{sub_id}.xlsx')
    emg_path = os.path.join(data_folder, 'EMG_mat', f'{sub_id}.mat')
    
    if not os.path.exists(nirs_path) or not os.path.exists(emg_path):
        print(f"Skipping {sub_id}: Data file(s) not found.")
        return
        
    sub_data = muscle_data(sub_id, nirs_path, emg_path, plot_path=None, scaling_coef=scaling_coef, stress_included=stress_included)
    sub_data.import_nirs()
    sub_data.import_emg()

    # Determine EMG device type from sampling frequency
    if sub_data.emg_sf == 2048.0:
        device_type = 'A'
    elif sub_data.emg_sf == 2000.0:
        device_type = 'B'
    else:
        print(f"Warning: Unknown EMG sampling frequency {sub_data.emg_sf} for {sub_id}. Defaulting to 'B'.")
        device_type = 'B'

    if rm_events is not None: sub_data.remove_event(rm_events)
    if add_events is not None: sub_data.add_event(add_events)
        
    sub_data.generate_task_events(offset=offset)
    sub_data.confirm_events()
    sub_data.sync_normalize_nirs()
    sub_data.set_params(sub_num, clench_array, mvc_events)
    
    # --- 3. Determine task end time and align data ---
    # Based on the experimental protocol:
    # - NIRS data is already sliced to start at the first event (our "Time Zero").
    # - EMG recording starts at the same "Time Zero".
    # - Therefore, we can directly compare the total duration of both signals.
    
    emg_full_time_vec = sub_data.emg_data['time'].values
    nirs_time_vec_synced = sub_data.nirs_data['time'].values
    
    # The duration of the task-relevant NIRS signal is its total length.
    nirs_duration = nirs_time_vec_synced[-1] if len(nirs_time_vec_synced) > 0 else 0
    
    # The duration of the task-relevant EMG signal is its total length.
    emg_duration = emg_full_time_vec[-1] if len(emg_full_time_vec) > 0 else 0
    
    # Determine the final duration based on the shorter of the two signals.
    final_duration_sec = emg_duration if stress_included else min(emg_duration, nirs_duration)
    final_duration_sec = max(0, final_duration_sec) # Ensure not negative

    # Find the end index for both signals based on this final duration.
    nirs_end_idx = np.argmin(np.abs(nirs_time_vec_synced - final_duration_sec))
    emg_end_idx = np.argmin(np.abs(emg_full_time_vec - final_duration_sec))

    # Slice both dataframes from their beginning to the calculated end index.
    nirs_df = sub_data.nirs_data.iloc[:nirs_end_idx]
    emg_df = sub_data.emg_data.iloc[:emg_end_idx]
    
    # --- 4. Combine NIRS sensors ---
    # Average O2Hb and HHb across three channels per side; keep TSI as-is
    # l_o2hb = nirs_df[['lo2hb1','lo2hb2','lo2hb3']].mean(axis=1)
    # l_hhb  = nirs_df[['lhhb1','lhhb2','lhhb3']].mean(axis=1)
    # r_o2hb = nirs_df[['ro2hb1','ro2hb2','ro2hb3']].mean(axis=1)
    # r_hhb  = nirs_df[['rhhb1','rhhb2','rhhb3']].mean(axis=1)
    l_o2hb1 = nirs_df[['lo2hb1']].squeeze()
    l_o2hb2 = nirs_df[['lo2hb2']].squeeze()
    l_o2hb3 = nirs_df[['lo2hb3']].squeeze()
    l_hhb1 = nirs_df[['lhhb1']].squeeze()
    l_hhb2 = nirs_df[['lhhb2']].squeeze()
    l_hhb3 = nirs_df[['lhhb3']].squeeze()
    r_o2hb1 = nirs_df[['ro2hb1']].squeeze()
    r_o2hb2 = nirs_df[['ro2hb2']].squeeze()
    r_o2hb3 = nirs_df[['ro2hb3']].squeeze()
    r_hhb1 = nirs_df[['rhhb1']].squeeze()
    r_hhb2 = nirs_df[['rhhb2']].squeeze()
    r_hhb3 = nirs_df[['rhhb3']].squeeze()
    l_tsi  = nirs_df[['ltsi1']].squeeze()
    r_tsi  = nirs_df[['rtsi1']].squeeze()
    # nirs_combined_df = pd.DataFrame({
    #     'lo2hb': l_o2hb,
    #     'lhhb': l_hhb,
    #     'ltsi': l_tsi,
    #     'ro2hb': r_o2hb,
    #     'rhhb': r_hhb,
    #     'rtsi': r_tsi,
    # })
    nirs_combined_df = pd.DataFrame({
        'lo2hb1': l_o2hb1,
        'lo2hb2': l_o2hb2,
        'lo2hb3': l_o2hb3,
        'lhhb1': l_hhb1,
        'lhhb2': l_hhb2,
        'lhhb3': l_hhb3,
        'ro2hb1': r_o2hb1,
        'ro2hb2': r_o2hb2,
        'ro2hb3': r_o2hb3,
        'rhhb1': r_hhb1,
        'rhhb2': r_hhb2,
        'rhhb3': r_hhb3,
        'ltsi': l_tsi,
        'rtsi': r_tsi,
    })
    
    # Force conversion to numeric, coerce errors to NaN, fill NaNs with 0, and ensure float32 type
    final_nirs_data = nirs_combined_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    
    # Do the same for EMG data to be safe
    final_emg_data = emg_df[['emg_masseter_l', 'emg_masseter_r']].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    
    # --- 5. Create task_phase_ids and intermediate_pain_indices ---
    nirs_time_vec = nirs_df['time'].values
    task_phase_ids = np.zeros(len(nirs_time_vec), dtype=np.int32)
    
    mvc_start_time = sub_data.events[mvc_events[0]-1]
    mvc_end_time = sub_data.events[mvc_events[1]-1]
    mvc_indices = np.where((nirs_time_vec >= mvc_start_time) & (nirs_time_vec <= mvc_end_time))
    task_phase_ids[mvc_indices] = 2
    
    intermediate_pain_indices = []
    for i in clench_array:
        task_start = sub_data.task_events[(i*2)-2]
        task_end = sub_data.task_events[(i*2)-1]
        task_indices = np.where((nirs_time_vec >= task_start) & (nirs_time_vec <= task_end))
        task_phase_ids[task_indices] = 1
        
        rating_collection_time = task_end + 30
        rating_idx = np.argmin(np.abs(nirs_time_vec - rating_collection_time))
        intermediate_pain_indices.append(rating_idx)

    # --- 6. Save to HDF5 ---
    output_path = os.path.join(raw_output_dir, f"{sub_id}.hdf5")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('emg', data=final_emg_data)
        f.create_dataset('nirs', data=final_nirs_data)
        f.create_dataset('device_type', data=np.string_(device_type))
        
        grp = f.create_group('labels')
        grp.create_dataset('post_mvc_pain', data=subject_ratings[sub_id]['post_mvc_pain'])
        grp.create_dataset('intermediate_pain', data=subject_ratings[sub_id]['intermediate_pain'])
        # fatigue removed
        grp.create_dataset('intermediate_pain_indices', data=np.array(intermediate_pain_indices, dtype=np.int32))
        grp.create_dataset('task_phase_ids', data=task_phase_ids)
        # Persist class label if available
        if 'class_label' in subject_ratings.get(sub_id, {}):
            grp.create_dataset('class_label', data=np.array(subject_ratings[sub_id]['class_label'], dtype=np.int32))

    print(f"Successfully saved processed data for {sub_id} to {output_path}")

def main():
    # --- Dynamically construct paths relative to the script location ---
    # This makes the script runnable from any directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')

    # --- Configuration ---
    METADATA_PATH = os.path.join(project_root, 'data/subject_metadata.csv')
    DATA_FOLDER = os.path.join(project_root, 'data/raw_source')
    RAW_OUTPUT_DIR = os.path.join(project_root, 'data/raw_no_normalization')
    
    # Paths for training set
    RATINGS_PATH_TRAIN = os.path.join(project_root, 'data/ratings/StO2AssessmentForm_DATA.csv')
    PARTICIPANTS_TRAIN_PATH = os.path.join(project_root, 'data/ratings/Participants_classification.csv')
    
    # Paths for validation/test set
    RATINGS_PATH_TEST = os.path.join(project_root, 'data/ratings/StO2AssessmentForm_DATA_VAL.csv')
    PARTICIPANTS_TEST_PATH = os.path.join(project_root, 'data/ratings/Validation_participants_classification.csv')

    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    
    train_ratings = load_and_prepare_ratings(RATINGS_PATH_TRAIN, PARTICIPANTS_TRAIN_PATH)
    test_ratings = load_and_prepare_ratings(RATINGS_PATH_TEST, PARTICIPANTS_TEST_PATH)
    all_ratings = {**train_ratings, **test_ratings}
    
    # --- Load participant lists and metadata ---
    participants_train_df = pd.read_csv(PARTICIPANTS_TRAIN_PATH)
    participants_test_df = pd.read_csv(PARTICIPANTS_TEST_PATH)
    all_participant_ids = pd.concat([participants_train_df['sub_id'], participants_test_df['sub_id']]).unique()

    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Filter metadata to only include subjects from the official participant lists
    metadata_to_process = metadata_df[metadata_df['sub_id'].isin(all_participant_ids)]

    # --- Process only the subjects specified in the participant lists ---
    for _, row in tqdm(metadata_to_process.iterrows(), total=len(metadata_to_process), desc="Processing Subjects"):
        try:
            process_subject(row.to_dict(), all_ratings, DATA_FOLDER, RAW_OUTPUT_DIR)
        except Exception as e:
            print(f"!!! ERROR processing subject {row['sub_id']}: {e}")
    
    print("\nData preparation complete.")

if __name__ == '__main__':
    main()