import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.baseline_features import extract_subject_features
from src.baseline_models import (
    PLSPreRegisteredModel,
    SpectralShiftLinearModel,
    LinearEMGModel,
    XGBoostEMGModel,
    evaluate_predictions,
    cv_overall_pval
)

def load_cohort_dataset(participant_csv_path, raw_dir):
    """Loads feature sets and targets for all available subjects in a cohort CSV."""
    df = pd.read_csv(participant_csv_path)
    sub_ids = [s for s in df['sub_id'].dropna() if s != '**']
    
    data = []
    print(f"Loading {os.path.basename(participant_csv_path)} ({len(sub_ids)} subjects)...")
    for sub_id in sub_ids:
        hdf5_path = os.path.join(raw_dir, f"{sub_id}.hdf5")
        if not os.path.exists(hdf5_path):
            continue
        feat_dict = extract_subject_features(hdf5_path)
        data.append({
            'sub_id': sub_id,
            'left': feat_dict['left'],
            'right': feat_dict['right'],
            'device': feat_dict['device']
        })
    return data


def run_cross_validation(data_list, model_cls, feature_key='pls_features', n_splits=5):
    """Runs 5-fold cross-validation separately for Left and Right sides."""
    n_subs = len(data_list)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rhos_left, rhos_right = [], []

    fold_data_l, fold_data_r = [], []
    
    device_list = [subject['device'] for subject in data_list]

    for train_idx, val_idx in kf.split(range(n_subs), device_list):
        train_subs = [data_list[i] for i in train_idx]
        val_subs = [data_list[i] for i in val_idx]

        # Extract X and y for Left
        X_tr_l = np.array([s['left'][feature_key] for s in train_subs])
        y_tr_l = np.array([s['left']['target'] for s in train_subs])
        X_va_l = np.array([s['left'][feature_key] for s in val_subs])
        y_va_l = np.array([s['left']['target'] for s in val_subs])

        # Extract X and y for Right
        X_tr_r = np.array([s['right'][feature_key] for s in train_subs])
        y_tr_r = np.array([s['right']['target'] for s in train_subs])
        X_va_r = np.array([s['right'][feature_key] for s in val_subs])
        y_va_r = np.array([s['right']['target'] for s in val_subs])

        # Left model
        m_l = model_cls().fit(X_tr_l, y_tr_l)
        pred_l = m_l.predict(X_va_l)
        rho_l, _ = evaluate_predictions(y_va_l, pred_l)
        rhos_left.append(rho_l)

        # Right model
        m_r = model_cls().fit(X_tr_r, y_tr_r)
        pred_r = m_r.predict(X_va_r)
        rho_r, _ = evaluate_predictions(y_va_r, pred_r)
        rhos_right.append(rho_r)

        fold_data_l.append({'labels': y_va_l, 'preds': pred_l})
        fold_data_r.append({'labels': y_va_r, 'preds': pred_r})

    # Overal p-value
    p_value_left = cv_overall_pval(fold_data_l, rhos_left)
    p_value_right = cv_overall_pval(fold_data_r, rhos_right)


    return (np.mean(rhos_left), np.std(rhos_left), p_value_left), (np.mean(rhos_right), np.std(rhos_right), p_value_right)

def run_transfer_evaluation(train_data, test_data, model_cls, feature_key='pls_features'):
    """Evaluates zero-shot transfer performance from train_data to test_data."""
    # Left side
    X_tr_l = np.array([s['left'][feature_key] for s in train_data])
    y_tr_l = np.array([s['left']['target'] for s in train_data])
    X_te_l = np.array([s['left'][feature_key] for s in test_data])
    y_te_l = np.array([s['left']['target'] for s in test_data])

    # Right side
    X_tr_r = np.array([s['right'][feature_key] for s in train_data])
    y_tr_r = np.array([s['right']['target'] for s in train_data])
    X_te_r = np.array([s['right'][feature_key] for s in test_data])
    y_te_r = np.array([s['right']['target'] for s in test_data])

    m_l = model_cls().fit(X_tr_l, y_tr_l)
    pred_l = m_l.predict(X_te_l)
    rho_l, p_l = evaluate_predictions(y_te_l, pred_l)

    m_r = model_cls().fit(X_tr_r, y_tr_r)
    pred_r = m_r.predict(X_te_r)
    rho_r, p_r = evaluate_predictions(y_te_r, pred_r)

    return (rho_l, p_l), (rho_r, p_r)

def main():
    raw_dir = os.path.join(project_root, 'data/raw')
    ratings_dir = os.path.join(project_root, 'data/ratings')

    train_path = os.path.join(ratings_dir, 'Participants_control_train.csv')
    test_path = os.path.join(ratings_dir, 'Participants_control_test.csv')
    tmd_path = os.path.join(ratings_dir, 'Participants_TMD.csv')

    # Load features if pre-computed, else extract from raw data
    out_features_dir = os.path.join(project_root, 'data/baseline_features')
    if not os.path.exists(out_features_dir):
        os.makedirs(out_features_dir)

    train_save_path = os.path.join(out_features_dir, 'train_features.pkl')
    test_save_path = os.path.join(out_features_dir, 'test_features.pkl')
    tmd_save_path = os.path.join(out_features_dir, 'tmd_features.pkl')

    if os.path.exists(train_save_path) and os.path.exists(test_save_path) and os.path.exists(tmd_save_path):
        print("\n--- Loading Pre-computed Features ---")
        with open(train_save_path, 'rb') as f:
            control_train = pickle.load(f)
        with open(test_save_path, 'rb') as f:
            control_test = pickle.load(f)
        with open(tmd_save_path, 'rb') as f:
            tmd_cohort = pickle.load(f)
    else:
        print("\n--- Extracting Baseline Features from Raw HDF5 Data ---")
        control_train = load_cohort_dataset(train_path, raw_dir)
        with open(train_save_path, 'wb') as f:
            pickle.dump(control_train, f)
        control_test = load_cohort_dataset(test_path, raw_dir)
        with open(test_save_path, 'wb') as f:
            pickle.dump(control_test, f)
        tmd_cohort = load_cohort_dataset(tmd_path, raw_dir)
        with open(tmd_save_path, 'wb') as f:
            pickle.dump(tmd_cohort, f)

    print(f"\nLoaded cohorts: Control-train (n={len(control_train)}), Control-test (n={len(control_test)}), mTMD (n={len(tmd_cohort)})")

    models_config = [
        ('PLS-PreRegistered', PLSPreRegisteredModel, 'pls_features'),
        ('SpectralShift-Linear', SpectralShiftLinearModel, 'spectral_features'),
        ('Linear-EMG', LinearEMGModel, 'classical_features'),
        ('XGBoost-EMG', XGBoostEMGModel, 'classical_features'),
    ]

    results = []

    for name, model_cls, feat_key in models_config:
        print(f"\nRunning evaluations for model: {name}...")

        # 1. Control-train 5-fold CV
        (cv_l_mean, cv_l_std, cv_l_pval), (cv_r_mean, cv_r_std, cv_r_pval) = run_cross_validation(control_train, model_cls, feat_key, n_splits=5)

        # 2. Combined dataset (Control-train + Control-test + mTMD) 5-fold CV
        combined_dataset = control_train + control_test + tmd_cohort

        (comb_cv_l_mean, comb_cv_l_std, comb_cv_l_pval), (comb_cv_r_mean, comb_cv_r_std, comb_cv_r_pval) = run_cross_validation(combined_dataset, model_cls, feat_key, n_splits=5)
        
        # 3. Control-train -> Control-test Generalization
        (gen_l_rho, gen_l_p), (gen_r_rho, gen_r_p) = run_transfer_evaluation(control_train, control_test, model_cls, feat_key)

        # 4. Control-train -> mTMD Transfer
        (tr_l_rho, tr_l_p), (tr_r_rho, tr_r_p) = run_transfer_evaluation(control_train, tmd_cohort, model_cls, feat_key)

        # 5. mTMD 5-fold CV
        (tmd_cv_l_mean, tmd_cv_l_std, tmd_cv_l_pval), (tmd_cv_r_mean, tmd_cv_r_std, tmd_cv_r_pval) = run_cross_validation(tmd_cohort, model_cls, feat_key, n_splits=5)

        # 6. mTMD -> Control-train Reverse Transfer
        (rev_l_rho, rev_l_p), (rev_r_rho, rev_r_p) = run_transfer_evaluation(tmd_cohort, control_train, model_cls, feat_key)

        # 7.mTMD -> Control-test Reverse Transfer
        (rev_ct_l_rho, rev_ct_l_p), (rev_ct_r_rho, rev_ct_r_p) = run_transfer_evaluation(tmd_cohort, control_test, model_cls, feat_key)

        results.append({
            'Model': name,
            'Control-train 5-Fold CV (L)': f"{cv_l_mean:.2f} ± {cv_l_std:.2f} ({cv_l_pval:.4f})",
            'Control-train 5-Fold CV (R)': f"{cv_r_mean:.2f} ± {cv_r_std:.2f} ({cv_r_pval:.4f})",
            'Combined Dataset 5-Fold CV (L)': f"{comb_cv_l_mean:.2f} ± {comb_cv_l_std:.2f} ({comb_cv_l_pval:.4f})",
            'Combined Dataset 5-Fold CV (R)': f"{comb_cv_r_mean:.2f} ± {comb_cv_r_std:.2f} ({comb_cv_r_pval:.4f})",
            'Control-test Gen (L)': f"{gen_l_rho:.2f} (p={gen_l_p:.4f})",
            'Control-test Gen (R)': f"{gen_r_rho:.2f} (p={gen_r_p:.4f})",
            'Control -> mTMD Transfer (L)': f"{tr_l_rho:.2f} (p={tr_l_p:.4f})",
            'Control -> mTMD Transfer (R)': f"{tr_r_rho:.2f} (p={tr_r_p:.4f})",
            'mTMD 5-Fold CV (L)': f"{tmd_cv_l_mean:.2f} ± {tmd_cv_l_std:.2f} ({tmd_cv_l_pval:.4f})",
            'mTMD 5-Fold CV (R)': f"{tmd_cv_r_mean:.2f} ± {tmd_cv_r_std:.2f} ({tmd_cv_r_pval:.4f})",
            'mTMD -> Control Rev Transfer (L)': f"{rev_l_rho:.2f} (p={rev_l_p:.4f})",
            'mTMD -> Control Rev Transfer (R)': f"{rev_r_rho:.2f} (p={rev_r_p:.4f})",
            'mTMD -> Control-test Reverse Transfer (L)': f"{rev_ct_l_rho:.2f} (p={rev_ct_l_p:.4f})",
            'mTMD -> Control-test Reverse Transfer (R)': f"{rev_ct_r_rho:.2f} (p={rev_ct_r_p:.4f})",
        })

    res_df = pd.DataFrame(results)
    print("\n=======================================================")
    print("           BASELINE MODELS PERFORMANCE SUMMARY        ")
    print("=======================================================\n")
    print(res_df.to_string(index=False))

    out_csv = os.path.join(project_root, 'data/baseline_results.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved baseline evaluation summary to {out_csv}")

if __name__ == '__main__':
    main()
