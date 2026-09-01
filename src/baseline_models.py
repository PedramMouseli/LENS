import numpy as np
from scipy.stats import spearmanr, permutation_test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor

class PLSPreRegisteredModel:
    """
    Pre-registered PLS model using PLSCanonical/PLSRegression and nested 10-fold CV.
    """
    def __init__(self, max_iter=100000):
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        p_grid = {'tol': [1e-5, 1e-6, 1e-7, 1e-8]}
        base = PLSCanonical(n_components=1, max_iter=self.max_iter)
        grid = GridSearchCV(base, p_grid, cv=KFold(n_splits=min(10, max(2, len(X))), shuffle=True, random_state=42), scoring='neg_mean_squared_error')
        try:
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_
        except Exception:
            self.model = PLSCanonical(n_components=1, max_iter=self.max_iter)
            self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.squeeze(preds)

class SpectralShiftLinearModel:
    """
    Linear regression model based on median frequency shift features (refs 46-47).
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        p_grid = {'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        grid = GridSearchCV(Ridge(), p_grid, cv=KFold(n_splits=min(10, max(2, len(X))), shuffle=True, random_state=42), scoring='neg_mean_squared_error')
        grid.fit(X_scaled, y)
        self.model = grid.best_estimator_
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class LinearEMGModel:
    """
    Ridge linear regression on full 18 classical EMG + NIRS features.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        p_grid = {'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        grid = GridSearchCV(Ridge(), p_grid, cv=KFold(n_splits=min(10, max(2, len(X))), shuffle=True, random_state=42), scoring='neg_mean_squared_error')
        grid.fit(X_scaled, y)
        self.model = grid.best_estimator_
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGBOOST = False

class XGBoostEMGModel:
    """
    XGBoost / Gradient Boosted Decision Trees regressor on full 18 classical EMG + NIRS features.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def evaluate_predictions(y_true, y_pred):
    """Computes Spearman correlation and p-value."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0, 1.0
    rho, _ = spearmanr(y_true, y_pred)
    pval = permutation_test(
                    (y_true, y_pred),
                    lambda x,y: spearmanr(x,y).statistic,
                    permutation_type='pairings',
                    n_resamples=10000,
                    random_state=42,
                    alternative='greater'
                    ).pvalue
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(pval)

def cv_overall_pval(fold_data, observed_rho, n_permutations=5000):
    """Computes the overall p-value for the CV results using permutation testing."""
    null_mean_rhos = []
    rng = np.random.default_rng(seed=42)
    
    for _ in range(n_permutations):
        perm_rhos = []
        
        for fold_dat in fold_data:
            shuffled_labels = rng.permutation(fold_dat['labels'])
            r = spearmanr(shuffled_labels, fold_dat['preds']).statistic
            perm_rhos.append(r)
            
        null_mean_rhos.append(np.mean(perm_rhos))
    
    null_mean_rhos = np.array(null_mean_rhos)
    
    observed_mean_rho = np.mean(observed_rho)
    p_value_global = (np.sum(null_mean_rhos >= observed_mean_rho) + 1) / (n_permutations + 1)
    
    return p_value_global