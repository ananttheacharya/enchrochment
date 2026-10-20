"""Gradient Boosted Tree models: Classifier, Coverage Regressor, Label Cleaner, and Tuning."""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from .features import FEATURES

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAS_LGB = False


def train_classifier(Xtr, ytr, Xva, yva, params=None, seed=42):
    """Train gradient-boosted decision tree classifier."""
    t0 = time.time()
    if HAS_LGB:
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        model = lgb.LGBMClassifier(**default_params, n_estimators=600)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=True,
        )
        model.fit(Xtr, ytr)
        
    print(f"GBM classifier trained in {time.time() - t0:.1f}s")
    return model


def train_regressor(Xmix, cov_mix, seed=42):
    """Train continuous footprint coverage regressor (0.0 to 1.0)."""
    t0 = time.time()
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(Xmix, cov_mix)
    else:
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(Xmix, cov_mix)
    print(f"Coverage regressor trained in {time.time() - t0:.1f}s")
    return model


def clean_label_noise(Xtr, ytr, bids_tr, clean_thr=0.92, n_splits=5, seed=42):
    """Remove false negatives (unlabeled buildings) from training negatives via Out-Of-Fold predictions."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(ytr), dtype=np.float32)
    
    print("Running Out-Of-Fold label noise detection...")
    for fold, (idx_tr, idx_va) in enumerate(sgkf.split(Xtr, ytr, groups=bids_tr)):
        m = train_classifier(Xtr[idx_tr], ytr[idx_tr], Xtr[idx_va], ytr[idx_va], seed=seed + fold)
        oof_preds[idx_va] = m.predict_proba(Xtr[idx_va])[:, 1]
        
    # Pixels labeled 0 where model is overwhelmingly sure (>= clean_thr) that it is a building
    suspect_fn = (ytr == 0) & (oof_preds >= clean_thr)
    n_dropped = suspect_fn.sum()
    print(f"Detected {n_dropped:,} suspect false-negative pixels ({100 * n_dropped / (ytr == 0).sum():.2f}% of negatives). Dropping them.")
    
    keep = ~suspect_fn
    return Xtr[keep], ytr[keep], bids_tr[keep]


def find_optimal_threshold(model, Xva, yva):
    """Find probability threshold maximizing F1 on spatial validation set."""
    probs = model.predict_proba(Xva)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(yva, probs)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thr = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_f1 = float(f1_scores[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])
    auc = roc_auc_score(yva, probs)
    
    print(f"Validation Optimal Threshold: {best_thr:.3f} | F1: {best_f1:.3f} (Prec: {best_p:.3f}, Rec: {best_r:.3f}) | ROC-AUC: {auc:.3f}")
    return best_thr, {
        "threshold": best_thr, "f1": best_f1, "precision": best_p, "recall": best_r, "roc_auc": auc
    }


def save_gbm_models(cfg, clf, reg=None, meta=None):
    joblib.dump(clf, cfg.P("gbm_classifier.joblib"))
    if reg is not None:
        joblib.dump(reg, cfg.P("gbm_regressor.joblib"))
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=FEATURES[:clf.n_features_in_]).sort_values(ascending=False)
        fi.to_csv(cfg.P("gbm_feature_importance.csv"))
        print("\nTop 10 Feature Importances:")
        print(fi.head(10).to_string())
    print("Saved GBM models to", cfg.OUT_DIR)
