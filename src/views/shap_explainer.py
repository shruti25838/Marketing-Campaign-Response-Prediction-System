from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _to_dense(X):
    """Convert to dense numpy array for SHAP compatibility."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _get_explainer(model, X_sample):
    """Use TreeExplainer for tree models (no background data), else Explainer with background."""
    import shap
    X_sample = _to_dense(X_sample)
    # TreeExplainer for LightGBM, XGBoost (no data needed for tree_path_dependent)
    if hasattr(model, "feature_importances_"):
        try:
            return shap.TreeExplainer(model), X_sample
        except Exception:
            pass
    # Fallback: generic Explainer (use small background for speed)
    return shap.Explainer(model, X_sample), X_sample


def shap_global_explanation(
    model, X_transformed, feature_names: Optional[list] = None, save_path: Optional[str] = None
):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is not installed. Install it with `pip install shap`.") from exc

    X = _to_dense(X_transformed)
    if X.size == 0 or X.shape[0] == 0:
        raise ValueError("Empty feature matrix for SHAP")
    # Limit rows for SHAP (memory and speed)
    max_rows = 500
    if X.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X = X[idx]
    explainer, X = _get_explainer(model, X)
    shap_values = explainer(X)
    if feature_names is not None and hasattr(shap_values, "feature_names"):
        shap_values.feature_names = feature_names
    shap.plots.bar(shap_values, max_display=20, show=False)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return shap_values


def shap_local_explanation(
    model, X_transformed, row_index: int = 0, save_path: Optional[str] = None
):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is not installed. Install it with `pip install shap`.") from exc

    X = _to_dense(X_transformed)
    if X.size == 0 or X.shape[0] == 0:
        raise ValueError("Empty feature matrix for SHAP")
    row_index = min(int(row_index), X.shape[0] - 1)
    explainer, _ = _get_explainer(model, X)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[row_index], show=False)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return shap_values
