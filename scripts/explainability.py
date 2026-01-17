"""Explainability helpers: SHAP and waterfall utilities (stub)."""

import shap
import pandas as pd

def explain_sample(model, X):
    """Return SHAP values for a single sample (stub)."""
    expl = shap.Explainer(model)
    return expl(X)
