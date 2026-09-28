"""CUSTOMERLENS AI - application package.

Pure-Python / pandas / scikit-learn logic only. No Streamlit imports live in
this package so the ML layer stays independently testable and reusable.
"""

from __future__ import annotations

__all__ = [
    "constants",
    "data_detection",
    "data_processing",
    "feature_engineering",
    "clustering",
    "churn_model",
    "evaluation",
    "visualization",
]
