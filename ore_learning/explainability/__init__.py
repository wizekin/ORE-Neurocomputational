"""SHAP/LIME/counterfactual diagnostic records (Section 5, Steps 27-31)."""
from .counterfactual import (  # noqa: F401
    CounterfactualReport,
    ExplanationChannelError,
    ExplanationRecord,
    ExplanationRegistry,
    build_explanation_record,
    counterfactual,
    feature_importance_stability,
)
from .lime import LimeExplanation, LimeExplainer  # noqa: F401
from .shap import ShapExplanation, ShapExplainer, sampling_shapley_values  # noqa: F401

__all__ = (
    "CounterfactualReport",
    "ExplanationChannelError",
    "ExplanationRecord",
    "ExplanationRegistry",
    "LimeExplanation",
    "LimeExplainer",
    "ShapExplanation",
    "ShapExplainer",
    "build_explanation_record",
    "counterfactual",
    "feature_importance_stability",
    "sampling_shapley_values",
)
