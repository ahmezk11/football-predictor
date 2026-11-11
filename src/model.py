"""Model definitions: candidate estimators wrapped in leak-safe pipelines."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES

# Strong L2 regularisation. Cross-validated RPS is flat from C=0.001 to 0.01 on
# the larger folds and best at 0.002 overall (src/experiments.py); the original
# C=0.5 was chosen before there were correlated rating features to rein in.
LOGISTIC_C = 0.002


def _preprocessor(numeric: list[str], categorical: list[str], *, scale: bool) -> ColumnTransformer:
    # keep_empty_features keeps the column layout stable when an early
    # CV fold happens to contain no head-to-head history at all.
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_model(name: str, features: list[str], *, seed: int = 42) -> Pipeline:
    """Assemble a preprocessing + estimator pipeline for a named model."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    numeric = [c for c in features if c not in categorical]

    if name == "logistic":
        # Multinomial (softmax) is the default in modern scikit-learn.
        est = LogisticRegression(max_iter=3000, C=LOGISTIC_C, random_state=seed)
        pre = _preprocessor(numeric, categorical, scale=True)

    elif name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=600, max_depth=12, min_samples_leaf=25,
            max_features="sqrt", n_jobs=-1, random_state=seed)
        pre = _preprocessor(numeric, categorical, scale=False)

    elif name == "gradient_boosting":
        est = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12,
            random_state=seed)
        # HistGB handles NaN natively, but a shared preprocessor keeps the
        # comparison honest and lets the same code path serve every model.
        pre = _preprocessor(numeric, categorical, scale=False)

    else:
        raise ValueError(f"unknown model '{name}'")

    return Pipeline([("prep", pre), ("clf", est)])


MODEL_NAMES = ["logistic", "random_forest", "gradient_boosting"]


def class_order(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["clf"].classes_)


def probabilities(pipeline: Pipeline, X) -> np.ndarray:
    return pipeline.predict_proba(X)
