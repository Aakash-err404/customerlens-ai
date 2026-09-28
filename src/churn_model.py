"""Logistic-regression churn modelling.

All hyper-parameter search happens through cross-validation on the training
partition only. The test partition is touched exactly once, at the end, to
produce the reported metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import constants as C

# scikit-learn 1.8 deprecated the ``penalty`` argument in favour of
# ``l1_ratio``. ``penalty`` is still the portable spelling across 1.3 - 1.9, so
# it is used here on purpose and the deprecation notice is silenced rather than
# letting it flood the interface.
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=.*", category=UserWarning)

#: Scorer name -> human label, exposed in the UI.
SCORING_OPTIONS: dict[str, str] = {
    "roc_auc": "ROC-AUC (threshold independent)",
    "accuracy": "Accuracy",
    "f1": "F1 score",
    "balanced_accuracy": "Balanced accuracy",
}

SOLVERS = ("liblinear", "lbfgs", "saga")


# ==========================================================================
# Train / test split
# ==========================================================================
@dataclass
class SplitResult:
    """A stratified train/test split with the index bookkeeping kept intact."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    test_size: float
    random_state: int
    n_train: int
    n_test: int
    train_class_counts: dict[int, int]
    test_class_counts: dict[int, int]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["Test size", f"{self.test_size:.0%}"],
                ["Random state", str(self.random_state)],
                ["Training rows", f"{self.n_train:,}"],
                ["Test rows (never seen during training)", f"{self.n_test:,}"],
                [
                    "Training class balance",
                    ", ".join(f"{k}: {v:,}" for k, v in sorted(self.train_class_counts.items())),
                ],
                [
                    "Test class balance",
                    ", ".join(f"{k}: {v:,}" for k, v in sorted(self.test_class_counts.items())),
                ],
            ],
            columns=["Setting", "Value"],
        )


def make_stratified_split(
    y: Sequence[int] | pd.Series,
    *,
    test_size: float = C.DEFAULT_TEST_SIZE,
    random_state: int = C.RANDOM_STATE,
) -> SplitResult:
    """Stratified split so both classes stay proportionally represented."""
    values = np.asarray(y, dtype=int)
    if len(np.unique(values)) < 2:
        raise ValueError(
            "Target contains only one class. Churn prediction needs both a churned and a "
            "retained group - check the target column, or adjust the synthetic churn rule so "
            "that it produces both classes."
        )
    if len(values) < 20:
        raise ValueError(
            f"Only {len(values)} labelled record(s) are available. At least 20 are needed for a "
            "meaningful stratified train/test split."
        )

    indices = np.arange(len(values))
    test_size = float(min(max(test_size, 0.05), 0.5))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=values,
    )

    y_train = values[train_idx]
    y_test = values[test_idx]
    return SplitResult(
        train_idx=np.sort(train_idx),
        test_idx=np.sort(test_idx),
        y_train=y_train,
        y_test=y_test,
        test_size=test_size,
        random_state=random_state,
        n_train=len(train_idx),
        n_test=len(test_idx),
        train_class_counts={int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        test_class_counts={int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    )


# ==========================================================================
# Segment augmentation (Model B)
# ==========================================================================
@dataclass
class SegmentFeature:
    """One-hot encoding of the K-Means segment, fitted on training data only."""

    encoder: OneHotEncoder
    k: int
    trained_on: str

    def transform(self, segments: np.ndarray) -> np.ndarray:
        return np.asarray(self.encoder.transform(np.asarray(segments).reshape(-1, 1)))

    @property
    def column_names(self) -> list[str]:
        return [f"segment_{int(c)}" for c in self.encoder.categories_[0]]


def fit_segment_feature(train_segments: np.ndarray, k: int) -> SegmentFeature:
    """Build the one-hot segment encoder from *training* segment labels only."""
    encoder = OneHotEncoder(
        categories=[np.arange(k)],
        handle_unknown="ignore",
        sparse_output=False,
        dtype="float64",
    )
    encoder.fit(np.asarray(train_segments).reshape(-1, 1))
    return SegmentFeature(encoder=encoder, k=k, trained_on="training partition only")


def attach_segments(matrix: np.ndarray, segments: np.ndarray, segment_feature: SegmentFeature) -> np.ndarray:
    """Append the one-hot segment block to a feature matrix."""
    one_hot = segment_feature.transform(segments)
    return np.hstack([matrix, one_hot])


# ==========================================================================
# Logistic regression
# ==========================================================================
def _penalty_is_deprecated() -> bool:
    """True on scikit-learn >= 1.8, where ``penalty`` gives way to ``l1_ratio``."""
    try:
        return tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 8)
    except (AttributeError, ValueError):
        return False


#: Regularisation axis and the C values worth searching per penalty.
#:
#: L1 is only searched at small C on purpose. At large C the L1 solution drifts
#: towards the unregularised fit while costing far more to converge - on a
#: 3,500 x 63 matrix, ``liblinear``/L1 at C=100 needed ~250 iterations and 15
#: seconds per fit, versus 20 iterations and 0.06 s for L2 at the same C. L2
#: and lbfgs cover the weakly-regularised end of the space at negligible cost.
_C_VALUES_L1 = [0.01, 0.1, 1.0]
_C_VALUES_L2 = [0.01, 0.1, 1.0, 10.0, 100.0]


def default_param_grid(prefix: str = "logistic__") -> list[dict[str, list[Any]]]:
    """Solver/penalty-aware grid - only valid combinations are searched.

    Keys are prefixed for the pipeline step name so the grid can be handed
    straight to ``GridSearchCV``.

    On scikit-learn 1.8 and newer the regularisation is expressed through
    ``l1_ratio`` (``0`` is L2, ``1`` is L1), which is both the supported spelling
    and measurably faster than the deprecated ``penalty`` argument. Older
    versions only understand ``penalty``, so the grid is generated to match
    whichever the installed version provides. Either way the required search
    space - C, penalty, solver and class weight - is covered.

    ``saga`` is deliberately left out: it is the slowest solver by a wide margin
    on wide matrices, and a two-model run has to stay interactive.
    """
    l1_key, l1_value = ("l1_ratio", 1.0) if _penalty_is_deprecated() else ("penalty", "l1")
    l2_key, l2_value = ("l1_ratio", 0.0) if _penalty_is_deprecated() else ("penalty", "l2")

    return [
        {
            f"{prefix}solver": ["liblinear"],
            f"{prefix}{l1_key}": [l1_value],
            f"{prefix}C": _C_VALUES_L1,
            f"{prefix}class_weight": [None, "balanced"],
        },
        {
            f"{prefix}solver": ["liblinear", "lbfgs"],
            f"{prefix}{l2_key}": [l2_value],
            f"{prefix}C": _C_VALUES_L2,
            f"{prefix}class_weight": [None, "balanced"],
        },
    ]


@dataclass
class FittedChurnModel:
    """A trained pipeline plus everything needed to explain and reuse it."""

    pipeline: Pipeline
    best_params: dict[str, Any]
    best_score: float
    scoring: str
    cv_folds: int
    n_train: int
    train_accuracy: float
    cv_accuracy: float
    cv_results: pd.DataFrame
    feature_names: list[str]
    oof_probabilities: np.ndarray
    label: str
    uses_segment: bool
    notes: list[str] = field(default_factory=list)

    @property
    def classifier(self) -> LogisticRegression:
        return self.pipeline.named_steps["logistic"]

    @property
    def best_params_flat(self) -> dict[str, Any]:
        """Best hyper-parameters with the pipeline prefix stripped."""
        return {k.replace("logistic__", ""): v for k, v in self.best_params.items()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = C.DEFAULT_THRESHOLD) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def cv_results_frame(self, top: int = 15) -> pd.DataFrame:
        results = self.cv_results.copy()
        # GridSearchCV prefixes every parameter with the pipeline step name, so
        # the columns arrive as ``param_logistic__C``. The regularisation
        # argument is named ``penalty`` on older scikit-learn versions and
        # ``l1_ratio`` on 1.8+, so both spellings are accepted and reported with
        # one readable label.
        wanted = {
            "param_logistic__C": "C",
            "param_logistic__penalty": "regularisation",
            "param_logistic__l1_ratio": "regularisation",
            "param_logistic__solver": "solver",
            "param_logistic__class_weight": "class_weight",
        }
        keep = [c for c in wanted if c in results.columns]
        keep += [
            c
            for c in ("mean_test_score", "std_test_score", "rank_test_score")
            if c in results.columns
        ]
        frame = results[keep].rename(
            columns={
                **wanted,
                "mean_test_score": "mean CV score",
                "std_test_score": "std CV score",
                "rank_test_score": "rank",
            }
        )
        frame = frame.sort_values("rank").head(top)
        if "regularisation" in frame.columns:
            if _penalty_is_deprecated():
                frame["regularisation"] = frame["regularisation"].map(
                    lambda v: "L1" if float(v) >= 1 else ("L2" if float(v) <= 0 else f"elasticnet {v}")
                )
            else:
                frame["regularisation"] = frame["regularisation"].astype(str).str.upper()
        if "class_weight" in frame.columns:
            frame["class_weight"] = frame["class_weight"].map(
                lambda v: "balanced" if v == "balanced" else "None"
            )
        return frame.reset_index(drop=True)


def _is_degenerate(pipeline: Pipeline, tol: float = 1e-8) -> bool:
    """True when the fitted logistic model shrunk every coefficient to zero."""
    classifier = pipeline.named_steps.get("logistic")
    if classifier is None or not hasattr(classifier, "coef_"):
        return False
    return bool(np.all(np.abs(classifier.coef_.ravel()) <= tol))


def _refit_non_degenerate(
    pipeline: Pipeline,
    search: GridSearchCV,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_candidates: int = 5,
):
    """Pick the best candidate from *this* search that is not a null model.

    The candidates are walked in descending cross-validated score and each is
    refitted, which reuses the search that already ran instead of launching a
    second one. Only the leading candidates are tried so a slow solver cannot
    turn this guard into the most expensive step in the pipeline.
    Returns ``(fitted_pipeline, best_params, best_score)`` or ``None`` when no
    leading combination avoids the collapse.
    """
    results = search.cv_results_
    order = np.argsort(results["mean_test_score"])[::-1][:max_candidates]
    for idx in order:
        params = {
            key[len("param_") :]: values[idx] for key, values in results.items() if key.startswith("param_")
        }
        try:
            candidate = clone(pipeline).set_params(**params).fit(X_train, y_train)
        except (ValueError, TypeError):
            continue
        if not _is_degenerate(candidate):
            return candidate, params, float(results["mean_test_score"][idx])
    return None


def train_logistic_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    label: str = "Logistic Regression",
    scoring: str = "roc_auc",
    cv_folds: int = C.DEFAULT_CV_FOLDS,
    random_state: int = C.RANDOM_STATE,
    param_grid: list[dict[str, list[Any]]] | None = None,
    uses_segment: bool = False,
    feature_names: Sequence[str] | None = None,
    max_iter: int = 5000,
    compute_oof: bool = True,
    n_jobs: int = -1,
) -> FittedChurnModel:
    """Tune and fit a logistic regression with cross-validated grid search."""
    y_train = np.asarray(y_train, dtype=int)
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "The training partition contains only one class. Increase the number of records, "
            "adjust the test size, or change the churn rule so both classes are represented."
        )
    notes: list[str] = []

    min_class = int(np.min(np.bincount(y_train)))
    folds = int(max(2, min(cv_folds, min_class)))
    if folds < cv_folds:
        notes.append(
            f"Cross-validation folds were reduced from {cv_folds} to {folds} because the "
            f"smallest training class only has {min_class} member(s)."
        )

    pipeline = Pipeline(
        [
            (
                "logistic",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            )
        ]
    )

    grid = param_grid if param_grid is not None else default_param_grid()
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    # A strongly regularised L1 fit can shrink every coefficient to exactly zero,
    # which predicts one constant class and is technically the "best" grid
    # candidate when the data carries no signal. Such a null model is useless to
    # a reader, so it is detected and replaced by the best non-degenerate
    # candidate from the same search.
    if _is_degenerate(best) and len(search.cv_results_) > 1:
        refit = _refit_non_degenerate(pipeline, search, X_train, y_train)
        if refit is not None:
            best, best_params, best_score = refit
            notes.append(
                "The cross-validated winner shrank every coefficient to zero, which makes it "
                "predict a single constant class. The best non-degenerate combination from the "
                "same search was used instead; the reported scores still come from held-out data."
            )
        else:
            notes.append(
                "Every combination in the grid shrank the model to a constant predictor, which "
                "means the training data contains no learnable separation between the classes."
            )
    else:
        best_params = search.best_params_
        best_score = float(search.best_score_)

    train_accuracy = float((best.predict(X_train) == y_train).mean())
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    cv_accuracy = float(
        cross_val_score(clone(best), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=n_jobs).mean()
    )

    # Out-of-fold probabilities come from the *training* partition only, using
    # the hyper-parameters chosen above. The test set is not involved.
    oof = np.zeros(len(y_train), dtype="float64")
    if compute_oof and folds >= 2:
        oof = cross_val_predict(
            clone(best), X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]

    return FittedChurnModel(
        pipeline=best,
        best_params={k.replace("logistic__", ""): v for k, v in best_params.items()},
        best_score=best_score,
        scoring=scoring,
        cv_folds=folds,
        n_train=len(y_train),
        train_accuracy=train_accuracy,
        cv_accuracy=cv_accuracy,
        cv_results=pd.DataFrame(search.cv_results_),
        feature_names=list(feature_names or []),
        oof_probabilities=oof,
        label=label,
        uses_segment=uses_segment,
        notes=notes,
    )


# ==========================================================================
# Prediction and risk banding
# ==========================================================================
def predict_churn(
    model: FittedChurnModel, X: np.ndarray, *, threshold: float = C.DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Score every record and label it at the chosen decision threshold."""
    probabilities = model.predict_proba(X)
    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": (probabilities >= threshold).astype(int),
        }
    )


def generate_risk_categories(
    probabilities: Sequence[float],
    *,
    low_max: float = C.DEFAULT_RISK_CUTOFFS[0],
    high_min: float = C.DEFAULT_RISK_CUTOFFS[1],
) -> np.ndarray:
    """Band probabilities into transparent, user-configurable risk levels.

    These are descriptive bands on a model score. They are not statistically
    validated guarantees of any business, financial or clinical outcome.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    return np.where(
        probabilities >= high_min,
        "High",
        np.where(probabilities >= low_max, "Medium", "Low"),
    )


def risk_threshold_table(low_max: float, high_min: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Low", f"probability < {low_max:.0%}", "No action triggered by the model"],
            ["Medium", f"{low_max:.0%} <= probability < {high_min:.0%}", "Worth reviewing"],
            ["High", f"probability >= {high_min:.0%}", "Highest modelled churn risk"],
        ],
        columns=["Risk level", "Rule", "Interpretation"],
    )


def choose_threshold(
    probabilities: np.ndarray, y_true: np.ndarray, *, objective: str = "f1"
) -> tuple[float, pd.DataFrame]:
    """Pick a decision threshold from *training* out-of-fold probabilities.

    This is only ever called with out-of-fold predictions from the training
    partition, so the test set remains untouched.
    """
    from .evaluation import threshold_sweep

    sweep = threshold_sweep(probabilities, y_true)
    if objective == "accuracy":
        column = "accuracy"
    elif objective == "balanced_accuracy":
        column = "balanced_accuracy"
    else:
        column = "f1"
    best_row = sweep.sort_values(column, ascending=False).iloc[0]
    return float(best_row["threshold"]), sweep


# ==========================================================================
# Coefficients
# ==========================================================================
def prettify_feature_name(raw: str) -> str:
    """Turn a ``ColumnTransformer`` output name into something readable.

    ``cat__Promotion_Response_Ignored`` becomes ``Promotion_Response = Ignored``
    and ``num__Total_Spend`` becomes ``Total_Spend``.
    """
    name = str(raw)
    if "__" in name:
        block, name = name.split("__", 1)
        if block == "cat" and "_" in name:
            base, _, level = name.rpartition("_")
            if base and level:
                return f"{base} = {level}"
        if block == "num":
            return name
        if block:
            return f"{name} [{block}]"
    return name


def feature_coefficients(
    model: FittedChurnModel, feature_names: Sequence[str] | None = None
) -> pd.DataFrame:
    """Readable logistic-regression coefficients.

    Coefficients describe association with the predicted probability. They are
    not causal effects.
    """
    names = list(feature_names or model.feature_names)
    coefs = model.classifier.coef_.ravel()
    if not names or len(names) != len(coefs):
        names = [f"feature_{i}" for i in range(len(coefs))]

    frame = pd.DataFrame(
        {
            "Feature": [prettify_feature_name(n) for n in names],
            "Coefficient": coefs,
            "Magnitude": np.abs(coefs),
            "Direction": np.where(
                coefs > 0,
                "Higher predicted churn probability",
                "Lower predicted churn probability",
            ),
        }
    )
    return frame.sort_values("Magnitude", ascending=False).reset_index(drop=True)


