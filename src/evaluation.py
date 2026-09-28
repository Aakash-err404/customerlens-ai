"""Model evaluation, comparison and leakage self-checks.

Everything in this module is descriptive. No metric is adjusted, threshold
tuned on the test set, or label modified in order to reach a target number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import constants as C

METRIC_ROWS = (
    "Accuracy",
    "Balanced accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "ROC-AUC",
    "Log loss",
)


# ==========================================================================
# Core evaluation
# ==========================================================================
@dataclass
class EvaluationResult:
    """Metrics and curves for one model on one held-out partition."""

    label: str
    threshold: float
    metrics: dict[str, float]
    confusion: np.ndarray
    classification_report: str
    fpr: np.ndarray
    tpr: np.ndarray
    roc_auc: float
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    uses_segment: bool = False
    notes: list[str] = field(default_factory=list)

    def metric_rows(self) -> list[dict[str, Any]]:
        return [{"Metric": k, "Value": v} for k, v in self.metrics.items()]

    def confusion_frame(self) -> pd.DataFrame:
        tn, fp, fn, tp = (int(v) for v in self.confusion.ravel())
        return pd.DataFrame(
            [
                {
                    "": "Predicted: no churn (0)",
                    "Actual: no churn (0)": tn,
                    "Actual: churn (1)": fn,
                },
                {
                    "": "Predicted: churn (1)",
                    "Actual: no churn (0)": fp,
                    "Actual: churn (1)": tp,
                },
            ]
        ).set_index("")

    def error_breakdown(self) -> pd.DataFrame:
        tn, fp, fn, tp = (int(v) for v in self.confusion.ravel())
        return pd.DataFrame(
            [
                [
                    "True positive",
                    tp,
                    "Predicted to churn and actually churned - correctly targeted for retention.",
                ],
                [
                    "False positive",
                    fp,
                    "Predicted to churn but the customer does NOT churn - wasted retention effort.",
                ],
                [
                    "False negative",
                    fn,
                    "Predicted not to churn but the customer actually churns - a missed save.",
                ],
                [
                    "True negative",
                    tn,
                    "Predicted not to churn and the customer does not churn - correct.",
                ],
            ],
            columns=["Outcome", "Count", "Meaning for a retention campaign"],
        )


def evaluate_model(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    label: str = "Model",
    threshold: float = C.DEFAULT_THRESHOLD,
    uses_segment: bool = False,
    train_accuracy: float | None = None,
    cv_accuracy: float | None = None,
) -> EvaluationResult:
    """Score a model at a fixed decision threshold.

    The threshold is applied exactly as configured - it is never changed to
    improve any reported number.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype="float64")
    y_pred = (probabilities >= threshold).astype(int).ravel()

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "The evaluation set contains only one class, so precision, recall, F1 and ROC-AUC "
            "cannot be computed. Check the target column and the train/test split."
        )

    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "Log loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }
    if train_accuracy is not None:
        metrics["Training accuracy"] = float(train_accuracy)
    if cv_accuracy is not None:
        metrics["Cross-validation accuracy"] = float(cv_accuracy)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["no churn (0)", "churn (1)"], zero_division=0
    )

    return EvaluationResult(
        label=label,
        threshold=threshold,
        metrics=metrics,
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        classification_report=report,
        fpr=fpr,
        tpr=tpr,
        roc_auc=metrics["ROC-AUC"],
        precision_curve=precision,
        recall_curve=recall,
        probabilities=probabilities,
        y_true=y_true,
        y_pred=y_pred,
        uses_segment=uses_segment,
    )


# ==========================================================================
# Threshold behaviour
# ==========================================================================
def threshold_sweep(
    probabilities: Sequence[float],
    y_true: Sequence[int],
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """How precision / recall / F1 / accuracy move with the decision threshold."""
    probabilities = np.asarray(probabilities, dtype="float64")
    y_true = np.asarray(y_true, dtype=int)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for t in thresholds:
        pred = (probabilities >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(accuracy_score(y_true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# Model comparison
# ==========================================================================
def majority_baseline(y_true: Sequence[int]) -> EvaluationResult:
    """The trivial predictor that always names the more common class.

    A model is only worth reporting if it beats this. Without the row, an
    accuracy of 51% on a 67/33 split looks like a weak model rather than the
    clear statement it is: worse than answering "no" every time.
    """
    y = np.asarray(y_true, dtype=int)
    majority = int(np.bincount(y).argmax()) if y.size else 0
    # Encoding the majority class as probability 1.0/0.0 makes the confusion
    # matrix and the metrics come out of the same code path as a real model.
    predicted = np.full(len(y), 1.0 if majority else 0.0, dtype=float)
    return evaluate_model(
        y_true=y,
        probabilities=predicted,
        label="Majority-class baseline",
        threshold=0.5,
        uses_segment=False,
    )


def compare_models(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side metric table for every evaluated model.

    The majority-class baseline is placed first when it is supplied, so the two
    tuned models are compared against a reference that requires no modelling.
    """
    keys = ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]
    ordered = list(results)
    if any(r.label == "Majority-class baseline" for r in ordered):
        baseline = [r for r in ordered if r.label == "Majority-class baseline"]
        ordered = baseline + [r for r in ordered if r.label != "Majority-class baseline"]
    frame = pd.DataFrame(
        {r.label: {k: r.metrics.get(k, float("nan")) for k in keys} for r in ordered}
    ).T
    tuned = [r for r in ordered if r.label != "Majority-class baseline"]
    if len(tuned) == 2:
        a, b = tuned[0], tuned[1]
        delta = (frame.loc[b.label] - frame.loc[a.label]).rename(
            f"Change ({b.label} - {a.label})"
        )
        frame = pd.concat([frame, delta.to_frame().T])
    return frame.reset_index().rename(columns={"index": "Model"})


def _model_pair(tuned: pd.DataFrame) -> tuple[str, str] | None:
    """The two tuned models to contrast, skipping the reference baseline row."""
    models = [str(name) for name in tuned["Model"] if name != "Majority-class baseline"]
    return (models[0], models[1]) if len(models) >= 2 else None


def comparison_verdict(frame: pd.DataFrame) -> str:
    """Plain-language summary of whether segmentation helped."""
    if "Model" not in frame.columns or len(frame) < 3:
        return ""

    change_rows = frame[frame["Model"].astype(str).str.startswith("Change (")]
    if change_rows.empty:
        return ""
    delta_row = change_rows.iloc[0]
    tuned = frame[~frame["Model"].astype(str).str.startswith("Change (")]
    if len(tuned) < 2:
        return ""

    pair = _model_pair(tuned)
    if pair is None:
        return ""
    base_name, tuned_name = pair
    delta_acc = float(delta_row.get("Accuracy", 0.0))
    delta_auc = float(delta_row.get("ROC-AUC", 0.0))

    def _word(delta: float, tolerance: float) -> str:
        if abs(delta) < tolerance:
            return "was essentially unchanged"
        return "improved" if delta > 0 else "reduced"

    lines = [
        f"Adding the K-Means segment {_word(delta_acc, 0.005)} test accuracy by {delta_acc:+.2%} and "
        f"{_word(delta_auc, 0.005)} ROC-AUC by {delta_auc:+.4f}, relative to {base_name}."
    ]
    if abs(delta_acc) < 0.005 and abs(delta_auc) < 0.005:
        lines.append(
            "Both changes are within noise for a test set of this size, so neither "
            "configuration can be called better on this evidence."
        )
    else:
        better = tuned_name if delta_acc > 0 else base_name
        lines.append(
            f"On this dataset and this split, {better} scores higher. That is an observation about "
            "this dataset, not a general rule: segmentation only helps when the clusters carry "
            "information the raw features do not already expose to a linear model."
        )
    return " ".join(lines)


def baseline_verdict(frame: pd.DataFrame) -> str:
    """State plainly whether the tuned models beat the trivial predictor."""
    if "Model" not in frame.columns:
        return ""
    rows = {str(r["Model"]): r for _, r in frame.iterrows()}
    baseline = rows.get("Majority-class baseline")
    if baseline is None:
        return ""
    base_acc = float(baseline.get("Accuracy", float("nan")))
    tuned = [
        (name, float(row.get("Accuracy", float("nan"))))
        for name, row in rows.items()
        if name != "Majority-class baseline" and not name.startswith("Change (")
    ]
    if not tuned:
        return ""
    best_name, best_acc = max(tuned, key=lambda item: item[1])
    parts = [
        f"The majority-class baseline reaches {base_acc:.2%} accuracy without using a single feature. "
    ]
    if best_acc <= base_acc:
        parts.append(
            f"The best model here reaches {best_acc:.2%}, which does not beat it. On this dataset the "
            "features carry no usable signal, so no threshold or model choice can honestly produce a "
            "better result."
        )
    else:
        parts.append(
            f"The best model reaches {best_acc:.2%} ({best_name}), which is "
            f"{best_acc - base_acc:+.2%} against that reference."
        )
    return "".join(parts)


# ==========================================================================
# Leakage and honesty self-checks
# ==========================================================================
@dataclass
class DiagnosticReport:
    """Automated sanity checks over the finished pipeline."""

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    univariate: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspicious_features: list[str] = field(default_factory=list)
    best_univariate_auc: float = 0.0
    train_test_gap: float = 0.0
    signal_detected: bool = True


def univariate_target_association(
    X: pd.DataFrame, y: Sequence[int], *, sample: int = 4000, random_state: int = C.RANDOM_STATE
) -> pd.DataFrame:
    """Rank features by how well a *single* feature separates the target.

    A feature with a very high univariate AUC is the classic signature of a
    target-derived feature, so this doubles as a leakage detector.
    """
    y = np.asarray(y, dtype=int)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["Feature", "Univariate AUC", "Direction"])
    frame = X
    if len(frame) > sample:
        frame = frame.sample(sample, random_state=random_state)

    rows = []
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        # Keep the feature and the label row-aligned: the sub-sample above
        # shuffled ``frame`` but ``y`` is still in its original order.
        aligned = pd.concat([values.rename("x"), pd.Series(y, index=X.index, name="y")], axis=1).dropna()
        if len(aligned) < 10 or aligned["x"].nunique() < 2 or aligned["y"].nunique() < 2:
            continue
        try:
            auc = float(roc_auc_score(aligned["y"].to_numpy(), aligned["x"].to_numpy()))
        except ValueError:
            continue
        auc = max(auc, 1.0 - auc)
        rows.append(
            {
                "Feature": col,
                "Univariate AUC": auc,
                "Direction": "separates" if auc >= 0.6 else "no separation",
            }
        )
    return pd.DataFrame(rows).sort_values("Univariate AUC", ascending=False).reset_index(drop=True)


def run_diagnostics(
    *,
    result: EvaluationResult,
    train_accuracy: float,
    cv_accuracy: float,
    univariate: pd.DataFrame,
    excluded_features: Sequence[str],
    target_origin: str,
) -> DiagnosticReport:
    """Combine the checks into one auditable verdict."""
    report = DiagnosticReport(univariate=univariate)

    test_accuracy = float(result.metrics["Accuracy"])
    report.train_test_gap = float(train_accuracy - test_accuracy)

    if not univariate.empty:
        report.best_univariate_auc = float(univariate["Univariate AUC"].max())
        report.suspicious_features = univariate.loc[
            univariate["Univariate AUC"] >= C.SUSPICIOUS_UNIVARIATE_AUC, "Feature"
        ].tolist()
        weak = float(univariate["Univariate AUC"].max()) < 0.60
        if weak:
            report.signal_detected = False

    if test_accuracy >= C.SUSPICIOUS_ACCURACY or result.roc_auc >= C.SUSPICIOUS_ACCURACY:
        report.warnings.append(
            "Unusually high performance detected. Review the feature pipeline for possible data "
            "leakage or target-derived features."
        )

    if report.suspicious_features:
        report.warnings.append(
            "The following features separate the target almost perfectly on their own, which is "
            f"the classic signature of a target-derived feature: {', '.join(report.suspicious_features)}. "
            "Check that they were not used to build the label."
        )

    if report.train_test_gap > C.SUSPICIOUS_ACCURACY_GAP:
        report.warnings.append(
            f"Training accuracy ({train_accuracy:.1%}) is much higher than test accuracy "
            f"({test_accuracy:.1%}). The model is memorising the training partition."
        )
    elif train_accuracy > cv_accuracy + 0.05:
        report.notes.append(
            f"When the model was being tested during practice it got {cv_accuracy:.1%} right, "
            f"but {train_accuracy:.1%} right on the customers it had already seen. That gap is "
            "common and worth keeping an eye on."
        )

    if not report.signal_detected:
        report.warnings.append(
            "No pattern was found connecting what we know about these customers to whether they "
            "churn. Not one of the details on its own separates leavers from stayers by even a "
            "little, so a score close to the overall churn rate is the best any model could "
            "honestly manage on this data. Either the things that decide who leaves were not "
            "included in the file, or what counts as churn here is not driven by them at all."
        )

    if target_origin == "synthetic":
        report.warnings.append(
            "The churn label is synthetic, so the reported performance measures how well the "
            "model reproduces a rule you defined - it is not evidence of real-world predictive "
            "power."
        )

    if excluded_features:
        report.notes.append(
            "Some columns were deliberately kept out of the model because they give the answer "
            f"away, so they cannot be used to predict it: {', '.join(excluded_features)}."
        )

    return report


def performance_banner(test_accuracy: float) -> tuple[str, str]:
    """Compare test accuracy against the 85% benchmark without inflating it."""
    if test_accuracy >= C.ACCURACY_BENCHMARK:
        return (
            "Performance target achieved",
            f"Unseen test accuracy is {test_accuracy:.2%}, at or above the "
            f"{C.ACCURACY_BENCHMARK:.0%} benchmark.",
        )
    return (
        "Performance target not achieved",
        f"Unseen test accuracy is {test_accuracy:.2%}, below the {C.ACCURACY_BENCHMARK:.0%} "
        "benchmark. The result is reported exactly as measured - no data, label or threshold "
        "was adjusted to change it.",
    )


def evaluation_summary_frame(
    results: Sequence[EvaluationResult],
    *,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """A tidy, downloadable summary of every evaluated model."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "Model": result.label,
            "Decision threshold": result.threshold,
            "Uses K-Means segment": result.uses_segment,
        }
        row.update({k: round(float(v), 6) for k, v in result.metrics.items()})
        tn, fp, fn, tp = (int(v) for v in result.confusion.ravel())
        row.update(
            {
                "True positives": tp,
                "False positives": fp,
                "False negatives": fn,
                "True negatives": tn,
            }
        )
        rows.append(row)
    if extra:
        for key, value in extra.items():
            for row in rows:
                row[key] = value
    return pd.DataFrame(rows)
