"""CUSTOMERLENS AI - Customer Segmentation & Churn Prediction.

A dataset-adaptive Streamlit application in two pages. **Results** carries the
segmentation and the churn model; **Final Summary** turns those measured numbers
into a written report. Nothing about the uploaded file is assumed: the schema,
the grain, the churn target and the usable features are all discovered at
runtime, and every judgement is shown with the reason behind it.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from src import clustering
from src import constants as C
from src import data_detection as dd
from src import data_processing as dp
from src import evaluation as ev
from src import feature_engineering as fe
from src import summary as sm
from src import visualization as vz
from src.churn_model import (
    attach_segments,
    choose_threshold,
    feature_coefficients,
    fit_segment_feature,
    generate_risk_categories,
    make_stratified_split,
    predict_churn,
    train_logistic_model,
)

APP_VERSION = "2.0.0"
APP_ROOT = Path(__file__).resolve().parent

#: How many rows of the risk list and the coefficient table are shown inline.
#: The complete versions are always one download away.
RISK_ROWS_SHOWN = 25
DRIVERS_SHOWN = 10

CSS = """
<style>
  .cl-header {padding: 1rem 0 .35rem 0;}
  .cl-title {font-size: 2rem; font-weight: 800; letter-spacing: .4px; margin: 0;}
  .cl-sub {opacity: .75; margin-top: .15rem; font-size: .95rem;}
  .cl-card {border: 1px solid rgba(128,128,128,.28); border-radius: .6rem;
            padding: .8rem 1rem; margin-bottom: .6rem;}
  footer, #MainMenu, header [data-testid="stStatusWidget"] {visibility: hidden;}
  .block-container {padding-top: 1.4rem;}
</style>
"""


def release_unused_memory() -> None:
    """Return freed heap to the operating system.

    Python gives memory back lazily and glibc keeps freed chunks in per-thread
    arenas, so a process can sit at its peak RSS long after the temporary
    DataFrames are gone. On Linux (including Streamlit Cloud) ``malloc_trim``
    flushes those arenas; on Windows the heap manager already returns memory,
    so only a garbage collection happens here.
    """
    gc.collect()
    if sys.platform == "linux":
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# ==========================================================================
# Session state containers
# ==========================================================================
@dataclass
class Analysis:
    """Everything derived from one uploaded file, computed once per file."""

    source_name: str
    read_meta: dict[str, Any]
    rows_in_file: int
    rows_analysed: int
    cleaning: dp.CleaningReport
    coerced: list[str]
    date_report: dict[str, dict[str, Any]]
    profile: dd.Profile
    schema: dd.TransactionSchema
    verdict: dd.DatasetVerdict
    customers: pd.DataFrame
    specs: pd.DataFrame
    agg_notes: list[str]
    customer_id: str | None
    target: dp.TargetBundle
    selection: fe.FeatureSelection

    @property
    def churn_rate(self) -> float:
        return self.target.churn_rate

    @property
    def rate_text(self) -> str:
        if self.target.origin == "unavailable":
            return "n/a"
        return f"{self.churn_rate:.1%}"

    @property
    def can_model(self) -> bool:
        return bool(self.target.y.nunique() > 1 and self.selection.numeric)

    @property
    def cluster_features(self) -> list[str]:
        return [*self.selection.numeric, *self.selection.boolean]


@dataclass
class Modelling:
    """Fitted models plus every number derived from the held-out test set."""

    split: Any
    preprocessor: fe.ColumnTransformer
    feature_names: list[str]
    scaler_report: pd.DataFrame
    dashboard_kmeans: clustering.KMeansModel
    dashboard_segments: np.ndarray
    dashboard_names: dict[int, str]
    dashboard_profile: clustering.ClusterProfile
    k_search: clustering.KSearchResult
    k: int
    kmeans: clustering.KMeansModel
    matrix_train: np.ndarray
    matrix_test: np.ndarray
    numeric_train: np.ndarray
    numeric_test: np.ndarray
    train_segments: np.ndarray
    test_segments: np.ndarray
    model_a: Any
    model_b: Any
    result_a: ev.EvaluationResult
    result_b: ev.EvaluationResult
    baseline: ev.EvaluationResult
    comparison: pd.DataFrame
    coefficients: pd.DataFrame
    univariate: pd.DataFrame
    diagnostics: ev.DiagnosticReport
    sweep: pd.DataFrame
    suggested_threshold: float


# ==========================================================================
# Cached stages
# ==========================================================================
@st.cache_data(show_spinner="Reading and profiling the file…")
def build_analysis(payload: bytes, source_name: str) -> Analysis:
    """Read, clean, profile, classify and aggregate.

    Everything here depends only on the file, never on a train/test split, so
    caching it cannot leak information between partitions.
    """
    raw, read_meta = dd.read_csv_bytes(payload)
    rows_in_file = int(len(raw))

    frame, cleaning = dp.clean_dataframe(raw)
    del raw
    frame, coerced, _ = dp.coerce_numeric_columns(frame)
    frame, date_report = dd.infer_datetime_columns(frame)
    rows_analysed = int(len(frame))

    profile = dd.profile_dataset(frame)
    schema = dd.detect_transaction_columns(frame, profile)
    verdict = dd.detect_dataset_type(frame, profile, schema)

    aggregation: fe.AggregationResult | None = None
    if verdict.is_transaction and schema.customer_id:
        aggregation = fe.aggregate_transactions(frame, schema, profile=profile)
        customers = aggregation.customers
        specs = aggregation.specs_frame()
        notes = list(aggregation.notes)
    else:
        customers = frame.copy()
        specs = pd.DataFrame(columns=["Feature", "Group", "Description", "Built from"])
        notes = []
    del frame

    customer_profile = dd.profile_dataset(customers)
    target_column, _ranked, _notes = dd.detect_target_column(customers, customer_profile)

    if target_column:
        target = dp.build_observed_target(customers, target_column)
    elif "recency_days" in customers.columns:
        target = dp.build_synthetic_inactivity_target(
            customers,
            customer_col=schema.customer_id,
            recency_col="recency_days",
            inactivity_days=C.DEFAULT_INACTIVITY_DAYS,
        )
    else:
        target = dp.TargetBundle(
            y=pd.Series(0, index=customers.index, dtype=int),
            column="",
            origin="unavailable",
            description=(
                "No churn label was found and no date column was detected, so a churn target "
                "cannot be constructed from this file."
            ),
            warnings=["Churn prediction is unavailable for this dataset."],
        )

    guard: dict[str, str] = {}
    for name in target.excluded_features:
        if name in customers.columns:
            guard[name] = (
                f"Used to build the churn target ({target.description}) Including it would let "
                "the model read the answer straight out of the question."
            )
    if schema.customer_id and schema.customer_id in customers.columns:
        guard[schema.customer_id] = (
            "Customer identifier - used for reporting only, never as a predictor."
        )

    selection = fe.select_features(
        customers,
        profile=customer_profile,
        customer_id=schema.customer_id,
        target_column=target.column or None,
        id_columns=[schema.customer_id] if schema.customer_id else [],
        extra_excluded=list(guard),
        extra_reasons=guard,
        specs=aggregation.specs if aggregation is not None else None,
    )

    release_unused_memory()

    return Analysis(
        source_name=source_name,
        read_meta=read_meta,
        rows_in_file=rows_in_file,
        rows_analysed=rows_analysed,
        cleaning=cleaning,
        coerced=list(coerced),
        date_report=date_report,
        profile=profile,
        schema=schema,
        verdict=verdict,
        customers=customers,
        specs=specs,
        agg_notes=notes,
        customer_id=schema.customer_id,
        target=target,
        selection=selection,
    )


@st.cache_data(show_spinner="Fitting models on the training partition…")
def run_modelling(
    customers: pd.DataFrame,
    selection: fe.FeatureSelection,
    y: np.ndarray,
    *,
    target_origin: str,
    test_size: float,
    k_override: int | None,
    cv_folds: int,
    random_state: int,
    n_jobs: int,
) -> Modelling:
    """Segment and model, with every fitted object learning from train only.

    The order is deliberate:

    1. split the customers first, stratified on the target;
    2. fit the imputer/scaler/encoder on the *training* rows alone;
    3. fit K-Means on the *training* rows alone, and only then use it to label
       the test rows;
    4. fit the one-hot segment encoder on the *training* segment labels alone;
    5. tune and fit both logistic regressions on the training rows;
    6. transform and score the test rows once, at the very end.

    The test partition is therefore never seen by any fitted component.
    """
    split = make_stratified_split(y, test_size=test_size, random_state=random_state)
    features = selection.all_features
    cluster_features = [*selection.numeric, *selection.boolean]
    categoricals = list(selection.categorical)

    # -- supervised path: everything below is training-only ----------------
    preprocessor = fe.build_preprocessor(selection)
    train_frame = customers.iloc[split.train_idx]
    test_frame = customers.iloc[split.test_idx]
    matrix_train, numeric_train, names = fe.prepare_matrix(train_frame, selection, preprocessor)
    matrix_test = fe.transform_matrix(test_frame, selection, preprocessor)
    numeric_test = fe.numeric_block(matrix_test, selection)

    search = clustering.find_optimal_k(numeric_train, random_state=random_state)
    k = int(k_override or search.recommended_k)
    kmeans = clustering.train_kmeans(
        numeric_train,
        k=k,
        feature_names=cluster_features,
        random_state=random_state,
        trained_on="training partition only",
    )
    train_segments = kmeans.predict(numeric_train)
    test_segments = kmeans.predict(numeric_test)

    segment_feature = fit_segment_feature(train_segments, k)
    matrix_b_train = attach_segments(matrix_train, train_segments, segment_feature)
    matrix_b_test = attach_segments(matrix_test, test_segments, segment_feature)

    model_a = train_logistic_model(
        matrix_train,
        split.y_train,
        label="Logistic Regression",
        cv_folds=cv_folds,
        random_state=random_state,
        feature_names=names,
        n_jobs=n_jobs,
    )
    model_b = train_logistic_model(
        matrix_b_train,
        split.y_train,
        label="Logistic + K-Means",
        cv_folds=cv_folds,
        random_state=random_state,
        feature_names=names + segment_feature.column_names,
        uses_segment=True,
        n_jobs=n_jobs,
    )

    result_a = ev.evaluate_model(
        split.y_test,
        model_a.predict_proba(matrix_test),
        label=model_a.label,
        train_accuracy=model_a.train_accuracy,
        cv_accuracy=model_a.cv_accuracy,
    )
    result_b = ev.evaluate_model(
        split.y_test,
        model_b.predict_proba(matrix_b_test),
        label=model_b.label,
        uses_segment=True,
        train_accuracy=model_b.train_accuracy,
        cv_accuracy=model_b.cv_accuracy,
    )
    baseline = ev.majority_baseline(split.y_test)

    # -- descriptive path: the segmentation shown on the dashboard ---------
    # This describes the whole customer base rather than predicting anything, so
    # it is fitted on every customer. The segmentation that feeds the churn
    # model is the training-only one above.
    _, full_numeric, _full_names = fe.prepare_matrix(
        customers, selection, fe.build_preprocessor(selection)
    )
    dashboard_k = int(k)
    dashboard_kmeans = clustering.train_kmeans(
        full_numeric,
        k=dashboard_k,
        feature_names=cluster_features,
        random_state=random_state,
        trained_on="all customers (descriptive segmentation only)",
    )
    dashboard_segments = dashboard_kmeans.predict(full_numeric)
    dashboard_profile = clustering.generate_cluster_profiles(
        full_numeric,
        dashboard_segments,
        feature_names=cluster_features,
        raw_frame=customers[features],
        categorical_features=categoricals,
    )
    dashboard_names = clustering.generate_segment_names(
        dashboard_profile, customers[features], feature_names=cluster_features
    )

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    univariate = ev.univariate_target_association(
        customers[features], y, random_state=random_state
    )
    diagnostics = ev.run_diagnostics(
        result=result_a,
        train_accuracy=model_a.train_accuracy,
        cv_accuracy=model_a.cv_accuracy,
        univariate=univariate,
        excluded_features=[row["Column"] for row in selection.excluded],
        target_origin=target_origin,
    )
    sweep = ev.threshold_sweep(model_a.oof_probabilities, split.y_train)
    suggested, _ = choose_threshold(model_a.oof_probabilities, split.y_train)

    release_unused_memory()

    return Modelling(
        split=split,
        preprocessor=preprocessor,
        feature_names=names,
        scaler_report=scaler.describe(cluster_features),
        dashboard_kmeans=dashboard_kmeans,
        dashboard_segments=dashboard_segments,
        dashboard_names=dashboard_names,
        dashboard_profile=dashboard_profile,
        k_search=search,
        k=k,
        kmeans=kmeans,
        matrix_train=matrix_train,
        matrix_test=matrix_test,
        numeric_train=numeric_train,
        numeric_test=numeric_test,
        train_segments=train_segments,
        test_segments=test_segments,
        model_a=model_a,
        model_b=model_b,
        result_a=result_a,
        result_b=result_b,
        baseline=baseline,
        comparison=ev.compare_models([result_a, result_b, baseline]),
        coefficients=feature_coefficients(model_a, names),
        univariate=univariate,
        diagnostics=diagnostics,
        sweep=sweep,
        suggested_threshold=float(suggested),
    )


# ==========================================================================
# Presentation helpers
# ==========================================================================
def header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="cl-header"><div class="cl-title">{title}</div>'
        f'<div class="cl-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def card(body: str) -> None:
    st.markdown(f'<div class="cl-card">{body}</div>', unsafe_allow_html=True)


def notice(level: str, message: str) -> None:
    """Render a message at the requested level: success, info, warning, error."""
    getattr(st, level)(message)


def metrics(items: list[tuple[str, str]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)


def table(frame: pd.DataFrame, key: str | None = None) -> None:
    st.dataframe(frame, width="stretch", hide_index=True, key=key)


def chart(figure: Any) -> None:
    st.plotly_chart(figure, width="stretch")


def download(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def download_text(label: str, text: str, filename: str, key: str) -> None:
    st.download_button(
        label,
        text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        key=key,
    )


def target_rule_text(target: dp.TargetBundle) -> str:
    """One human-readable line describing how a synthetic target was built."""
    if not target.rule:
        return ""
    if target.rule.get("type") == "inactivity":
        condition = f"inactive for {target.rule.get('inactivity_days')} days or more"
        if target.rule.get("require_low_frequency"):
            condition += (
                f" and low purchase frequency on {target.rule.get('frequency_column')}"
            )
        source = target.rule.get("recency_column")
        provenance = f" (measured from `{source}`)" if source else ""
        return f"Churned = {condition}{provenance}."
    if target.rule.get("type") == "feature_rule":
        direction = "at or above" if target.rule.get("direction") == "greater" else "below"
        return (
            f"Churned = `{target.rule.get('feature_column')}` {direction} "
            f"{target.rule.get('threshold'):.4g}."
        )
    return ""


def show_target_notes(target: dp.TargetBundle) -> None:
    if target.description:
        st.caption(target.description)
    rule = target_rule_text(target)
    if rule:
        st.caption(rule)
    if target.mapping:
        st.caption(
            "Original values encoded as "
            + ", ".join(f"`{value}` → {code}" for value, code in target.mapping.items())
            + "."
        )
    for warning in target.warnings:
        st.warning(warning)


def scored_frame(analysis: Analysis, modelling: Modelling) -> pd.DataFrame:
    """Test-partition scores, joined back to identifiers and segment names."""
    threshold = C.DEFAULT_THRESHOLD
    scored = predict_churn(modelling.model_a, modelling.matrix_test, threshold=threshold)
    scored = scored.rename(
        columns={"churn_probability": "Churn probability", "predicted_churn": "Predicted churn"}
    )
    scored.insert(
        0,
        "Segment",
        [
            modelling.dashboard_names.get(int(cluster), str(cluster))
            for cluster in modelling.dashboard_segments[modelling.split.test_idx]
        ],
    )
    if analysis.customer_id:
        identifiers = analysis.customers.iloc[modelling.split.test_idx][analysis.customer_id]
        scored.insert(0, "Customer", [str(value) for value in identifiers])
    scored["Risk category"] = generate_risk_categories(scored["Churn probability"].to_numpy())
    return scored


# ==========================================================================
# Derived result views
# ==========================================================================
def segment_table(analysis: Analysis, modelling: Modelling) -> pd.DataFrame:
    """Segment name, size, share and the churn rate observed across every customer.

    This is the descriptive clustering refitted on all analysed customers, so the
    counts add up to the whole base. The churn column is what actually happened to
    those customers, not a prediction.
    """
    frame = modelling.dashboard_profile.table[["Cluster", "Customers", "Share"]].copy()
    rates = pd.Series(analysis.target.y.to_numpy(dtype=int)).groupby(modelling.dashboard_segments).mean()
    frame["Churn rate"] = [float(rates.get(cluster, float("nan"))) for cluster in frame["Cluster"]]
    frame["Segment"] = [
        modelling.dashboard_names.get(int(cluster), str(cluster)) for cluster in frame["Cluster"]
    ]
    frame = frame[["Segment", "Customers", "Share", "Churn rate"]]
    return frame.sort_values("Customers", ascending=False).reset_index(drop=True)


def segment_churn_frame(analysis: Analysis, modelling: Modelling) -> pd.DataFrame:
    """Two columns for the churn-by-segment bar chart: label and observed churn."""
    return pd.DataFrame(
        {
            "Segment": [
                modelling.dashboard_names.get(int(cluster), str(cluster))
                for cluster in modelling.dashboard_segments
            ],
            "Churn": analysis.target.y.to_numpy(dtype=int),
        }
    )


def segment_traits(modelling: Modelling, top: int = 3, minimum: float = 0.15) -> dict[int, str]:
    """The features that most distinguish each cluster, in plain words."""
    z = modelling.dashboard_profile.z_table
    traits: dict[int, str] = {}
    for cluster in z.index:
        row = z.loc[cluster]
        ranked = row.abs().sort_values(ascending=False).head(top)
        parts = [
            f"{'high' if row[feature] > 0 else 'low'} {str(feature).replace('_', ' ')}"
            for feature, magnitude in ranked.items()
            if magnitude >= minimum
        ]
        traits[int(cluster)] = ", ".join(parts)
    return traits


def _plain_direction(value: object) -> str:
    """Turn the model's direction wording into something a non-analyst can read."""
    text = str(value).lower()
    if "higher" in text:
        return "More likely to churn"
    if "lower" in text:
        return "Less likely to churn"
    return str(value)


def comparison_table(modelling: Modelling) -> pd.DataFrame:
    """The model scoreboard, with column headings that say what each number means."""
    return modelling.comparison.rename(
        columns={
            "Model": "Approach",
            "Accuracy": "Overall right",
            "Balanced accuracy": "Fair to both kinds",
            "Precision": "Right when it flags",
            "Recall": "Churners it catches",
            "F1 score": "Both together",
            "ROC-AUC": "Ranking quality",
        }
    )


def driver_table(modelling: Modelling, limit: int = DRIVERS_SHOWN) -> pd.DataFrame:
    """The strongest drivers, with headings a non-analyst can read."""
    frame = modelling.coefficients.head(limit)[["Feature", "Coefficient", "Direction"]].copy()
    frame["Direction"] = frame["Direction"].map(_plain_direction)
    return frame.rename(
        columns={"Feature": "What", "Coefficient": "Strength", "Direction": "Effect"}
    )


def risk_table(scored: pd.DataFrame, limit: int = RISK_ROWS_SHOWN) -> pd.DataFrame:
    """The risk list, with headings a non-analyst can read."""
    keep = [column for column in scored.columns if column in {"Customer", "Segment", "Churn probability", "Predicted churn", "Risk category"}]
    frame = scored[keep].sort_values("Churn probability", ascending=False).head(limit)
    return frame.rename(
        columns={
            "Segment": "Group",
            "Churn probability": "Chance of churning",
            "Predicted churn": "Flagged at risk",
            "Risk category": "Risk level",
        }
    )


def segment_risk_frame(analysis: Analysis, modelling: Modelling) -> pd.DataFrame:
    """Held-out customers only: their segment, the predicted risk band and the outcome.

    Restricted to the test partition on purpose. Predicting a customer the model
    was trained on would flatter the result, so the relationship between groups
    and churn is only ever drawn from customers the model has never seen.

    The bands are thirds of this file's own predictions rather than fixed
    probability cut-offs. A model that barely separates churners pushes almost
    every prediction onto the same number, and fixed cut-offs would then put
    nearly everyone in one band and hide the ordering the model actually found.
    """
    scored = scored_frame(analysis, modelling)
    total = len(scored)
    if total >= 3:
        rank = scored["Churn probability"].rank(method="first")
        scored["Risk third"] = pd.cut(
            rank,
            bins=[0, total / 3, 2 * total / 3, total],
            labels=["Lower risk", "Middle", "Higher risk"],
            include_lowest=True,
        ).astype(str)
    else:
        scored["Risk third"] = "Middle"
    scored["Actually churned"] = modelling.split.y_test
    return scored[["Segment", "Risk third", "Actually churned"]]


def segment_agreement_frame(analysis: Analysis, modelling: Modelling) -> pd.DataFrame:
    """One row per held-out customer: the group, the predicted chance and the outcome."""
    scored = scored_frame(analysis, modelling)
    scored["Actually churned"] = modelling.split.y_test
    return scored[["Segment", "Churn probability", "Actually churned"]]


def short_reason(reason: str, limit: int = 100) -> str:
    """Reduce a stored exclusion reason to the claim it makes, never mid-sentence.

    Reasons in this project read ``<claim> - <explanation>``, sometimes with a
    long bracketed description in between. Splitting on the first period is not
    safe: the bracketed part often contains one, which would cut a sentence in
    half. When a candidate cut lands inside a bracket, the bracket is dropped
    instead and the claim before it is kept.
    """

    def balanced(value: str) -> bool:
        return value.count("(") == value.count(")")

    text = " ".join(str(reason).split())
    for stop in (" - ", ". "):
        if stop not in text:
            continue
        head = text.split(stop)[0]
        if balanced(head) and len(head) <= limit:
            return head.rstrip(" .,;")
        if "(" in head:
            return head.split("(")[0].strip(" ,;-")
    if not balanced(text) and "(" in text:
        text = text.split("(")[0].strip(" ,;-")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text.rstrip(" .,;")


#: Technical wording from the exclusion rules, restated for the written summary.
PLAIN_EXCLUSIONS = (
    ("high cardinality", "too many different values to be useful"),
    ("near-constant", "almost every customer has the same value"),
    ("customer identifier", "the customer number, used for reporting only"),
    ("churn target", "this is the answer we are trying to predict"),
    ("this is the churn target", "this is the answer we are trying to predict"),
    ("date", "a date, not a measurement"),
    ("constant", "the same value for everyone"),
)


def exclusion_phrases(excluded: list[dict[str, str]], limit: int = 4) -> list[str]:
    """Short, real reasons for a few held-back columns, for the written summary."""
    phrases = []
    for row in excluded[:limit]:
        name = str(row.get("Column", ""))
        reason = short_reason(row.get("Reason", ""))
        lowered = reason.lower()
        for marker, plain in PLAIN_EXCLUSIONS:
            if marker in lowered:
                reason = plain
                break
        phrases.append(f"`{name}` ({reason})" if reason else f"`{name}`")
    return phrases


def baseline_gap(modelling: Modelling) -> tuple[float, float]:
    """Test accuracy of the reported model and of the majority-class baseline."""
    return float(modelling.result_a.metrics["Accuracy"]), float(
        modelling.baseline.metrics["Accuracy"]
    )


def honest_verdict(modelling: Modelling) -> tuple[str, str]:
    """One sentence about whether the model is worth using, at face value.

    Accuracy is reported against the majority-class baseline, because a number
    that only clears an absolute bar is not evidence of anything: on a file where
    85% of customers churn, always predicting "churned" scores 85% using no
    features at all.
    """
    accuracy, baseline = baseline_gap(modelling)
    gap = (accuracy - baseline) * 100
    guessing = f"always guessing that the {baseline:.1%} majority stay"
    if gap > 0.5:
        level = "success"
        message = (
            f"The model gets {accuracy:.1%} of unseen customers right, against {baseline:.1%} for "
            f"{guessing}. That is {gap:.1f} points better, so its scores are worth acting on."
        )
    elif gap > -0.5:
        level = "warning"
        message = (
            f"The model gets {accuracy:.1%} of unseen customers right, no better than {guessing}. "
            "Since it adds nothing, do not use its scores to choose which customers to contact."
        )
    else:
        level = "error"
        message = (
            f"The model gets {accuracy:.1%} of unseen customers right, worse than the {baseline:.1%} "
            f"from {guessing}. It should not be used to score customers."
        )
    return level, message


# ==========================================================================
# Sidebar
# ==========================================================================
def dataset_panel() -> None:
    """Load a dataset from the sidebar: an upload, or a bundled example."""
    just_loaded: str | None = st.session_state.pop("just_loaded", None)
    if just_loaded:
        st.success(f"Loaded {just_loaded}")

    mode = st.radio("Dataset", ["Upload a file", "Bundled example"], horizontal=True)
    payload: bytes | None = None
    name = ""

    if mode == "Upload a file":
        uploaded = st.file_uploader(
            "CSV file", type=["csv", "txt", "tsv"], label_visibility="collapsed"
        )
        if uploaded is None:
            st.caption("Choose a CSV file to begin.")
            return
        payload, name = uploaded.getvalue(), uploaded.name
    else:
        examples = C.discover_examples(APP_ROOT)
        if not examples:
            st.warning("No example CSV files were found alongside the application.")
            return
        choice = st.selectbox("Example", list(examples), label_visibility="collapsed")
        payload, name = examples[choice].read_bytes(), choice

    st.caption(f"{name} · {len(payload) / 1024 / 1024:.1f} MB")
    if not st.button("Load dataset", type="primary", width="stretch"):
        return

    try:
        analysis = build_analysis(payload, name)
    except Exception as exc:  # surfaced to the user rather than a blank page
        st.error(f"The file could not be read: {exc}")
        return

    st.session_state["analysis"] = analysis
    st.session_state.pop("modelling", None)
    st.session_state["just_loaded"] = name
    st.rerun()


def sidebar() -> tuple[str, dict[str, Any]]:
    settings: dict[str, Any] = {}
    with st.sidebar:
        st.markdown(
            f'<div class="cl-title" style="font-size:1.35rem">{C.APP_NAME}</div>'
            f'<div class="cl-sub" style="font-size:.8rem">{C.APP_TAGLINE}</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio("Navigate", list(C.PAGES), label_visibility="collapsed")

        dataset_panel()

        analysis: Analysis | None = st.session_state.get("analysis")
        if analysis is not None:
            st.divider()
            st.caption("Current dataset")
            st.markdown(f"**{analysis.source_name}**")
            st.caption(
                f"{analysis.verdict.label} · {len(analysis.customers):,} customers · "
                f"target: {analysis.target.origin}"
            )

        if analysis is not None and analysis.can_model:
            st.divider()
            st.caption("Model settings")
            settings["test_size"] = float(
                st.slider(
                    "Test share",
                    0.10,
                    0.40,
                    C.DEFAULT_TEST_SIZE,
                    0.05,
                    help="Share of customers held out and never shown to any fitted step.",
                )
            )
            settings["cv_folds"] = int(st.slider("CV folds", 3, 10, C.DEFAULT_CV_FOLDS))
            settings["k"] = int(
                st.number_input(
                    "Override k (0 = auto)",
                    0,
                    20,
                    0,
                    help="Leave at 0 to accept the silhouette/elbow recommendation.",
                )
            )
            settings["random_state"] = int(
                st.number_input("Random seed", 0, 9999, C.RANDOM_STATE)
            )
            settings["n_jobs"] = int(
                st.select_slider(
                    "Parallel jobs",
                    options=[-1, 1],
                    value=-1,
                    format_func=lambda value: "all cores" if value == -1 else "single core",
                )
            )
            st.caption(
                "Imputation, scaling, encoding, K-Means and the regression are fitted on the "
                "training partition alone, and refit whenever a setting changes."
            )

        st.divider()
        if st.button("Reset session", width="stretch"):
            for key in ("analysis", "modelling", "settings", "just_loaded"):
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()
    return page, settings


def maybe_fit(analysis: Analysis, settings: dict[str, Any]) -> None:
    """Fit on demand, and automatically whenever the settings change."""
    changed = st.session_state.get("settings") != settings
    requested = st.sidebar.button("Fit models", type="primary", width="stretch")
    st.session_state["settings"] = settings

    if st.session_state.get("modelling") is None or changed or requested:
        st.session_state["modelling"] = run_modelling(
            analysis.customers,
            analysis.selection,
            analysis.target.y.to_numpy(dtype=int),
            target_origin=analysis.target.origin,
            test_size=float(settings["test_size"]),
            k_override=int(settings["k"]) or None,
            cv_folds=int(settings["cv_folds"]),
            random_state=int(settings["random_state"]),
            n_jobs=int(settings["n_jobs"]),
        )


def require_analysis() -> Analysis | None:
    analysis: Analysis | None = st.session_state.get("analysis")
    if analysis is None:
        card(
            "<b>No dataset loaded.</b><br>Use the panel in the sidebar to upload a CSV or pick "
            "one of the bundled examples. Everything on this page is derived from that file."
        )
    return analysis


def require_modelling() -> tuple[Analysis, Modelling] | None:
    analysis = require_analysis()
    if analysis is None:
        return None
    if not analysis.can_model:
        return None
    modelling: Modelling | None = st.session_state.get("modelling")
    if modelling is None:
        st.info("Models are fitting now - this page updates as soon as they are ready.")
    return (analysis, modelling) if modelling is not None else None


# ==========================================================================
# Page 1 - Results
# ==========================================================================
def details_expander(analysis: Analysis, modelling: Modelling | None) -> None:
    """The evidence behind the numbers, kept out of the way until it is wanted."""
    with st.expander("Data quality, exclusions and method"):
        st.markdown("**What was read**")
        st.dataframe(analysis.cleaning.as_frame(), width="stretch", hide_index=True)
        for reason in analysis.verdict.reasons:
            st.markdown(f"- {reason}")

        st.markdown("**Detected schema**")
        st.dataframe(
            pd.DataFrame(analysis.schema.as_rows()), width="stretch", hide_index=True
        )

        st.markdown("**Churn target**")
        show_target_notes(analysis.target)

        if analysis.agg_notes:
            st.markdown("**Aggregation notes**")
            for note in analysis.agg_notes:
                st.caption(f"• {note}")

        if modelling is not None:
            st.markdown("**Why this number of segments**")
            st.dataframe(
                modelling.k_search.to_frame(), width="stretch", hide_index=True
            )
            st.caption(modelling.k_search.reason)
            if modelling.k_search.warnings:
                for warning in modelling.k_search.warnings:
                    st.warning(warning)

            diagnostics = modelling.diagnostics
            if diagnostics.suspicious_features:
                st.error(
                    "These features separate the target almost perfectly, which usually means they "
                    "are derived from it: " + ", ".join(diagnostics.suspicious_features)
                )
            for note in diagnostics.notes:
                st.caption(note)

            st.markdown("**Model comparison in technical terms**")
            st.dataframe(modelling.comparison, width="stretch", hide_index=True)
            st.caption(ev.baseline_verdict(modelling.comparison))
            st.caption(ev.comparison_verdict(modelling.comparison))

        st.markdown("**Columns held back from the model**")
        if analysis.selection.excluded:
            st.dataframe(
                analysis.selection.excluded_frame(), width="stretch", hide_index=True
            )
        else:
            st.caption("No column had to be excluded.")

        method_and_limits()


def unmodellable_view(analysis: Analysis) -> None:
    """Shown when the file yields no usable churn label."""
    header("Results", f"{analysis.source_name} · no usable churn label")
    st.error(
        " ".join(analysis.target.warnings)
        or "This dataset cannot be modelled: no usable churn label and no numeric features."
    )
    st.caption(analysis.target.description)
    metrics(
        [
            ("Rows in file", f"{analysis.rows_in_file:,}"),
            ("Customers", f"{len(analysis.customers):,}"),
            ("Grain", analysis.verdict.label),
            ("Features found", f"{len(analysis.selection.all_features)}"),
        ]
    )
    st.write("")
    st.caption(
        "The dataset is still profiled below so you can see what the app understood from it, but "
        "no segments or churn scores can be produced without a label."
    )
    details_expander(analysis, None)


def page_results() -> None:
    analysis = require_analysis()
    if analysis is None:
        return
    if not analysis.can_model:
        unmodellable_view(analysis)
        return

    pair = require_modelling()
    if pair is None:
        return
    analysis, modelling = pair

    header(
        "Results",
        f"{analysis.source_name} · {analysis.verdict.label} · {len(analysis.customers):,} customers",
    )

    accuracy, baseline = baseline_gap(modelling)
    gap = (accuracy - baseline) * 100
    metrics(
        [
            ("Customers", f"{len(analysis.customers):,}"),
            ("Churn rate", analysis.rate_text),
            ("Groups found", f"{modelling.k}"),
            ("Right on new customers", f"{accuracy:.1%}"),
            ("vs always guessing", f"{gap:+.1f} points"),
        ]
    )
    st.write("")

    # -- the groups ------------------------------------------------------
    st.subheader("The customer groups")
    st.caption(
        f"We sorted all {len(analysis.customers):,} customers into {modelling.k} groups by who they "
        f"resemble, using only what they do, never the churn label. The churn rate column is the "
        "share of each group that actually churned."
    )
    table(segment_table(analysis, modelling))
    chart(vz.churn_rate_by_segment(segment_churn_frame(analysis, modelling), "Segment", "Churn"))

    # -- how the groups connect to the churn result -----------------------
    st.subheader("How the groups connect to churn")
    st.caption(
        f"These two charts use only the {modelling.split.n_test:,} customers the model never saw "
        "while learning, so nothing below is built on customers it already knew the answer for."
    )
    st.markdown("**Does the model agree with what happened?**")
    st.caption(
        "For each group, the blue bar is the average chance the model gave, and the orange bar is "
        "the share that really churned. Bars that line up mean the model is well calibrated for "
        "that group. A blue bar much longer than the orange one means the model is more "
        "pessimistic than reality for those customers, and the other way round means it is "
        "optimistic."
    )
    chart(
        vz.segment_prediction_vs_actual(
            segment_agreement_frame(analysis, modelling),
            "Segment",
            "Churn probability",
            "Actually churned",
        )
    )
    st.markdown("**How risk is spread inside each group**")
    st.caption(
        "Customers sorted by their predicted chance of leaving, split into equal thirds, so the bar "
        "shows how risk is distributed within a group. The white marker is the share who really "
        "churned. Where a group shows more risk than actually materialised, that is a group worth "
        "watching rather than acting on."
    )
    chart(
        vz.risk_mix_by_segment(
            segment_risk_frame(analysis, modelling), "Segment", "Risk third", "Actually churned"
        )
    )

    # -- churn model ------------------------------------------------------
    st.subheader("Churn prediction")
    st.caption(
        "How well the model does on customers it has never seen, next to the simplest possible "
        "alternative: always guessing that the majority group stays."
    )
    level, message = honest_verdict(modelling)
    notice(level, message)
    st.write("")
    table(comparison_table(modelling))
    st.caption(
        "Overall right is the share of unseen customers the approach labels correctly. Right when "
        "it flags is how often a customer it calls at risk really churned. Churners it catches is "
        "the share of real churners it found. Ranking quality runs from 0.5 (no better than a coin "
        "toss) to 1.0 (perfect ordering). The full-precision version is in the evidence section below."
    )
    chart(vz.roc_curve_figure([modelling.result_a]))

    # -- drivers ----------------------------------------------------------
    st.subheader("What makes customers more or less likely to churn")
    st.caption(
        "The things the model leans on most. A bar pointing right means customers with more of that "
        "attribute are more likely to churn; pointing left means less likely. These are patterns in "
        "your data, not proof that one thing causes the other."
    )
    chart(vz.feature_coefficient_figure(modelling.coefficients, limit=DRIVERS_SHOWN))
    table(driver_table(modelling))
    moved = abs(modelling.suggested_threshold - C.DEFAULT_THRESHOLD) > 1e-9
    st.caption(
        f"Customers are flagged as at risk at {C.DEFAULT_THRESHOLD:.0%} likelihood or higher. "
        + (
            f"A different cut-off of {modelling.suggested_threshold:.2f} fitted the training data "
            "slightly better, but it is shown for reference only - no number above was recalculated "
            "with it."
            if moved
            else "A search for a better cut-off on the training data agreed with this one."
        )
    )

    # -- the groups and the drivers together ------------------------------
    st.subheader("What makes each group different")
    st.caption(
        "Each row is one group; each column is something that sets that group apart from the "
        "average customer. Darker means further from average. The label on each column repeats "
        "whether more of that thing raises or lowers predicted churn, so you can read across: the "
        "cells that are both strong and marked as raising risk are what to act on."
    )
    chart(
        vz.segment_driver_heatmap(
            modelling.dashboard_profile.z_table,
            modelling.dashboard_names,
            modelling.coefficients,
            limit=DRIVERS_SHOWN,
        )
    )

    # -- who to act on ---------------------------------------------------
    st.subheader("Customers most likely to churn")
    st.caption(
        f"The {RISK_ROWS_SHOWN} highest-risk customers out of the {modelling.split.n_test:,} the "
        "model had not seen, most likely to churn first. The complete list is a download below."
    )
    table(risk_table(scored_frame(analysis, modelling)))

    details_expander(analysis, modelling)


# ==========================================================================
# Page 2 - Final Summary
# ==========================================================================
def summary_facts(analysis: Analysis, modelling: Modelling) -> sm.SummaryFacts:
    """Collect the measured values the written summary is allowed to state."""
    traits = segment_traits(modelling)
    rates = pd.Series(analysis.target.y.to_numpy(dtype=int)).groupby(modelling.dashboard_segments).mean()

    segments: list[sm.SegmentFact] = []
    for _, row in modelling.dashboard_profile.table.sort_values(
        "Customers", ascending=False
    ).iterrows():
        cluster = int(row["Cluster"])
        segments.append(
            sm.SegmentFact(
                name=modelling.dashboard_names.get(cluster, str(cluster)),
                customers=int(row["Customers"]),
                share=float(row["Share"]),
                churn_rate=float(rates.get(cluster, float("nan"))),
                traits=traits.get(cluster, ""),
            )
        )

    settings = st.session_state.get("settings") or {}
    excluded = analysis.selection.excluded
    accuracy, baseline = baseline_gap(modelling)
    first_sentence = (
        f"a split into {modelling.k_search.recommended_k} groups scored best out of the "
        "splits tried"
    )

    return sm.SummaryFacts(
        source_name=analysis.source_name,
        grain_label=analysis.verdict.label,
        rows_in_file=analysis.rows_in_file,
        rows_analysed=analysis.rows_analysed,
        customer_count=len(analysis.customers),
        columns_analysed=len(analysis.customers.columns),
        target_origin=analysis.target.origin,
        target_column=analysis.target.column or "",
        target_description=analysis.target.description,
        target_rule=target_rule_text(analysis.target),
        churn_rate=analysis.churn_rate if analysis.target.origin != "unavailable" else None,
        k=modelling.k,
        silhouette=modelling.dashboard_kmeans.silhouette,
        k_reason=first_sentence + ".",
        segments=segments,
        model_label=modelling.model_a.label,
        test_accuracy=accuracy,
        test_auc=float(modelling.result_a.metrics.get("ROC-AUC", float("nan"))),
        baseline_accuracy=baseline,
        beats_baseline=accuracy > baseline + 0.005,
        drivers=modelling.coefficients,
        warnings=[*analysis.target.warnings, *modelling.diagnostics.warnings],
        notes=[
            *modelling.diagnostics.notes,
            f"The segments are a descriptive clustering of all {len(analysis.customers):,} customers, "
            f"fitted without the churn label. The churn model used a separate clustering of the "
            f"{modelling.split.n_train:,} training customers only, so the held-out rows never "
            "influenced the model.",
        ],
        excluded_columns=len(excluded),
        excluded_examples=exclusion_phrases(excluded),
        train_rows=modelling.split.n_train,
        test_rows=modelling.split.n_test,
        test_size=float(settings.get("test_size", C.DEFAULT_TEST_SIZE)),
        random_state=int(settings.get("random_state", C.RANDOM_STATE)),
        threshold=C.DEFAULT_THRESHOLD,
        suggested_threshold=modelling.suggested_threshold,
        can_model=True,
    )


def unmodellable_facts(analysis: Analysis) -> sm.SummaryFacts:
    """Summary facts for a file that yields no churn label at all."""
    return sm.SummaryFacts(
        source_name=analysis.source_name,
        grain_label=analysis.verdict.label,
        rows_in_file=analysis.rows_in_file,
        rows_analysed=analysis.rows_analysed,
        customer_count=len(analysis.customers),
        columns_analysed=len(analysis.customers.columns),
        target_origin=analysis.target.origin,
        target_description=analysis.target.description,
        warnings=list(analysis.target.warnings),
        excluded_columns=len(analysis.selection.excluded),
        excluded_examples=exclusion_phrases(analysis.selection.excluded),
        can_model=False,
    )


def page_summary() -> None:
    analysis = require_analysis()
    if analysis is None:
        return

    pair = require_modelling() if analysis.can_model else None
    if analysis.can_model and pair is None:
        return

    header("Final Summary", "Written from the numbers on the Results page.")
    if pair is None:
        facts = unmodellable_facts(analysis)
    else:
        facts = summary_facts(pair[0], pair[1])

    st.markdown(sm.build_summary(facts))

    if pair is not None:
        analysis, modelling = pair
        st.subheader("Segment profiles")
        st.caption(
            f"The per-segment averages behind the segment names, measured across all "
            f"{len(analysis.customers):,} analysed customers."
        )
        table(modelling.dashboard_profile.table)

    st.subheader("Downloads")
    st.caption("Everything the run produced, including the full-length versions of the tables "
               "shown above.")
    if pair is not None:
        analysis, modelling = pair
        left, right = st.columns(2)
        with left:
            download_text(
                "Summary (text)",
                sm.build_plain_summary(facts),
                "customerlens_summary.txt",
                "dl_summary_txt",
            )
            st.write("")
            download(
                "Segment summary",
                sm.summary_frame(facts),
                "customerlens_segment_summary.csv",
                "dl_seg_summary",
            )
            st.write("")
            download(
                "Segment profiles",
                modelling.dashboard_profile.table,
                "customerlens_segment_profiles.csv",
                "dl_profiles",
            )
            st.write("")
            download(
                "Churn scores",
                scored_frame(analysis, modelling).sort_values(
                    "Churn probability", ascending=False
                ),
                "customerlens_churn_scores.csv",
                "dl_scores",
            )
        with right:
            download(
                "Model comparison",
                modelling.comparison,
                "customerlens_model_comparison.csv",
                "dl_comp",
            )
            st.write("")
            download(
                "Feature coefficients",
                modelling.coefficients,
                "customerlens_coefficients.csv",
                "dl_coef",
            )
            st.write("")
            download(
                "Customer features",
                analysis.customers,
                "customerlens_customer_features.csv",
                "dl_cust2",
            )
            if analysis.selection.excluded:
                st.write("")
                download(
                    "Excluded columns and reasons",
                    analysis.selection.excluded_frame(),
                    "customerlens_excluded.csv",
                    "dl_excl",
                )
    else:
        download(
            "Customer features",
            analysis.customers,
            "customerlens_customer_features.csv",
            "dl_cust3",
        )


def method_and_limits() -> None:
    """Method, scope and stack, kept out of the main output on purpose."""
    st.markdown(
        """
1. **Read** - encoding and separator are detected from the bytes, never assumed.
2. **Profile** - every column is typed, and identifier and target roles are scored with the
   reasons recorded.
3. **Aggregate** - when the grain is a transaction, per-customer RFM, behaviour and value features
   are derived. A feature is only created when the columns it needs exist.
4. **Select** - identifiers, constants, near-constants, high-cardinality text and anything derived
   from the churn label are removed, each with a stated reason.
5. **Segment** - K-Means with k chosen by silhouette cross-checked against the elbow. A split
   where a couple of outliers form their own "cluster" is rejected as degenerate.
6. **Model** - the stratified split happens first; preprocessing, K-Means and the regression are
   fitted on the training partition alone and scored once on the test partition.
7. **Report** - metrics appear as measured, beside a majority-class baseline and a train/CV gap,
   with a plain-language note whenever a number is not meaningful.
"""
    )

    st.markdown(
        "**What this app will not do:** move the decision threshold to make accuracy look better; "
        "use the customer identifier, the target, or a feature derived from the target as a "
        "predictor; report a score above chance without saying so when the cross-validated result "
        "is at or below chance; invent a column the dataset does not contain."
    )
    st.markdown(
        "**On synthetic targets:** when no churn label exists the target is built from an "
        "inactivity rule. Performance then measures how faithfully the model reproduces a rule you "
        "chose - it is not evidence of real-world predictive power, and the target is labelled as "
        "synthetic everywhere it appears."
    )
    st.caption(
        f"{C.APP_NAME} {APP_VERSION} · streamlit · pandas {pd.__version__} · "
        f"numpy {np.__version__} · scikit-learn {sklearn.__version__} · plotly"
    )


PAGES: dict[str, Callable[[], None]] = {
    "Results": page_results,
    "Final Summary": page_summary,
}


def main() -> None:
    st.set_page_config(
        page_title=f"{C.APP_NAME} · {C.APP_TAGLINE}",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    page, settings = sidebar()

    analysis: Analysis | None = st.session_state.get("analysis")
    if analysis is not None and analysis.can_model:
        try:
            maybe_fit(analysis, settings)
        except Exception as exc:
            st.error(f"Modelling failed: {exc}")
            st.session_state.pop("modelling", None)

    PAGES[page]()


if __name__ == "__main__":
    main()
