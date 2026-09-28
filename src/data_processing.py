"""Data cleaning, type coercion and target construction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import constants as C
from .data_detection import ColumnMeta, Profile, TransactionSchema, encode_binary_target

_CURRENCY = re.compile(r"[^0-9eE+\-.,()%]")


@dataclass
class CleaningReport:
    """Human-readable record of everything the cleaning step changed."""

    initial_rows: int
    final_rows: int
    rows_dropped: int
    duplicate_rows_removed: int = 0
    columns_dropped_all_null: list[str] = field(default_factory=list)
    columns_dropped_constant: list[str] = field(default_factory=list)
    columns_coerced_numeric: list[str] = field(default_factory=list)
    columns_coerced_datetime: list[str] = field(default_factory=list)
    infinite_values_replaced: int = 0
    notes: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = [
            ("Rows before cleaning", f"{self.initial_rows:,}"),
            ("Rows after cleaning", f"{self.final_rows:,}"),
            ("Rows removed", f"{self.rows_dropped:,}"),
            ("Exact duplicate rows removed", f"{self.duplicate_rows_removed:,}"),
            ("All-null columns removed", ", ".join(self.columns_dropped_all_null) or "none"),
            ("Constant columns removed", ", ".join(self.columns_dropped_constant) or "none"),
            ("Text columns coerced to numeric", ", ".join(self.columns_coerced_numeric) or "none"),
            ("Text columns coerced to datetime", ", ".join(self.columns_coerced_datetime) or "none"),
            ("Infinite values replaced with missing", f"{self.infinite_values_replaced:,}"),
        ]
        rows.extend((f"Note {i + 1}", note) for i, note in enumerate(self.notes))
        return pd.DataFrame(rows, columns=["Step", "Result"])


def clean_dataframe(
    df: pd.DataFrame,
    *,
    strip_column_names: bool = True,
    strip_text_values: bool = True,
    drop_duplicates: bool = True,
    drop_all_null_columns: bool = True,
    drop_constant_columns: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Normalise a raw upload into a modelling-friendly frame.

    Deliberately conservative: nothing that carries information is removed
    silently, and every change is recorded in the returned report.
    """
    report = CleaningReport(initial_rows=len(df), final_rows=len(df), rows_dropped=0)

    frame = df
    if strip_column_names:
        renamed = {c: str(c).strip() for c in frame.columns}
        if renamed != dict(zip(frame.columns, frame.columns)):
            frame = frame.rename(columns=renamed)

    if strip_text_values:
        for col in frame.columns:
            if frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col]):
                frame[col] = frame[col].map(
                    lambda v: v.strip() if isinstance(v, str) else v
                )

    if drop_duplicates and len(frame):
        before = len(frame)
        frame = frame.drop_duplicates()
        report.duplicate_rows_removed = before - len(frame)
        if report.duplicate_rows_removed:
            report.notes.append(
                f"{report.duplicate_rows_removed:,} exactly duplicated rows were removed."
            )

    report.final_rows = len(frame)
    report.rows_dropped = report.initial_rows - report.final_rows

    if drop_all_null_columns:
        all_null = [c for c in frame.columns if frame[c].isna().all()]
        if all_null:
            frame = frame.drop(columns=all_null)
            report.columns_dropped_all_null = all_null
            report.notes.append(
                f"Removed {len(all_null)} column(s) that contained only missing values: "
                f"{', '.join(map(str, all_null))}."
            )

    if drop_constant_columns and len(frame):
        constant = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
        if constant:
            frame = frame.drop(columns=constant)
            report.columns_dropped_constant = constant
            report.notes.append(
                f"Removed {len(constant)} constant column(s) that carry no information: "
                f"{', '.join(map(str, constant))}."
            )

    numeric = frame.select_dtypes(include=[np.number]).columns
    if len(numeric):
        inf_mask = np.isinf(frame[numeric].to_numpy(dtype="float64", na_value=np.nan))
        n_inf = int(inf_mask.sum())
        if n_inf:
            report.infinite_values_replaced = n_inf
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
            report.notes.append(
                f"Replaced {n_inf:,} infinite numeric value(s) with missing values so that "
                "imputation and scaling behave correctly."
            )

    report.final_rows = len(frame)
    report.rows_dropped = report.initial_rows - report.final_rows
    return frame, report


def coerce_numeric_columns(
    df: pd.DataFrame, *, threshold: float = 0.9, max_unique: int = 2000
) -> tuple[pd.DataFrame, list[str], list[ColumnMeta]]:
    """Convert numeric-looking text columns (e.g. ``"1,234.56"``) to numbers.

    Datasets exported from spreadsheets frequently store measures as text.
    A column is only converted when the large majority of its values parse as
    numbers, which keeps genuine categorical columns untouched.
    """
    converted: list[str] = []
    frame = df
    metas: list[ColumnMeta] = []

    for col in frame.columns:
        series = frame[col]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty or non_null.nunique() > max_unique:
            continue
        cleaned = non_null.astype(str).str.replace(_CURRENCY, "", regex=True).str.replace(
            ",", "", regex=False
        )
        try:
            numbers = pd.to_numeric(cleaned, errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        success = float(numbers.notna().mean())
        if success < threshold:
            continue
        # Guard against converting a genuine text column of small integers.
        if non_null.nunique() <= 2 and success == 1.0 and len(non_null) == non_null.nunique():
            continue
        numeric = pd.to_numeric(
            series.astype(str).str.replace(_CURRENCY, "", regex=True).str.replace(",", "", regex=False),
            errors="coerce",
        )
        if col not in frame.columns:  # pragma: no cover - defensive
            continue
        frame[col] = numeric
        converted.append(col)
        metas.append(
            ColumnMeta(
                name=str(col),
                dtype="float64",
                kind="numeric",
                n_rows=len(frame),
                n_unique=int(numeric.nunique(dropna=True)),
                unique_ratio=float(numeric.nunique(dropna=True)) / max(len(frame), 1),
                unique_approximate=False,
                n_missing=int(numeric.isna().sum()),
                missing_pct=float(numeric.isna().mean()),
                n_infinite=0,
                is_constant=numeric.nunique(dropna=True) <= 1,
                is_binary=numeric.nunique(dropna=True) <= 2,
                is_bool_like=False,
                is_high_cardinality=False,
                is_integer_like=bool(
                    numeric.dropna().map(float.is_integer).all() and numeric.notna().any()
                ),
            )
        )
    return frame, converted, metas


def drop_rows_without_customer(df: pd.DataFrame, customer_col: str) -> tuple[pd.DataFrame, int]:
    """Remove rows that cannot be attributed to a customer.

    Anonymous transactions cannot be aggregated or scored, so they are
    excluded - and the count is reported to the user.
    """
    if customer_col not in df.columns:
        raise KeyError(f"Customer column '{customer_col}' is not present in the dataset.")
    series = df[customer_col]
    mask = series.notna()
    if pd.api.types.is_numeric_dtype(series):
        mask &= series.astype("float64") > 0
    mask &= series.astype(str).str.strip().str.lower() != ""
    mask &= series.astype(str).str.strip().str.lower() != "nan"
    kept = int(mask.sum())
    return df.loc[mask].copy(), len(df) - kept


# ==========================================================================
# Churn targets
# ==========================================================================
@dataclass
class TargetBundle:
    """The churn label plus everything the UI needs to describe it honestly."""

    y: pd.Series
    column: str
    origin: str  # 'observed' | 'synthetic'
    description: str
    mapping: dict[str, int] | None = None
    excluded_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rule: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return self.origin == "synthetic"

    @property
    def churn_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0


def build_observed_target(df: pd.DataFrame, column: str) -> TargetBundle:
    """Use a real churn/retention column exactly as it appears in the data."""
    if column not in df.columns:
        raise KeyError(f"Target column '{column}' is not present in the dataset.")
    y, mapping = encode_binary_target(df[column])
    y = y.dropna()
    if y.nunique() < 2:
        raise ValueError(
            f"Target column '{column}' contains only one class after cleaning. "
            "Churn prediction needs both a churned and a retained group."
        )
    return TargetBundle(
        y=y,
        column=column,
        origin="observed",
        description=(
            f"Observed churn label taken directly from the column '{column}' "
            f"(mapped to {mapping}). The labels are used as supplied and are never modified."
        ),
        mapping=mapping,
    )


def build_synthetic_inactivity_target(
    customer_df: pd.DataFrame,
    *,
    customer_col: str | None,
    recency_col: str,
    inactivity_days: int,
    snapshot: pd.Timestamp | None = None,
    frequency_col: str | None = None,
    require_low_frequency: bool = False,
) -> TargetBundle:
    """Derive a churn label from an inactivity rule.

    This is explicitly a *synthetic* target: it encodes a business assumption,
    not an observed outcome. The feature used to build the label is returned in
    ``excluded_features`` so it can never be used as a predictor.
    """
    if recency_col not in customer_df.columns:
        raise KeyError(f"Recency column '{recency_col}' is not present in the customer table.")

    recency = pd.to_numeric(customer_df[recency_col], errors="coerce")
    churn = recency > float(inactivity_days)

    rule_description = f"recency > {inactivity_days} days"
    if require_low_frequency and frequency_col and frequency_col in customer_df.columns:
        median_freq = float(pd.to_numeric(customer_df[frequency_col], errors="coerce").median())
        churn = churn & (
            pd.to_numeric(customer_df[frequency_col], errors="coerce") <= median_freq
        )
        rule_description += f" AND frequency <= {median_freq:.0f} (median)"

    warnings = [
        "This churn target is derived from an inactivity rule rather than observed churn "
        "outcomes. It should be treated as a synthetic target.",
    ]
    if snapshot is not None:
        warnings.append(
            f"Recency is measured against the dataset snapshot date {pd.Timestamp(snapshot).date()}, "
            "not the present day."
        )

    index = customer_df.index
    y = pd.Series(churn.astype("float64"), index=index, name="Synthetic_Churn")

    return TargetBundle(
        y=y,
        column="Synthetic_Churn",
        origin="synthetic",
        description=(
            f"Synthetic churn label: a customer is labelled as churned when "
            f"{rule_description}. Snapshot date: "
            f"{pd.Timestamp(snapshot).date() if snapshot is not None else 'n/a'}."
        ),
        mapping=None,
        excluded_features=[recency_col],
        warnings=warnings,
        rule={
            "type": "inactivity",
            "inactivity_days": int(inactivity_days),
            "recency_column": recency_col,
            "frequency_column": frequency_col if require_low_frequency else None,
            "require_low_frequency": bool(require_low_frequency),
            "snapshot": str(snapshot) if snapshot is not None else None,
        },
    )


def build_synthetic_feature_target(
    customer_df: pd.DataFrame,
    *,
    feature_col: str,
    direction: str = "greater",
    threshold: float | None = None,
    quantile: float = 0.5,
) -> TargetBundle:
    """Derive a churn label from an existing customer-level column.

    Used for customer-level uploads that have no churn label at all but do
    contain a recency-like column. The defining column is always excluded
    from the feature set.
    """
    if feature_col not in customer_df.columns:
        raise KeyError(f"Column '{feature_col}' is not present in the customer table.")
    values = pd.to_numeric(customer_df[feature_col], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        raise ValueError(f"Column '{feature_col}' has no numeric values to derive a label from.")

    if threshold is None:
        cut = float(valid.quantile(quantile))
        rule_description = f"{feature_col} {'>' if direction == 'greater' else '<'} {cut:.4g} (the {quantile:.0%} quantile)"
    else:
        cut = float(threshold)
        rule_description = f"{feature_col} {'>' if direction == 'greater' else '<'} {cut:.4g}"

    churn = values > cut if direction == "greater" else values < cut
    y = pd.Series(churn.astype("float64"), index=customer_df.index, name="Synthetic_Churn")

    return TargetBundle(
        y=y,
        column="Synthetic_Churn",
        origin="synthetic",
        description=f"Synthetic churn label: a customer is labelled as churned when {rule_description}.",
        mapping=None,
        excluded_features=[feature_col],
        warnings=[
            "This churn target is derived from a rule applied to an existing column rather than "
            "from observed churn outcomes. It should be treated as a synthetic target.",
        ],
        rule={
            "type": "feature_rule",
            "feature_column": feature_col,
            "direction": direction,
            "threshold": cut,
            "quantile": quantile,
        },
    )


def suggest_inactivity_thresholds(recency: pd.Series) -> list[int]:
    """Data-driven inactivity cut-offs, always including the fixed defaults."""
    valid = pd.to_numeric(recency, errors="coerce").dropna()
    suggestions = set(C.SYNTHETIC_INACTIVITY_CHOICES)
    if not valid.empty:
        for q in (0.5, 0.6, 0.7, 0.8):
            value = float(valid.quantile(q))
            if 1 <= value <= 730:
                suggestions.add(int(round(value)))
    return sorted(int(s) for s in suggestions)
