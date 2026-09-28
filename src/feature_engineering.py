"""Customer feature engineering, feature selection and preprocessing.

Feature generation is *conditional*: a feature is produced only when the
underlying column actually exists and holds usable data. No placeholder or
default values are ever invented for information the dataset does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from . import constants as C
from .data_detection import (
    Profile,
    TransactionSchema,
    _has_any_form,
    _pick_date,
    has_token,
    name_tokens,
    normalise_name,
)

_LARGE_CUTOFF = 1_000_000


# ==========================================================================
# Feature registry
# ==========================================================================
@dataclass
class FeatureSpec:
    """One engineered feature and the evidence that justifies it."""

    name: str
    group: str  # rfm | behaviour | value | engagement | demographic | categorical | derived
    description: str
    sources: list[str]


@dataclass
class AggregationResult:
    """Customer-level frame produced from a transaction-level dataset."""

    customers: pd.DataFrame
    customer_id: str | None
    snapshot: pd.Timestamp | None
    specs: list[FeatureSpec]
    notes: list[str] = field(default_factory=list)
    rows_used: int = 0
    rows_skipped: int = 0

    def specs_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Feature": s.name,
                    "Group": s.group,
                    "Definition": s.description,
                    "Built from": ", ".join(s.sources),
                }
                for s in self.specs
            ]
        )


# ==========================================================================
# Transaction -> customer aggregation
# ==========================================================================
def aggregate_transactions(
    df: pd.DataFrame, schema: TransactionSchema, *, profile: Profile | None = None
) -> AggregationResult:
    """Roll a transaction log up to one row per customer.

    RFM plus additional behavioural features are produced only when the
    matching source column is present, so the same code works for a sales
    log, a clickstream, a support history or a subscription table.
    """
    notes: list[str] = []
    specs: list[FeatureSpec] = []
    rows_in = len(df)

    if not schema.customer_id:
        raise ValueError(
            "No customer identifier was detected. Choose the column that identifies a customer "
            "on the Feature Engineering page, or upload a dataset that contains one."
        )
    if not schema.date:
        raise ValueError(
            "No usable date column was detected, so purchase recency cannot be computed. "
            "Select a date/time column manually, or upload data that contains one."
        )

    work = df
    cust_col = schema.customer_id
    date_col = schema.date

    # -- clean the keys -------------------------------------------------
    work = work.assign(**{cust_col: work[cust_col].astype("string").str.strip()})
    invalid = work[cust_col].isna() | (work[cust_col] == "") | (work[cust_col] == "nan")
    n_invalid = int(invalid.sum())
    if n_invalid:
        notes.append(
            f"Dropped {n_invalid:,} transaction line(s) with a missing or blank '{cust_col}' - "
            "they cannot be attributed to a customer."
        )
        work = work.loc[~invalid]

    dates = pd.to_datetime(work[date_col], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        notes.append(
            f"Dropped {bad_dates:,} transaction line(s) whose '{date_col}' could not be parsed "
            "as a date."
        )
    work = work.assign(**{f"__date__{date_col}": dates})
    work = work.loc[work[f"__date__{date_col}"].notna()]

    if work.empty:
        raise ValueError(
            "No usable transaction lines remain after removing rows with a missing customer id "
            "or an unparseable date. Check the column selections on the Feature Engineering page."
        )

    if profile is not None:
        valid_customers = profile.meta(cust_col).n_unique
    else:
        valid_customers = int(work[cust_col].nunique())

    if valid_customers < 3:
        raise ValueError(
            f"Only {valid_customers} distinct customer(s) were found. At least 3 customers are "
            "required to build behavioural features."
        )

    snapshot = work[f"__date__{date_col}"].max() + pd.Timedelta(days=1)
    notes.append(
        f"Recency is measured from a snapshot date of {snapshot.date()}, which is one day after "
        f"the most recent transaction in the file ({work[f'__date__{date_col}'].max().date()})."
    )

    # -- prepare the line-level working columns ------------------------
    qty_col = schema.quantity if schema.quantity in work.columns else None
    value_col = schema.unit_value if schema.unit_value in work.columns else None
    txn_col = schema.transaction_id if schema.transaction_id in work.columns else None
    product_col = schema.product if schema.product in work.columns else None
    category_col = schema.category if schema.category in work.columns else None

    if qty_col:
        work = work.assign(**{"__qty": pd.to_numeric(work[qty_col], errors="coerce")})
    if value_col:
        work = work.assign(**{"__price": pd.to_numeric(work[value_col], errors="coerce")})

    if qty_col and value_col:
        work = work.assign(**{"__line_value": work["__qty"] * work["__price"]})
        notes.append(
            f"Line value was derived as '{qty_col}' x '{value_col}'. Negative quantities are "
            "treated as returns and are netted off against purchases."
        )
    elif value_col:
        work = work.assign(**{"__line_value": work["__price"]})
        notes.append(
            f"No quantity column was detected, so line value uses '{value_col}' directly. "
            "Quantity-based features were skipped."
        )
    elif qty_col:
        work = work.assign(**{"__line_value": work["__qty"]})
        notes.append(
            f"No price/amount column was detected, so line value uses '{qty_col}' as a proxy. "
            "Monetary features are therefore counts, not currency amounts."
        )
        notes.append("Price-based features were skipped.")
    else:
        work = work.assign(**{"__line_value": np.nan})
        notes.append(
            "Neither a quantity nor a price column was detected, so monetary and quantity "
            "features were skipped. Only recency, frequency and tenure were built."
        )

    n_returns = 0
    if qty_col:
        returns = work["__qty"] < 0
        n_returns = int(returns.sum())
        if n_returns:
            notes.append(
                f"{n_returns:,} line(s) have a negative '{qty_col}' and were treated as returns."
            )

    grouped = work.groupby(cust_col, observed=True)
    date_series = f"__date__{date_col}"

    # -- recency / frequency / monetary --------------------------------
    last_purchase = grouped[date_series].max()
    first_purchase = grouped[date_series].min()
    features: dict[str, pd.Series] = {}
    features["recency_days"] = (snapshot - last_purchase).dt.total_seconds() / 86400.0
    features["frequency"] = (
        grouped[txn_col].nunique() if txn_col else grouped[date_series].nunique()
    )
    features["tenure_days"] = (snapshot - first_purchase).dt.total_seconds() / 86400.0

    specs.append(
        FeatureSpec(
            "recency_days",
            "rfm",
            "Days between the snapshot date and the customer's most recent transaction.",
            [date_col],
        )
    )
    specs.append(
        FeatureSpec(
            "frequency",
            "rfm",
            f"Number of distinct {'invoices' if txn_col else 'purchase dates'} per customer.",
            [txn_col or date_col],
        )
    )
    specs.append(
        FeatureSpec(
            "tenure_days",
            "rfm",
            "Days between the snapshot date and the customer's first observed transaction.",
            [date_col],
        )
    )

    # -- period counts (seasonality) -----------------------------------
    work = work.assign(
        **{
            "__year": work[date_series].dt.year,
            "__quarter": work[date_series].dt.to_period("Q").astype("string"),
            "__month": work[date_series].dt.to_period("M").astype("string"),
        }
    )
    grouped = work.groupby(cust_col, observed=True)
    features["active_quarters"] = grouped["__quarter"].nunique()
    features["active_months"] = grouped["__month"].nunique()
    specs.append(FeatureSpec("active_quarters", "behaviour", "Distinct calendar quarters in which the customer transacted.", [date_col]))
    specs.append(FeatureSpec("active_months", "behaviour", "Distinct calendar months in which the customer transacted.", [date_col]))

    # -- monotonic aggregates ------------------------------------------
    if "recency_days" in features:
        work = work.assign(
            **{
                "__recency": (snapshot - work[date_series]).dt.total_seconds() / 86400.0,
                "__cust_recency": work[cust_col].map(features["recency_days"]),
            }
        )
        recent = work.loc[work["__recency"] <= 365.0]
        grouped_recent = recent.groupby(cust_col, observed=True)
        features["recency_weighted_frequency"] = (
            grouped_recent[date_series].size() / (features["recency_days"] / 365.0).replace(0, np.nan)
        )
        # NOTE: a "days since first purchase" column must not be added here. The
        # only per-customer date gap available in this frame is the customer's
        # own recency, so any such column would be a byte-for-byte copy of
        # ``recency_days`` - which is exactly the synthetic churn label when
        # churn is defined as an inactivity period. ``tenure_days`` already
        # carries the genuine "first to snapshot" span.
        # window functions need a sorted time index within each customer
        work = work.sort_values([cust_col, date_series], kind="mergesort")
        gap = (
            work.groupby(cust_col, observed=True)[date_series]
            .diff()
            .dt.total_seconds()
            .div(86400.0)
        )
        features["avg_days_between_purchases"] = gap.groupby(work[cust_col], observed=True).mean()
        features["std_days_between_purchases"] = gap.groupby(work[cust_col], observed=True).std()
        features["purchase_rate_per_30d"] = (
            features["frequency"] / features["tenure_days"].replace(0, np.nan) * 30.0
        )

    specs.append(
        FeatureSpec(
            "recency_weighted_frequency",
            "rfm",
            "Transactions in the last 365 days divided by (recency / 365) - how regular recent activity is.",
            [date_col],
        )
    )
    specs.append(
        FeatureSpec("avg_days_between_purchases", "behaviour", "Mean gap in days between consecutive transactions.", [date_col])
    )
    specs.append(
        FeatureSpec("std_days_between_purchases", "behaviour", "Variability of the gap between consecutive transactions.", [date_col])
    )
    specs.append(
        FeatureSpec(
            "purchase_rate_per_30d",
            "behaviour",
            "Frequency normalised by tenure, expressed per 30 days.",
            [date_col],
        )
    )

    if "recency_days" in features:
        tenure_years = features["tenure_days"] / 365.25
        features["customer_lifetime_years"] = tenure_years
        features["frequency_per_year"] = features["frequency"] / tenure_years.replace(0, np.nan)
        specs.append(
            FeatureSpec("customer_lifetime_years", "behaviour", "Tenure expressed in years.", [date_col])
        )
        specs.append(
            FeatureSpec(
                "frequency_per_year",
                "behaviour",
                "Transactions per year of observed tenure.",
                [date_col],
            )
        )

    # -- quantity based -------------------------------------------------
    if qty_col:
        positive = work["__qty"] > 0
        pos = work.loc[positive]
        neg = work.loc[~positive]
        grouped_pos = pos.groupby(cust_col, observed=True)
        features["total_quantity"] = grouped_pos["__qty"].sum()
        features["avg_quantity_per_line"] = grouped_pos["__qty"].mean()
        features["max_line_quantity"] = grouped_pos["__qty"].max()
        if not neg.empty:
            features["returned_quantity"] = -neg.groupby(cust_col, observed=True)["__qty"].sum().clip(upper=0)
        specs.append(FeatureSpec("total_quantity", "value", "Sum of positive (non-return) units purchased.", [qty_col]))
        specs.append(FeatureSpec("avg_quantity_per_line", "behaviour", "Mean units per transaction line.", [qty_col]))
        specs.append(FeatureSpec("max_line_quantity", "behaviour", "Largest single-line unit count.", [qty_col]))
        if "returned_quantity" in features:
            specs.append(
                FeatureSpec("returned_quantity", "value", "Total units returned (absolute value).", [qty_col])
            )

    # -- money based ----------------------------------------------------
    if "recency_days" in features and work["__line_value"].notna().any():
        grouped_value = work.groupby(cust_col, observed=True)
        features["monetary_value"] = grouped_value["__line_value"].sum()
        features["avg_line_value"] = grouped_value["__line_value"].mean()
        features["max_line_value"] = grouped_value["__line_value"].max()
        specs.append(FeatureSpec("monetary_value", "value", "Net line value across all transactions (purchases minus returns).", [qty_col, value_col] if qty_col and value_col else [value_col or qty_col or "-"]))
        specs.append(FeatureSpec("avg_line_value", "value", "Mean value of a single transaction line.", [qty_col, value_col] if qty_col and value_col else [value_col or qty_col or "-"]))
        specs.append(FeatureSpec("max_line_value", "value", "Largest single-line value.", [qty_col, value_col] if qty_col and value_col else [value_col or qty_col or "-"]))

        if value_col:
            pos_value = work.loc[(work.get("__qty", 1) > 0)] if qty_col else work
            pos_value = pos_value.loc[pos_value["__price"] > 0]
            if not pos_value.empty:
                features["avg_unit_price"] = pos_value.groupby(cust_col, observed=True)["__price"].mean()
                features["avg_purchase_quantity"] = (
                    pos_value.groupby(cust_col, observed=True)["__qty"].mean() if qty_col else np.nan
                )
                specs.append(FeatureSpec("avg_unit_price", "value", "Mean unit price paid on positive-priced purchase lines.", [value_col]))
                if "avg_purchase_quantity" in features:
                    specs.append(FeatureSpec("avg_purchase_quantity", "behaviour", "Mean units per positive-priced purchase line.", [qty_col]))

        if txn_col:
            invoices = work.groupby([cust_col, txn_col], observed=True)["__line_value"].sum().reset_index()
            per_invoice = invoices.groupby(cust_col, observed=True)["__line_value"]
            features["avg_transaction_value"] = per_invoice.mean()
            features["max_transaction_value"] = per_invoice.max()
            features["std_transaction_value"] = per_invoice.std()
            line_counts = work.groupby([cust_col, txn_col], observed=True).size().reset_index(name="__lines")
            features["avg_items_per_transaction"] = line_counts.groupby(cust_col, observed=True)["__lines"].mean()
            specs.append(FeatureSpec("avg_transaction_value", "value", "Mean total value of one transaction/invoice.", [txn_col, qty_col, value_col]))
            specs.append(FeatureSpec("max_transaction_value", "value", "Highest total value of a single transaction.", [txn_col, qty_col, value_col]))
            specs.append(FeatureSpec("std_transaction_value", "value", "Dispersion of transaction values.", [txn_col, qty_col, value_col]))
            specs.append(FeatureSpec("avg_items_per_transaction", "behaviour", "Mean number of line items per transaction.", [txn_col]))
            features["avg_basket_value"] = features["monetary_value"] / features["frequency"].replace(0, np.nan)
            specs.append(FeatureSpec("avg_basket_value", "value", "Monetary value divided by number of transactions.", [txn_col]))
        else:
            notes.append(
                f"No transaction identifier was detected, so per-transaction averages "
                f"(average transaction value, items per transaction) were skipped."
            )

    # -- product / category dimensions ---------------------------------
    if product_col:
        features["unique_products"] = work.groupby(cust_col, observed=True)[product_col].nunique()
        grouped_product = work.groupby([cust_col, product_col], observed=True).size().rename("__n").reset_index()
        top = grouped_product.sort_values("__n", ascending=False).drop_duplicates(cust_col)
        features["top_product"] = top.set_index(cust_col)[product_col]
        specs.append(FeatureSpec("unique_products", "behaviour", "Distinct items/products purchased.", [product_col]))
        specs.append(FeatureSpec("top_product", "categorical", "Most frequently purchased item (ties broken alphabetically).", [product_col]))

    if category_col and work[category_col].nunique(dropna=True) <= 200:
        grouped_cat = work.groupby(cust_col, observed=True)[category_col]
        features["category_count"] = grouped_cat.nunique()
        dominant = grouped_cat.agg(lambda s: s.value_counts().index[0] if not s.dropna().empty else np.nan)
        features["dominant_category"] = dominant
        share = (
            work.assign(**{"__cat": work[category_col].astype("string")})
            .groupby([cust_col, "__cat"], observed=True)
            .size()
            .rename("__n")
            .reset_index()
        )
        totals = share.groupby(cust_col, observed=True)["__n"].transform("sum")
        share["__share"] = share["__n"] / totals
        dominant_share = share.sort_values(["__n"], ascending=False).drop_duplicates(cust_col)
        features["dominant_category_share"] = dominant_share.set_index(cust_col)["__share"]
        specs.append(FeatureSpec("category_count", "behaviour", "Number of distinct categories/regions purchased from.", [category_col]))
        specs.append(FeatureSpec("dominant_category", "categorical", "Most frequent category/region.", [category_col]))
        specs.append(FeatureSpec("dominant_category_share", "behaviour", "Share of transactions in the dominant category/region.", [category_col]))

    # -- assemble -------------------------------------------------------
    customers = pd.DataFrame(features)
    customers.index.name = cust_col
    if category_col:
        # Keep region codes readable: 12345 -> "12345", not "12345.0".
        customers["dominant_category"] = customers["dominant_category"].map(
            lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else str(v)
        )
    if product_col and "top_product" in customers.columns:
        customers["top_product"] = customers["top_product"].astype("string")

    customers = customers.replace([np.inf, -np.inf], np.nan)
    # Only structural absence (no purchases in the window) becomes a real zero;
    # averages over an empty sample stay missing and are imputed later.
    count_like = [
        "frequency",
        "active_quarters",
        "active_months",
        "total_quantity",
        "unique_products",
        "category_count",
        "returned_quantity",
    ]
    for col in count_like:
        if col in customers.columns:
            customers[col] = customers[col].fillna(0)
    for col in ("monetary_value", "total_quantity", "returned_quantity", "unique_products", "category_count"):
        if col in customers.columns:
            customers[col] = customers[col].fillna(0.0)

    return AggregationResult(
        customers=customers.reset_index(),
        customer_id=cust_col,
        snapshot=snapshot,
        specs=specs,
        notes=notes,
        rows_used=len(work),
        rows_skipped=rows_in - len(work),
    )


# ==========================================================================
# Customer-level features
# ==========================================================================
def engineer_customer_features(
    df: pd.DataFrame, profile: Profile, *, customer_id: str | None = None
) -> tuple[pd.DataFrame, list[FeatureSpec], list[str]]:
    """Prepare an already-customer-level table.

    No aggregation happens. Date columns are converted into elapsed-day
    measures (a tenure or recency reading) and everything else is passed
    through untouched.
    """
    notes: list[str] = []
    specs: list[FeatureSpec] = []
    frame = df.copy()

    for col in list(frame.columns):
        if not pd.api.types.is_datetime64_any_dtype(frame[col]):
            continue
        if profile is not None and col in profile.columns and profile.meta(col).role == "target":
            continue
        derived = f"days_since_{col}"
        reference = frame[col].max() + pd.Timedelta(days=1)
        frame[derived] = (reference - frame[col]).dt.total_seconds() / 86400.0
        specs.append(
            FeatureSpec(
                derived,
                "derived",
                f"Days between the dataset's latest '{col}' and each record (elapsed-time feature).",
                [col],
            )
        )
        notes.append(
            f"Date column '{col}' was converted into '{derived}', measured against "
            f"{reference.date()}."
        )
        frame = frame.drop(columns=[col])

    if customer_id and customer_id in frame.columns:
        frame[customer_id] = frame[customer_id].astype("string").str.strip()
        notes.append(f"Customer identifier kept as '{customer_id}' for reporting only.")

    return frame, specs, notes


# ==========================================================================
# Feature selection
# ==========================================================================
@dataclass
class FeatureSelection:
    """The chosen modelling matrix plus a full audit trail."""

    numeric: list[str]
    categorical: list[str]
    boolean: list[str]
    excluded: list[dict[str, str]]
    specs: list[FeatureSpec] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    customer_id: str | None = None

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.boolean]

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {
                "Feature": name,
                "Type": "numeric" if name in self.numeric else ("boolean" if name in self.boolean else "categorical"),
                "Group": _spec_group(self.specs, name),
            }
            for name in self.all_features
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Feature", "Type", "Group"])

    def excluded_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(list(self.excluded))
        if frame.empty:
            return pd.DataFrame(columns=["Column", "Reason"])
        return frame


def _spec_group(specs: Sequence[FeatureSpec], name: str) -> str:
    for spec in specs:
        if spec.name == name:
            return spec.group
    return ""


def select_features(
    df: pd.DataFrame,
    profile: Profile,
    *,
    customer_id: str | None = None,
    target_column: str | None = None,
    id_columns: Sequence[str] | None = None,
    extra_excluded: Sequence[str] = (),
    extra_reasons: dict[str, str] | None = None,
    max_missing_ratio: float = C.MAX_MISSING_RATIO,
    user_selected: Sequence[str] | None = None,
    specs: Sequence[FeatureSpec] | None = None,
) -> FeatureSelection:
    """Decide which columns may enter the model, recording a reason for each.

    ``user_selected`` is the manual override. When it is not ``None`` the
    user's choice is authoritative for the columns it mentions; columns the
    user removed are still excluded, columns the user added are still
    validated (a target or a customer key can never be added back).
    """
    id_columns = list(id_columns) if id_columns is not None else []
    excluded: list[dict[str, str]] = []
    numeric: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []

    override = None if user_selected is None else set(user_selected)
    never_allowed = set(extra_excluded) | {c for c in (target_column, customer_id) if c}

    for col in df.columns:
        name = str(col)
        meta = profile.columns.get(name)
        if meta is None:
            excluded.append({"Column": name, "Reason": "Column not present in the current frame."})
            continue

        if name in never_allowed:
            if name == target_column:
                reason = "This is the churn target - it must never be used as a predictor."
            elif name == customer_id:
                reason = "Customer identifier - used for reporting only, never as a predictor."
            elif extra_reasons and name in extra_reasons:
                reason = extra_reasons[name]
            else:
                reason = "Excluded by the leakage guard."
            excluded.append({"Column": name, "Reason": reason})
            continue

        if override is not None and name not in override:
            excluded.append({"Column": name, "Reason": "Deselected by the user."})
            continue

        if meta.is_constant:
            excluded.append({"Column": name, "Reason": "Constant column - carries no information."})
            continue

        near_constant = _near_constant_reason(df[col], meta)
        if near_constant:
            excluded.append({"Column": name, "Reason": near_constant})
            continue

        if meta.missing_pct > max_missing_ratio:
            excluded.append(
                {
                    "Column": name,
                    "Reason": f"Too many missing values ({meta.missing_pct:.0%} > {max_missing_ratio:.0%}).",
                }
            )
            continue

        if name in id_columns and meta.id_score >= C.ID_SCORE_THRESHOLD and meta.unique_ratio >= C.NEAR_UNIQUE_RATIO:
            excluded.append(
                {
                    "Column": name,
                    "Reason": (
                        "Identifier-like: almost every value is unique, so one-hot/scaling it "
                        "would only memorise individual records."
                    ),
                }
            )
            continue

        if meta.is_high_cardinality:
            excluded.append(
                {
                    "Column": name,
                    "Reason": (
                        f"High cardinality text ({meta.n_unique:,} distinct values, "
                        f"{meta.unique_ratio:.1%} of rows) - one-hot encoding would be unstable."
                    ),
                }
            )
            continue

        kind = _effective_kind(df[col], meta)
        if kind == "numeric":
            numeric.append(name)
        elif kind == "boolean":
            boolean.append(name)
        else:
            categorical.append(name)

    return FeatureSelection(
        numeric=numeric,
        categorical=categorical,
        boolean=boolean,
        excluded=excluded,
        specs=list(specs or []),
        id_columns=id_columns,
        target_column=target_column,
        customer_id=customer_id,
    )


def _near_constant_reason(series: pd.Series, meta) -> str | None:
    """Detect a column that is *effectively* constant.

    ``meta.is_constant`` only catches a single distinct value. A column where
    99.9% of rows share one value and the remaining 0.1% is a different value is
    numerically almost as useless, yet it survives the plain constant check and
    can dominate a distance-based model such as K-Means.
    """
    counts = series.value_counts(dropna=True, normalize=True)
    if len(counts) < 2 or len(counts) > C.NEAR_CONSTANT_MAX_LEVELS:
        return None
    share = float(counts.iloc[0])
    if share < C.NEAR_CONSTANT_MIN_DOMINANT_SHARE:
        return None
    return (
        f"Near-constant: {share:.1%} of rows share one value "
        f"({len(counts)} distinct value(s) in total) - almost no usable variation."
    )


def _effective_kind(series: pd.Series, meta) -> str:
    """Resolve how a column should be treated, coercing 0/1 integers to boolean."""
    if meta.kind == "boolean":
        return "boolean"
    if meta.kind == "numeric" and meta.is_binary and set(
        pd.unique(pd.to_numeric(series, errors="coerce").dropna())
    ) <= {0, 1}:
        return "boolean"
    if meta.kind == "datetime":
        return "categorical"
    return meta.kind


# ==========================================================================
# Preprocessing
# ==========================================================================
class SkewedLogStandardScaler(BaseEstimator, TransformerMixin):
    """``StandardScaler`` preceded by ``log1p`` on heavily skewed columns.

    Transaction features such as ``total_quantity`` or ``monetary_value`` are
    right-skewed by construction: a handful of wholesale customers can sit 50
    standard deviations from the mean. Fed raw to ``StandardScaler`` and
    ``KMeans``, those few points dominate every distance calculation, so the
    "clusters" become a group of outliers versus everyone else and the
    silhouette score is misleadingly high.

    This transformer keeps the required standardisation but compresses the tail
    of any non-negative, strongly skewed column first. Which columns qualify is
    decided in :meth:`fit`, so the decision is learned from the training
    partition only and never from data the model is later scored on.
    """

    def __init__(self, skew_threshold: float = C.LOG_SKEW_THRESHOLD) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("SkewedLogStandardScaler expects a 2D array of numeric features.")
        self.n_features_in_ = array.shape[1]
        self.log_columns_ = [
            index
            for index in range(array.shape[1])
            if _is_loggable(array[:, index], self.skew_threshold)
        ]
        self.scaler_ = StandardScaler().fit(self._apply_log(array))
        return self

    def _apply_log(self, array: np.ndarray) -> np.ndarray:
        out = np.asarray(array, dtype=float).copy()
        for index in self.log_columns_:
            out[:, index] = np.log1p(out[:, index])
        return out

    def transform(self, X):  # noqa: N803 - sklearn naming
        check_is_fitted(self, "scaler_")
        array = np.asarray(X, dtype=float)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} numeric feature(s) but received "
                f"{array.shape[1]}."
            )
        return self.scaler_.transform(self._apply_log(array))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Pass feature names through so downstream code can label columns."""
        check_is_fitted(self, "scaler_")
        if input_features is None:
            return np.asarray(
                [f"feature_{i}" for i in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)

    def describe(self, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
        """Which columns were log-compressed, for display in the interface."""
        names = list(feature_names or [])
        rows = []
        for index in range(getattr(self, "n_features_in_", 0)):
            name = names[index] if index < len(names) else f"feature_{index}"
            rows.append(
                {
                    "Feature": name,
                    "Treatment": "log1p then standardise"
                    if index in self.log_columns_
                    else "standardise",
                }
            )
        return pd.DataFrame(rows)


def _is_loggable(column: np.ndarray, threshold: float) -> bool:
    """True when a column is non-negative and strongly right-skewed."""
    finite = column[np.isfinite(column)]
    if finite.size < 8 or np.nanmin(finite) < 0:
        return False
    if float(np.max(finite) - np.min(finite)) == 0.0:
        return False
    try:
        skew = float(pd.Series(finite).skew())
    except (TypeError, ValueError):
        return False
    return np.isfinite(skew) and skew >= threshold


def build_preprocessor(selection: FeatureSelection) -> ColumnTransformer:
    """Median-impute + scale numerics, most-frequent-impute + one-hot categoricals."""
    blocks: list[tuple[str, Pipeline, list[str]]] = []

    numeric = [*selection.numeric, *selection.boolean]
    if numeric:
        blocks.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", SkewedLogStandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if selection.categorical:
        blocks.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                min_frequency=1,
                            ),
                        ),
                    ]
                ),
                selection.categorical,
            )
        )
    if not blocks:
        raise ValueError(
            "No usable numerical or categorical features were found. Review the excluded "
            "columns on the Feature Engineering page, or upload a dataset with usable features."
        )
    return ColumnTransformer(
        transformers=blocks,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )


def prepare_matrix(
    frame: pd.DataFrame, selection: FeatureSelection, preprocessor: ColumnTransformer
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit the preprocessor and return the full matrix, the numeric block and names.

    The numeric block is returned separately because K-Means must not be
    distorted by the number of one-hot columns a categorical variable expands
    into. Column order is ``[numeric..., boolean..., one-hot...]``.
    """
    n_numeric = len(selection.numeric) + len(selection.boolean)
    matrix = preprocessor.fit_transform(frame[selection.all_features])
    matrix = _to_dense(matrix)
    return matrix, matrix[:, :n_numeric], list(preprocessor.get_feature_names_out())


def transform_matrix(
    frame: pd.DataFrame, selection: FeatureSelection, preprocessor: ColumnTransformer
) -> np.ndarray:
    """Apply an already-fitted preprocessor to unseen data."""
    return _to_dense(preprocessor.transform(frame[selection.all_features]))


def numeric_block(matrix: np.ndarray, selection: FeatureSelection) -> np.ndarray:
    """Slice the scaled numeric/boolean columns out of a full matrix."""
    return matrix[:, : len(selection.numeric) + len(selection.boolean)]


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype="float64")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
