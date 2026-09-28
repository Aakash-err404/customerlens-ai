"""Dataset-adaptive structure detection.

Nothing here is tied to a particular dataset. Column *names* are treated as
one signal among many and are always combined with datatype, cardinality,
uniqueness ratio and value-distribution evidence. Every decision carries a
human-readable justification so the UI can show *why* something was chosen,
and every decision can be overridden by the user.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import constants as C

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ALPHA_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_SLASH_DATE = re.compile(r"^\s*\d{1,4}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{1,4}")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


# ==========================================================================
# Small lexical helpers
# ==========================================================================
def normalise_name(name: str) -> str:
    """Lower-case a column name and collapse punctuation into single spaces."""
    return _NON_ALNUM.sub(" ", str(name).strip().lower()).strip()


def tokenize(name: str) -> list[str]:
    """Split a column name into lowercase word tokens.

    Handles ``snake_case``, ``kebab-case``, ``camelCase`` and digit/letter
    boundaries, so ``CustomerID``, ``customer_id`` and ``Customer Id`` all
    produce the tokens ``customer`` and ``id``.
    """
    spaced = _CAMEL.sub(" ", str(name))
    spaced = _ALPHA_DIGIT.sub(" ", spaced)
    return [t for t in _TOKEN_SPLIT.split(spaced.lower()) if t]


def name_tokens(name: str) -> set[str]:
    return set(tokenize(name))


def has_token(name: str, vocabulary: Iterable[str]) -> bool:
    return bool(name_tokens(name) & set(vocabulary))


def _joined_token_forms(name: str) -> set[str]:
    """Token set plus the concatenated form, so 'Num_of_Purchases' -> 'numofpurchases'."""
    tokens = tokenize(name)
    forms = set(tokens)
    if len(tokens) > 1:
        forms.add("".join(tokens))
    return forms


def _has_any_form(name: str, vocabulary: Iterable[str]) -> bool:
    return bool(_joined_token_forms(name) & set(vocabulary))


# ==========================================================================
# Reading uploaded files
# ==========================================================================
def read_csv_bytes(data: bytes, *, max_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read CSV bytes with encoding / separator fallbacks.

    Real-world CSVs are frequently not UTF-8 and frequently not comma
    separated, so a small fallback chain is used instead of failing.

    Returns
    -------
    (dataframe, meta) where ``meta`` records the encoding, separator and
    any rows that were truncated by ``max_rows``.
    """
    if not data:
        raise ValueError("The uploaded file is empty.")

    errors: list[str] = []
    for encoding in C.ENCODINGS:
        try:
            text_head = data[:200_000].decode(encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:  # pragma: no cover - depends on file
            errors.append(f"{encoding}: {type(exc).__name__}")
            continue

        separator = _sniff_separator(text_head)
        try:
            frame = pd.read_csv(
                io.BytesIO(data),
                encoding=encoding,
                sep=separator,
                low_memory=False,
                nrows=max_rows,
                on_bad_lines="skip",
            )
        except Exception as exc:  # noqa: BLE001 - parser errors vary by pandas version
            errors.append(f"{encoding}/{separator!r}: {type(exc).__name__}: {exc}")
            continue

        meta = {
            "encoding": encoding,
            "separator": "tab" if separator == "\t" else separator,
            "rows_read": int(len(frame)),
            "truncated": bool(max_rows is not None and len(frame) >= max_rows),
        }
        if errors:
            meta["read_notes"] = errors
        return frame, meta

    raise ValueError(
        "The file could not be parsed as CSV. Tried the following "
        "encodings/separators and failed: " + " | ".join(errors)
    )


def _sniff_separator(text: str) -> str:
    """Pick the most plausible field separator from the first lines of text."""
    header = text.splitlines()[0] if text.splitlines() else ""
    counts = {sep: header.count(sep) for sep in C.CSV_SEPARATORS}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else ","


# ==========================================================================
# Column profiling
# ==========================================================================
@dataclass
class ColumnMeta:
    """Everything the app knows about one column."""

    name: str
    dtype: str
    kind: str  # numeric | categorical | boolean | datetime | text
    n_rows: int
    n_unique: int
    unique_ratio: float
    unique_approximate: bool
    n_missing: int
    missing_pct: float
    n_infinite: int
    is_constant: bool
    is_binary: bool
    is_bool_like: bool
    is_high_cardinality: bool
    is_integer_like: bool
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    sample_values: list[Any] = field(default_factory=list)
    value_domain: list[Any] = field(default_factory=list)
    id_score: int = 0
    id_reasons: list[str] = field(default_factory=list)
    target_score: int = 0
    target_reasons: list[str] = field(default_factory=list)
    role: str = "feature"  # feature | identifier | target | derived | ignore
    role_reason: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "Column": self.name,
            "Dtype": self.dtype,
            "Type": self.kind,
            "Unique values": self.n_unique,
            "Unique ratio": round(self.unique_ratio, 4),
            "Missing": self.n_missing,
            "Missing %": round(self.missing_pct * 100, 2),
            "Infinite": self.n_infinite,
            "Constant": self.is_constant,
            "Binary": self.is_binary,
            "High cardinality": self.is_high_cardinality,
            "ID score": self.id_score,
            "Target score": self.target_score,
            "Role": self.role,
            "Reason": self.role_reason,
        }


@dataclass
class Profile:
    """Profiling result for a whole frame."""

    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    duplicate_pct: float
    total_missing_cells: int
    total_cells: int
    missing_pct: float
    memory_mb: float
    columns: dict[str, ColumnMeta]
    table: pd.DataFrame
    date_detection: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------
    @property
    def numeric(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.kind == "numeric"]

    @property
    def categorical(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.kind in ("categorical", "text")]

    @property
    def booleans(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.kind == "boolean"]

    @property
    def datetimes(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.kind == "datetime"]

    def meta(self, column: str) -> ColumnMeta:
        return self.columns[column]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_pct": self.duplicate_pct,
            "total_missing_cells": self.total_missing_cells,
            "total_cells": self.total_cells,
            "missing_pct": self.missing_pct,
            "memory_mb": self.memory_mb,
        }


def classify_dtype(series: pd.Series) -> str:
    """Map a pandas dtype to one of the semantic kinds used across the app."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique <= max(20, 0.05 * max(len(series), 1)):
            return "categorical"
        return "text"
    return "categorical"


def _safe_nunique(series: pd.Series) -> tuple[int, bool]:
    """Count distinct values, sampling very large columns to stay responsive."""
    n = len(series)
    if n > C.UNIQUE_COUNT_SAMPLE_LIMIT:
        sample = series.sample(C.UNIQUE_COUNT_SAMPLE_SIZE, random_state=C.RANDOM_STATE)
        return int(sample.nunique(dropna=True)), True
    return int(series.nunique(dropna=True)), False


def _looks_like_date_strings(values: pd.Series) -> bool:
    """Cheap lexical gate before attempting an expensive datetime parse."""
    sample = values.dropna().astype(str).head(200)
    if sample.empty:
        return False
    if not sample.str.contains(r"\d", regex=True).any():
        return False
    # Require a separator or an alphabetic month name somewhere in the sample.
    separators = sample.str.contains(r"[/:\-.]|\s", regex=True)
    alpha_month = sample.str.contains(
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", case=False, regex=True
    )
    return bool((separators | alpha_month).any())


def infer_dayfirst(values: pd.Series) -> bool:
    """Decide day-first vs month-first for ambiguous numeric dates.

    Looks at the first two components of ``dd/mm/yy``-shaped strings. If any
    first component exceeds 12 the file must be day-first; if any second
    component exceeds 12 it must be month-first.
    """
    sample = values.dropna().astype(str).head(1000)
    firsts: list[int] = []
    seconds: list[int] = []
    for text in sample:
        m = _SLASH_DATE.match(text)
        if not m:
            continue
        parts = [p for p in re.split(r"[/\-.]", m.group(0).strip()) if p]
        if len(parts) != 3:
            continue
        try:
            firsts.append(int(parts[0]))
            seconds.append(int(parts[1]))
        except ValueError:
            continue
    if not firsts:
        return False
    if max(firsts) > 12:
        return True
    if max(seconds) > 12:
        return False
    return False  # ambiguous -> international/US reading


def infer_datetime_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Convert object columns that really contain dates into datetime columns.

    Only non-numeric columns are considered; numeric columns are never
    reinterpreted as dates because that produces nonsense (and occasional
    silent data corruption).
    """
    frame = df
    report: dict[str, dict[str, Any]] = {}

    candidates = [
        col
        for col in df.columns
        if (df[col].dtype == object or pd.api.types.is_string_dtype(df[col]))
        and not pd.api.types.is_numeric_dtype(df[col])
    ]

    new_columns: dict[str, pd.Series] = {}
    for col in candidates:
        series = df[col]
        n_non_null = int(series.notna().sum())
        if n_non_null < 3 or series.nunique(dropna=True) < 2:
            continue
        if not _looks_like_date_strings(series):
            continue

        sample = series.dropna().astype(str).head(500)
        dayfirst = infer_dayfirst(series)
        try:
            parsed_sample = pd.to_datetime(
                sample, format="mixed", errors="coerce", dayfirst=dayfirst
            )
        except Exception:  # noqa: BLE001 - parser can fail on exotic mixes
            continue
        success = float(parsed_sample.notna().mean())
        if success < 0.85:
            continue

        try:
            parsed = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=dayfirst)
        except Exception:  # noqa: BLE001
            continue
        valid = parsed.dropna()
        if valid.empty:
            continue
        # Reject implausible ranges (e.g. bare years parsed as 1970-01-01).
        if valid.min().year < 1900 or valid.max().year > pd.Timestamp.utcnow().year + 2:
            continue
        if parsed.nunique(dropna=True) < 2:
            continue

        new_columns[col] = parsed
        report[col] = {
            "parse_success_pct": round(success * 100, 2),
            "dayfirst": dayfirst,
            "min": str(valid.min()),
            "max": str(valid.max()),
        }

    if new_columns:
        frame = df.copy()
        for col, parsed in new_columns.items():
            frame[col] = parsed
    return frame, report


def profile_dataset(df: pd.DataFrame) -> Profile:
    """Build a full structural profile of a dataframe."""
    n_rows = int(len(df))
    n_columns = int(len(df.columns))

    duplicated = int(df.duplicated().sum()) if n_rows else 0
    missing_cells = int(df.isna().sum().sum()) if n_rows else 0
    total_cells = max(n_rows * n_columns, 1)

    columns: dict[str, ColumnMeta] = {}
    for col in df.columns:
        series = df[col]
        n_unique, approximate = _safe_nunique(series)
        n_missing = int(series.isna().sum())
        kind = classify_dtype(series)
        non_null = series.dropna()
        unique_ratio = n_unique / max(n_rows, 1)

        n_infinite = 0
        min_value = max_value = mean_value = None
        is_integer_like = False
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = numeric.dropna()
            n_infinite = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
            if not finite.empty:
                min_value = float(finite.min())
                max_value = float(finite.max())
                mean_value = float(finite.mean())
                is_integer_like = bool(np.allclose(finite.to_numpy(), np.round(finite.to_numpy())))

        unique_samples = list(non_null.unique()[:6])
        unique_samples = [
            v.item() if hasattr(v, "item") else v for v in unique_samples
        ]
        domain = list(non_null.unique()[:12])
        domain = [v.item() if hasattr(v, "item") else v for v in domain]

        meta = ColumnMeta(
            name=str(col),
            dtype=str(series.dtype),
            kind=kind,
            n_rows=n_rows,
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            unique_approximate=approximate,
            n_missing=n_missing,
            missing_pct=n_missing / max(n_rows, 1),
            n_infinite=n_infinite,
            is_constant=n_unique <= C.CONSTANT_UNIQUE_MAX,
            is_binary=0 < n_unique <= 2,
            is_bool_like=bool(
                pd.api.types.is_bool_dtype(series)
                or (kind == "categorical" and _is_affirmative_vocabulary(non_null))
            ),
            is_high_cardinality=bool(
                kind in ("categorical", "text")
                and n_unique > C.HIGH_CARD_MIN_UNIQUE
                and unique_ratio > C.HIGH_CARD_MIN_RATIO
            ),
            is_integer_like=is_integer_like,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            sample_values=unique_samples,
            value_domain=domain,
        )
        meta.id_score, meta.id_reasons = _score_identifier(meta)
        meta.target_score, meta.target_reasons = _score_target(meta)
        columns[meta.name] = meta

    profile = Profile(
        n_rows=n_rows,
        n_columns=n_columns,
        n_duplicate_rows=duplicated,
        duplicate_pct=duplicated / max(n_rows, 1),
        total_missing_cells=missing_cells,
        total_cells=total_cells,
        missing_pct=missing_cells / total_cells,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1_048_576),
        columns=columns,
        table=pd.DataFrame([m.as_row() for m in columns.values()]),
    )
    profile.date_detection = {}
    return profile


def _is_affirmative_vocabulary(values: pd.Series) -> bool:
    vocab = C.CHURN_POSITIVE_VALUES | C.CHURN_NEGATIVE_VALUES
    try:
        lowered = {str(v).strip().lower() for v in values.unique()}
    except Exception:  # noqa: BLE001
        return False
    return bool(lowered) and lowered.issubset(vocab) and len(lowered) <= 2


# ==========================================================================
# Identifier detection
# ==========================================================================
def _score_identifier(meta: ColumnMeta) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0

    if meta.kind == "datetime" or meta.is_constant or meta.n_unique == 0:
        return 0, ["Not an identifier: datetime or constant column"]

    # A column with few distinct values that all repeat is a coded attribute
    # (a status, a score band, a flag), not a key.
    if meta.n_unique <= C.MAX_CODED_UNIQUE and meta.unique_ratio < 0.5:
        return 0, [
            f"Not an identifier: only {meta.n_unique} distinct value(s) repeated across rows, "
            "which is a coded attribute rather than a key"
        ]

    if _has_any_form(meta.name, C.ID_NAME_TOKENS):
        score += C.ID_SCORE_NAME
        reasons.append(f"name '{meta.name}' contains an identifier-like token")

    lowered = normalise_name(meta.name)
    # ``normalise_name`` cannot split camel case, so also test a token-joined
    # form: "InvoiceNo" -> "invoice no" must still match the " no" suffix.
    separated = " " + " ".join(tokenize(meta.name))
    if lowered.endswith((" id", " no", " num", " key")) or separated.endswith(
        (" id", " no", " num", " number", " key", " code", " ref", " uuid", " guid")
    ):
        score += C.ID_SCORE_SUFFIX
        reasons.append("name ends with an identifier suffix (id/no/num/key/code/ref)")

    # Keys hold whole numbers, so a continuously distributed measure is never
    # treated as an identifier just because most of its values are unique.
    unique_evidence_allowed = meta.kind != "numeric" or meta.is_integer_like
    if unique_evidence_allowed and meta.unique_ratio >= C.ALMOST_UNIQUE_RATIO and meta.n_unique >= 20:
        score += C.ID_SCORE_ALMOST_UNIQUE
        reasons.append(f"every value is unique ({meta.unique_ratio:.0%} of rows)")
    elif unique_evidence_allowed and meta.unique_ratio >= C.NEAR_UNIQUE_RATIO:
        score += C.ID_SCORE_NEAR_UNIQUE
        reasons.append(f"almost every value is unique ({meta.unique_ratio:.0%} of rows)")

    if meta.kind == "numeric":
        if meta.is_integer_like:
            score += C.ID_SCORE_INT_LIKE
            reasons.append("numeric column contains only whole numbers")
        if meta.is_integer_like and meta.missing_pct > 0:
            score += C.ID_SCORE_FLOAT_ID
            reasons.append("whole numbers stored with missing values - a common id pattern")

    return score, reasons


def detect_id_columns(profile: Profile, *, threshold: int = C.ID_SCORE_THRESHOLD) -> list[str]:
    """Columns that behave like keys rather than features."""
    scored = [
        (meta.name, meta.id_score)
        for meta in profile.columns.values()
        if meta.id_score >= threshold
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in scored]


# ==========================================================================
# Churn target detection
# ==========================================================================
def _score_target(meta: ColumnMeta) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0

    if meta.kind == "datetime" or meta.is_constant or meta.n_unique == 0:
        return 0, ["Not a target: datetime or constant column"]
    if meta.unique_ratio > 0.5 and meta.n_unique > 20:
        return 0, ["Not a target: too many distinct values to be a label"]

    churn_named = _has_any_form(meta.name, C.CHURN_TOKENS)
    generic_named = _has_any_form(meta.name, C.TARGET_GENERIC_TOKENS)
    if churn_named:
        score += C.TARGET_SCORE_CHURN_NAME
        reasons.append(f"name '{meta.name}' contains a churn-related word")
    if generic_named:
        score += C.TARGET_SCORE_GENERIC_NAME
        reasons.append(f"name '{meta.name}' looks like a generic label column")

    vocab_score, vocab_reasons = _score_target_values(meta)
    score += vocab_score
    reasons.extend(vocab_reasons)

    return score, reasons


def _score_target_values(meta: ColumnMeta) -> tuple[int, list[str]]:
    """Score the *value domain* of a column as churn evidence.

    Numeric columns are only ever considered when they are genuinely binary:
    a count column such as ``Num_of_Returns`` can easily contain the values 0
    and 1 by coincidence, so digits alone never count as churn vocabulary.
    """
    if meta.n_unique < 2 or meta.n_unique > 10:
        return 0, []

    domain = [str(v).strip().lower() for v in meta.value_domain]

    if meta.kind == "numeric":
        if not meta.is_binary:
            return 0, []
        try:
            values = {float(v) for v in meta.value_domain}
        except (TypeError, ValueError):
            return 0, []
        if values.issubset({0.0, 1.0}):
            return C.TARGET_SCORE_BINARY, ["numeric 0/1 column - a conventional binary label"]
        return 0, []

    if meta.kind not in ("categorical", "boolean", "text"):
        return 0, []

    positives = sorted({v for v in domain if v in C.CHURN_POSITIVE_VALUES})
    negatives = sorted({v for v in domain if v in C.CHURN_NEGATIVE_VALUES})
    explicit = bool(
        positives
        and negatives
        and any(
            v in C.CHURN_TOKENS or v in {"retained", "inactive", "active", "churned", "churn"}
            for v in domain
        )
    )

    if meta.is_binary:
        if explicit:
            return C.TARGET_SCORE_CHURN_VALUES, [f"values carry churn semantics: {domain}"]
        if positives and negatives:
            return C.TARGET_SCORE_AFFIRMATIVE_VALUES, [f"binary yes/no style values: {domain}"]
        return C.TARGET_SCORE_BINARY, ["binary column with two values"]

    # More than two classes: only an explicit churn word pair counts.
    churn_words = {v for v in domain if v in C.CHURN_TOKENS} or {
        v for v in domain if v in {"retained", "inactive", "active", "churned", "churn"}
    }
    if churn_words:
        return C.TARGET_SCORE_CHURN_VALUES, [f"values contain churn vocabulary: {sorted(churn_words)}"]
    return 0, []


def detect_target_column(
    df: pd.DataFrame, profile: Profile, *, exclude: Sequence[str] = ()
) -> tuple[str | None, list[tuple[str, int, list[str]]], list[str]]:
    """Rank candidate churn targets.

    Returns ``(best_column_or_None, ranked_candidates, all_candidates)``.
    A column is only auto-selected when the evidence is strong: either a
    churn-flavoured name, or churn-flavoured values, or a generic label name
    combined with binary values.
    """
    excluded = set(exclude)
    ranked: list[tuple[str, int, list[str]]] = []
    candidates: list[str] = []

    for name, meta in profile.columns.items():
        if name in excluded or meta.target_score <= 0:
            continue
        candidates.append(name)
        accepted = (
            meta.target_score >= C.TARGET_SCORE_CHURN_NAME
            or meta.target_score >= C.TARGET_SCORE_THRESHOLD
        )
        if accepted:
            ranked.append((name, meta.target_score, meta.target_reasons))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    best = ranked[0][0] if ranked else None
    return best, ranked, candidates


def detect_binary_candidates(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Every binary / boolean column with its value domain - the manual override list."""
    rows = []
    for name, meta in profile.columns.items():
        if meta.kind == "datetime" or meta.n_unique < 2 or meta.n_unique > 3:
            continue
        try:
            values = sorted({str(v) for v in df[name].dropna().unique()[:8]})
        except Exception:  # noqa: BLE001
            values = []
        rows.append(
            {
                "Column": name,
                "Type": meta.kind,
                "Distinct values": meta.n_unique,
                "Values": ", ".join(values),
                "Target score": meta.target_score,
            }
        )
    return pd.DataFrame(rows)


def encode_binary_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Map a binary column onto 1 = churn / 0 = retained.

    The mapping is data driven: values are inspected for churn vocabulary
    first, and otherwise the minority/``True``/``1`` convention is used. The
    mapping is returned so the UI can show exactly what was assumed.
    """
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError("The selected target column contains no values.")

    if pd.api.types.is_bool_dtype(non_null):
        mapping = {False: 0, True: 1}
    else:
        as_text = {str(v).strip().lower(): v for v in non_null.unique()}
        positive = next((k for k in as_text if k in C.CHURN_POSITIVE_VALUES), None)
        negative = next((k for k in as_text if k in C.CHURN_NEGATIVE_VALUES), None)

        if positive is not None and negative is not None:
            mapping = {as_text[negative]: 0, as_text[positive]: 1}
        elif positive is not None:
            other = [v for v in non_null.unique() if str(v).strip().lower() != positive]
            mapping = {other[0]: 0, as_text[positive]: 1}
        else:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numeric.empty and set(numeric.unique()).issubset({0.0, 1.0}):
                mapping = {0.0: 0, 1.0: 1}
            else:
                counts = non_null.value_counts()
                mapping = {counts.index[-1]: 0, counts.index[0]: 1}

    if len(mapping) != 2:
        raise ValueError(
            f"Could not build a two-class mapping for '{series.name}'. "
            f"Observed values: {sorted(map(str, non_null.unique()[:8]))}."
        )
    return series.map(mapping).astype("float64"), {str(k): int(v) for k, v in mapping.items()}


# ==========================================================================
# Transaction schema detection
# ==========================================================================
@dataclass
class TransactionSchema:
    """Resolved column roles for a transaction-level dataset."""

    customer_id: str | None = None
    transaction_id: str | None = None
    date: str | None = None
    quantity: str | None = None
    unit_value: str | None = None
    product: str | None = None
    category: str | None = None
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return bool(self.customer_id and self.date)

    def as_rows(self) -> list[dict[str, str]]:
        return [
            {"Role": "Customer identifier", "Column": self.customer_id or "-", "Note": self.notes.get("customer_id", "")},
            {"Role": "Transaction identifier", "Column": self.transaction_id or "-", "Note": self.notes.get("transaction_id", "")},
            {"Role": "Date / timestamp", "Column": self.date or "-", "Note": self.notes.get("date", "")},
            {"Role": "Quantity", "Column": self.quantity or "-", "Note": self.notes.get("quantity", "")},
            {"Role": "Unit value / price", "Column": self.unit_value or "-", "Note": self.notes.get("unit_value", "")},
            {"Role": "Product / item", "Column": self.product or "-", "Note": self.notes.get("product", "")},
            {"Role": "Category / region", "Column": self.category or "-", "Note": self.notes.get("category", "")},
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "transaction_id": self.transaction_id,
            "date": self.date,
            "quantity": self.quantity,
            "unit_value": self.unit_value,
            "product": self.product,
            "category": self.category,
        }


def _pick_customer_id(profile: Profile, id_cols: Sequence[str]) -> tuple[str | None, str]:
    if not id_cols:
        return None, "No identifier-like column with repeated values was found."

    scored: list[tuple[int, str]] = []
    for col in id_cols:
        meta = profile.meta(col)
        bonus = 0
        if _has_any_form(col, C.CUSTOMER_ENTITY_TOKENS):
            bonus += 100
        if _has_any_form(col, C.TRANSACTION_ENTITY_TOKENS):
            bonus -= 60
        # In transaction data the customer key repeats, so it is *less* unique.
        bonus += int(25 * (1.0 - min(meta.unique_ratio, 1.0)))
        scored.append((bonus, col))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0][1]
    note = (
        f"'{best}' repeats across rows ({(1 - profile.meta(best).unique_ratio):.1%} of rows are "
        f"repeat records) and reads as a customer key."
    )
    return best, note


def _pick_transaction_id(
    profile: Profile, id_cols: Sequence[str], customer_id: str | None, n_rows: int
) -> tuple[str | None, str]:
    others = [c for c in id_cols if c != customer_id]
    if not others:
        return None, "No second identifier was found to group transactions by."

    scored: list[tuple[int, str]] = []
    for col in others:
        meta = profile.meta(col)
        bonus = 0
        if _has_any_form(col, C.TRANSACTION_ENTITY_TOKENS):
            bonus += 100
        if _has_any_form(col, C.CUSTOMER_ENTITY_TOKENS):
            bonus -= 60
        bonus += int(20 * min(meta.unique_ratio * 2, 1.0))
        scored.append((bonus, col))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0][1]
    meta = profile.meta(best)
    note = (
        f"'{best}' has {meta.n_unique:,} distinct values across {n_rows:,} rows "
        f"({meta.n_unique / max(n_rows, 1):.2f} rows per value), consistent with a "
        "transaction / invoice key."
    )
    return best, note


def _pick_date(profile: Profile) -> tuple[str | None, str]:
    dates = profile.datetimes
    if not dates:
        return None, "No column could be parsed as a date."
    for col in dates:
        if _has_any_form(col, C.DATE_TOKENS):
            return col, f"'{col}' is stored as a date and its name is date-related."
    col = dates[0]
    return col, f"'{col}' is the only column parseable as a date."


def _pick_numeric_by_tokens(
    profile: Profile, vocabulary: Iterable[str], *, exclude: Sequence[str] = (), min_unique: int = 5
) -> tuple[str | None, str]:
    tokens = set(vocabulary)
    best: tuple[int, str] | None = None
    for col in profile.numeric:
        if col in exclude:
            continue
        meta = profile.meta(col)
        if meta.n_unique < min_unique or meta.is_constant:
            continue
        if _has_any_form(col, tokens):
            candidate = (100, col)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best:
        return best[1], f"'{best[1]}' is numeric and its name matches a {sorted(tokens)[0]}-style measure."
    return None, ""


def _pick_object_column(
    profile: Profile, vocabulary: Iterable[str], *, min_unique: int = 2, max_unique: int = 5000
) -> tuple[str | None, str]:
    tokens = set(vocabulary)
    best: tuple[int, str] | None = None
    for col in profile.categorical:
        meta = profile.meta(col)
        if not (min_unique <= meta.n_unique <= max_unique):
            continue
        if _has_any_form(col, tokens):
            candidate = (100, col)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best:
        return best[1], f"'{best[1]}' is a text column whose name matches a known dimension."
    return None, ""


def _pick_product(profile: Profile) -> tuple[str | None, str]:
    product, note = _pick_object_column(profile, C.PRODUCT_TOKENS, max_unique=100_000)
    if product:
        return product, note
    # Fall back to the medium-cardinality text column that is not a customer key.
    candidates = [
        (profile.meta(c).n_unique, c)
        for c in profile.categorical
        if 2 <= profile.meta(c).n_unique <= 20_000 and not _has_any_form(c, C.CUSTOMER_ENTITY_TOKENS)
    ]
    if candidates:
        candidates.sort()
        col = candidates[0][1]
        return col, f"'{col}' is the lowest-cardinality text column - treated as the item dimension."
    return None, ""


def detect_transaction_columns(
    df: pd.DataFrame, profile: Profile, id_cols: Sequence[str] | None = None
) -> TransactionSchema:
    """Resolve the role of every column needed for RFM-style aggregation."""
    schema = TransactionSchema()
    ids = list(id_cols) if id_cols is not None else detect_id_columns(profile)

    schema.customer_id, schema.notes["customer_id"] = _pick_customer_id(profile, ids)
    schema.transaction_id, schema.notes["transaction_id"] = _pick_transaction_id(
        profile, ids, schema.customer_id, profile.n_rows
    )
    schema.date, schema.notes["date"] = _pick_date(profile)
    schema.quantity, schema.notes["quantity"] = _pick_numeric_by_tokens(
        profile, C.QUANTITY_TOKENS, exclude=[c for c in (schema.customer_id, schema.transaction_id) if c]
    )
    schema.unit_value, schema.notes["unit_value"] = _pick_numeric_by_tokens(
        profile,
        C.MONETARY_TOKENS,
        exclude=[c for c in (schema.customer_id, schema.transaction_id, schema.quantity) if c],
    )
    schema.product, schema.notes["product"] = _pick_product(profile)
    schema.category, schema.notes["category"] = _pick_object_column(
        profile, C.CATEGORY_TOKENS, max_unique=200
    )
    if not schema.notes.get("product"):
        schema.notes["product"] = ""
    if not schema.notes.get("quantity"):
        schema.notes["quantity"] = "Not detected - quantity-based features will be skipped."
    if not schema.notes.get("unit_value"):
        schema.notes["unit_value"] = "Not detected - value-based features will be skipped."
    if not schema.notes.get("category"):
        schema.notes["category"] = "Not detected - region/category features will be skipped."
    return schema


# ==========================================================================
# Customer-level vs transaction-level
# ==========================================================================
@dataclass
class DatasetVerdict:
    """The dataset-type decision plus the evidence behind it."""

    dataset_type: str  # 'transaction' | 'customer'
    label: str
    confidence: str
    reasons: list[str]
    score: int
    n_entities: int | None = None
    rows_per_entity: float | None = None

    @property
    def is_transaction(self) -> bool:
        return self.dataset_type == "transaction"


def detect_dataset_type(
    df: pd.DataFrame, profile: Profile, schema: TransactionSchema | None = None
) -> DatasetVerdict:
    """Decide whether the data is one-row-per-customer or one-row-per-event."""
    schema = schema or detect_transaction_columns(df, profile)
    reasons: list[str] = []
    score = 0

    customer_id = schema.customer_id
    n_entities: int | None = None
    rows_per_entity: float | None = None

    if customer_id is None:
        reasons.append(
            "No column behaves like a customer key that repeats across rows, so each row is "
            "treated as one customer record."
        )
        return DatasetVerdict(
            dataset_type="customer",
            label="Customer-level dataset",
            confidence="medium",
            reasons=reasons,
            score=0,
        )

    meta = profile.meta(customer_id)
    n_entities = meta.n_unique
    rows_per_entity = profile.n_rows / max(n_entities, 1)

    if meta.unique_ratio < 0.5:
        score += 3
        reasons.append(
            f"'{customer_id}' has {n_entities:,} distinct values across {profile.n_rows:,} rows "
            f"({rows_per_entity:.1f} rows per customer), so rows are events, not customers."
        )
    else:
        reasons.append(
            f"'{customer_id}' is unique on every row ({meta.unique_ratio:.0%}), which matches a "
            "one-row-per-customer table."
        )

    if rows_per_entity and rows_per_entity >= 3:
        score += 1
        reasons.append(f"Each customer appears {rows_per_entity:.1f} times on average.")

    if schema.date:
        score += 1
        reasons.append(f"'{schema.date}' records when each event happened.")

    if schema.transaction_id and n_entities:
        txn_unique = profile.meta(schema.transaction_id).n_unique
        if txn_unique > n_entities * 1.2:
            score += 1
            reasons.append(
                f"'{schema.transaction_id}' has {txn_unique:,} distinct values - more events than "
                f"customers, so rows are transaction lines."
            )

    if schema.quantity or schema.unit_value:
        if meta.unique_ratio < 0.5:
            score += 1
            reasons.append(
                "Line-item measures (quantity and/or price) are present, typical of a sales log."
            )

    if score >= C.TRANSACTION_TYPE_SCORE_THRESHOLD:
        confidence = "high" if score >= 6 else "medium"
        return DatasetVerdict(
            dataset_type="transaction",
            label="Transaction-level dataset",
            confidence=confidence,
            reasons=reasons,
            score=score,
            n_entities=n_entities,
            rows_per_entity=rows_per_entity,
        )

    reasons.append(
        "Weighted evidence did not reach the transaction threshold, so the rows are treated as "
        "individual customer records and no aggregation is performed."
    )
    return DatasetVerdict(
        dataset_type="customer",
        label="Customer-level dataset",
        confidence="high" if score == 0 else "medium",
        reasons=reasons,
        score=score,
        n_entities=n_entities,
        rows_per_entity=rows_per_entity,
    )
