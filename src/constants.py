"""Shared vocabulary and default configuration for CUSTOMERLENS AI.

Nothing in this module references a specific dataset. Every entry is a
*generic* lexical / statistical hint that the detection layer combines with
datatype, cardinality and distribution evidence.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "CUSTOMERLENS AI"
APP_TAGLINE = "Customer Segmentation & Churn Prediction"

# --------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------
ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
CSV_SEPARATORS: tuple[str, ...] = (",", ";", "\t", "|")

# --------------------------------------------------------------------------
# Lexical hints (tokenised on non-alphanumeric boundaries)
# --------------------------------------------------------------------------
CHURN_TOKENS: frozenset[str] = frozenset(
    {
        "churn",
        "churned",
        "attrition",
        "attrite",
        "attrited",
        "lapse",
        "lapsed",
        "lost",
        "cancellation",
        "cancelled",
        "canceled",
        "terminate",
        "terminated",
        "termination",
        "departed",
        "dropout",
        "dropped",
        "winback",
        "reactivation",
    }
)

#: Words that describe a *positive* churn state.
CHURN_POSITIVE_VALUES: frozenset[str] = frozenset(
    {
        "churn",
        "churned",
        "churn yes",
        "yes",
        "y",
        "true",
        "t",
        "1",
        "lost",
        "attrition",
        "attrited",
        "lapsed",
        "cancelled",
        "canceled",
        "terminated",
        "inactive",
        "departed",
        "dropped",
        "dropout",
        "not retained",
        "left",
    }
)

#: Words that describe a *negative* churn state (retained / still active).
CHURN_NEGATIVE_VALUES: frozenset[str] = frozenset(
    {
        "retained",
        "retain",
        "retention",
        "active",
        "no",
        "n",
        "false",
        "f",
        "0",
        "stay",
        "stayed",
        "staying",
        "loyal",
        "retained customer",
        "not churned",
        "win",
        "survived",
        "reactivated",
        "retain customer",
    }
)

#: Generic "this is a label column" markers.
TARGET_GENERIC_TOKENS: frozenset[str] = frozenset(
    {"target", "label", "class", "outcome", "dependent", "prediction"}
)

ID_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "id",
        "ids",
        "identifier",
        "key",
        "code",
        "ref",
        "reference",
        "uuid",
        "guid",
        "hash",
        "customer",
        "cust",
        "client",
        "user",
        "member",
        "membership",
        "account",
        "person",
        "player",
        "subscriber",
        "buyer",
        "shopper",
        "invoice",
        "order",
        "receipt",
        "bill",
        "session",
        "visit",
    }
)

#: Name tokens that strongly indicate a *customer* entity.
CUSTOMER_ENTITY_TOKENS: frozenset[str] = frozenset(
    {"customer", "cust", "client", "user", "member", "buyer", "shopper", "account", "subscriber", "player"}
)

#: Name tokens that strongly indicate a *transaction / order* entity.
TRANSACTION_ENTITY_TOKENS: frozenset[str] = frozenset(
    {"invoice", "order", "transaction", "txn", "receipt", "bill", "purchase", "basket", "cart", "sale"}
)

#: Name tokens for a product / item dimension.
PRODUCT_TOKENS: frozenset[str] = frozenset(
    {"sku", "product", "item", "article", "stock", "stockcode", "goods", "merchandise", "catalog", "itemid", "productid"}
)

#: Name tokens for a low-cardinality categorical dimension.
CATEGORY_TOKENS: frozenset[str] = frozenset(
    {
        "country",
        "region",
        "state",
        "province",
        "city",
        "town",
        "zone",
        "category",
        "channel",
        "brand",
        "store",
        "market",
        "segment",
        "tier",
        "type",
        "group",
        "status",
        "gender",
        "sex",
    }
)

#: Name tokens for a quantity-like measure.
QUANTITY_TOKENS: frozenset[str] = frozenset(
    {"quantity", "qty", "units", "unit", "count", "items", "volume", "qtyordered", "amount"}
)

#: Name tokens for a monetary / price-like measure.
MONETARY_TOKENS: frozenset[str] = frozenset(
    {
        "price",
        "unitprice",
        "amount",
        "cost",
        "value",
        "total",
        "revenue",
        "sales",
        "spend",
        "spent",
        "payment",
        "fee",
        "discount",
        "basket",
        "ordervalue",
        "invoicevalue",
    }
)

#: Name tokens for a date/time column.
DATE_TOKENS: frozenset[str] = frozenset(
    {
        "date",
        "time",
        "timestamp",
        "datetime",
        "day",
        "days",
        "month",
        "year",
        "created",
        "ordered",
        "purchased",
        "purchase",
        "signup",
        "registered",
        "joined",
        "birth",
        "dob",
        "last",
        "next",
        "since",
        "expiry",
        "expires",
    }
)

#: Name tokens marking a *recency-style* feature (higher value == longer since event).
RECENCY_TOKENS: frozenset[str] = frozenset(
    {
        "recency",
        "recy",
        "days",
        "daysago",
        "dayssince",
        "inactive",
        "idle",
        "elapsed",
        "lastseen",
        "lastpurchase",
        "lastorder",
        "lastlogin",
        "lastactivity",
    }
)

#: Name tokens marking a *tenure-style* feature (higher value == older / longer customer).
TENURE_TOKENS: frozenset[str] = frozenset(
    {
        "tenure",
        "lifetime",
        "lifespan",
        "age",
        "years",
        "yearscustomer",
        "months",
        "daysregistered",
        "member",
        "seniority",
        "duration",
    }
)

#: Name tokens marking an *engagement / frequency* feature.
FREQUENCY_TOKENS: frozenset[str] = frozenset(
    {
        "frequency",
        "freq",
        "orders",
        "purchases",
        "transactions",
        "visits",
        "count",
        "numof",
        "numberof",
        "sessions",
        "baskets",
        "items",
        "orderscount",
        "purchasecount",
    }
)

#: Name tokens marking a *monetary value* feature.
VALUE_TOKENS: frozenset[str] = frozenset(
    {
        "monetary",
        "value",
        "spend",
        "spent",
        "revenue",
        "sales",
        "amount",
        "total",
        "ltv",
        "lifetimevalue",
        "balance",
        "basket",
        "profit",
        "gmv",
        "mrr",
        "arr",
    }
)

# --------------------------------------------------------------------------
# Statistical thresholds
# --------------------------------------------------------------------------
#: A column whose name contains one of these is a *substring* match candidate.
ID_SCORE_NAME = 40
ID_SCORE_SUFFIX = 15
ID_SCORE_ALMOST_UNIQUE = 35
ID_SCORE_NEAR_UNIQUE = 25
ID_SCORE_INT_LIKE = 8
ID_SCORE_FLOAT_ID = 10
ID_SCORE_THRESHOLD = 50

TARGET_SCORE_CHURN_NAME = 50
TARGET_SCORE_GENERIC_NAME = 25
TARGET_SCORE_CHURN_VALUES = 45
TARGET_SCORE_AFFIRMATIVE_VALUES = 30
TARGET_SCORE_BINARY = 20
TARGET_SCORE_THRESHOLD = 45

#: unique_ratio at/above which a column is considered an almost-unique key.
ALMOST_UNIQUE_RATIO = 0.98
NEAR_UNIQUE_RATIO = 0.90

#: A column with at most this many distinct values, all of which repeat, is a
#: coded attribute rather than a key. Keys always have many distinct values.
MAX_CODED_UNIQUE = 20

#: A column is "high cardinality text" if it has many distinct values relative
#: to the number of rows. Such columns are excluded from one-hot encoding.
HIGH_CARD_MIN_UNIQUE = 50
HIGH_CARD_MIN_RATIO = 0.05

#: Columns missing more than this fraction of values are excluded by default.
MAX_MISSING_RATIO = 0.50

#: Columns with only this many distinct values (or fewer) are constant-like.
CONSTANT_UNIQUE_MAX = 1

#: A column is treated as near-constant when it has few distinct levels and one
#: of them covers at least this share of the rows. Such a column is numerically
#: almost as useless as a constant one and can dominate distance-based models.
NEAR_CONSTANT_MAX_LEVELS = 3
NEAR_CONSTANT_MIN_DOMINANT_SHARE = 0.99

#: A K-Means solution is rejected as degenerate when its smallest cluster holds
#: less than this share of the customers, because a handful of outliers is not a
#: customer segment.
MIN_CLUSTER_SHARE = 0.02
MIN_CLUSTER_SIZE = 5

#: A feature whose univariate separation from the target reaches this level is
#: treated as a suspected label proxy (a near-deterministic function of the
#: label) and is withheld from the primary model.
LEAKAGE_SUSPICIOUS_AUC = 0.95

#: A non-negative numeric column whose skewness reaches this level is passed
#: through ``log1p`` before standardisation. Transaction measures such as total
#: spend are right-skewed by construction, and without compression a handful of
#: wholesale customers dominate both K-Means distances and the scaler.
LOG_SKEW_THRESHOLD = 1.0

#: Above this many rows, unique-value counting is done on a sample.
UNIQUE_COUNT_SAMPLE_LIMIT = 1_000_000
UNIQUE_COUNT_SAMPLE_SIZE = 400_000

#: Above this many rows, silhouette scoring is done on a subsample.
SILHOUETTE_SAMPLE_LIMIT = 20_000

# --------------------------------------------------------------------------
# Dataset-type decision
# --------------------------------------------------------------------------
TRANSACTION_TYPE_SCORE_THRESHOLD = 4

# --------------------------------------------------------------------------
# Modelling defaults
# --------------------------------------------------------------------------
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 8
DEFAULT_THRESHOLD = 0.50
DEFAULT_RISK_CUTOFFS: tuple[float, float] = (0.35, 0.65)
DEFAULT_CV_FOLDS = 5

#: The aspirational accuracy benchmark. Never used to alter data or evaluation.
ACCURACY_BENCHMARK = 0.85
#: Accuracy at/above this level triggers the leakage self-check warning.
SUSPICIOUS_ACCURACY = 0.95
#: Univariate separation at/above this level flags a suspicious feature.
SUSPICIOUS_UNIVARIATE_AUC = 0.98
#: Train/test accuracy gap at/above this level flags overfitting/leakage.
SUSPICIOUS_ACCURACY_GAP = 0.15

#: Default synthetic churn inactivity thresholds (days).
SYNTHETIC_INACTIVITY_CHOICES: tuple[int, ...] = (30, 60, 90, 120, 180)
DEFAULT_INACTIVITY_DAYS = 90

#: Suggested synthetic thresholds derived from the data itself.
SYNTHETIC_AUTO_BUCKETS = (30, 45, 60, 90, 120, 150, 180, 270, 365)

SEGMENT_FALLBACK_PREFIX = "Segment"

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
#: The application is deliberately two pages wide: results, then the summary.
#: Everything that used to be a separate analytical page now lives inside one of
#: these two, collapsed behind an expander or reduced to a table.
PAGES: tuple[str, ...] = ("Results", "Final Summary")

MAX_PREVIEW_ROWS = 10

#: Folder scanned for sample datasets offered on the Upload page. No filename is
#: ever hard-coded: whatever CSVs are present are discovered at runtime.
EXAMPLE_DATASET_DIR = "datasets"


def discover_examples(root: str | Path = ".") -> dict[str, Path]:
    """Find the sample CSVs shipped alongside the app, keyed by display name."""
    folder = Path(root) / EXAMPLE_DATASET_DIR
    if not folder.is_dir():
        return {}
    return {
        path.stem.replace("_", " ").title(): path
        for path in sorted(folder.glob("*.csv"))
        if path.is_file()
    }
