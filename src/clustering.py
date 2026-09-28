"""K-Means customer segmentation.

Clustering is always performed on standardised numeric customer features so
that large-magnitude measures (e.g. spend) do not dominate distance
calculations. Segment names are derived from the measured cluster statistics -
they are never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import constants as C
from .data_detection import _has_any_form

#: Features whose *higher* value means the customer is more engaged.
_ENGAGEMENT_TOKENS = C.FREQUENCY_TOKENS | C.VALUE_TOKENS
#: Features whose *lower* value means the customer is more at risk.
_RISK_TOKENS = C.RECENCY_TOKENS
#: Features whose *higher* value means a longer relationship.
_TENURE_TOKENS = C.TENURE_TOKENS


# ==========================================================================
# Choosing k
# ==========================================================================
@dataclass
class KSearchResult:
    """Inertia / silhouette for every candidate k."""

    table: pd.DataFrame
    k_range: tuple[int, int]
    recommended_k: int
    elbow_k: int
    best_silhouette_k: int
    reason: str
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return self.table


def find_optimal_k(
    X: np.ndarray,
    *,
    k_min: int = C.DEFAULT_K_MIN,
    k_max: int = C.DEFAULT_K_MAX,
    random_state: int = C.RANDOM_STATE,
    n_init: int = 10,
) -> KSearchResult:
    """Sweep k and recommend a value from inertia + silhouette.

    The recommendation is the k with the best silhouette score; the elbow
    (largest relative drop in inertia) is reported alongside so the user can
    see both signals and override the suggestion.
    """
    n_samples = X.shape[0]
    warnings: list[str] = []
    k_max = int(min(k_max, max(k_min, n_samples - 1)))
    if n_samples < 3:
        raise ValueError(
            f"Only {n_samples} record(s) are available. At least 3 are required to cluster."
        )
    if n_samples < 2 * k_min:
        k_max = max(2, min(k_max, n_samples - 1))
        warnings.append(
            f"The dataset is small, so the maximum k was reduced to {k_max} to leave enough "
            "customers per cluster."
        )

    rows: list[dict[str, float]] = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)
        labels = kmeans.labels_
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) == n_samples:
            silhouette = float("nan")
            warnings.append(f"k={k} produced degenerate clusters; silhouette was not computed.")
        else:
            sample = None
            if n_samples > C.SILHOUETTE_SAMPLE_LIMIT:
                sample = C.SILHOUETTE_SAMPLE_LIMIT
            silhouette = float(
                silhouette_score(X, labels, sample_size=sample, random_state=random_state)
            )
        sizes = np.bincount(labels, minlength=k)
        smallest = int(sizes.min()) if sizes.size else 0
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": silhouette,
                "customers_per_cluster": float(n_samples / k),
                "smallest_cluster": smallest,
                "smallest_share": smallest / max(n_samples, 1),
            }
        )

    table = pd.DataFrame(rows)
    elbow_k = _detect_elbow(table["inertia"].to_numpy())
    valid = table.dropna(subset=["silhouette"])
    # A very high silhouette can be an artefact: a handful of extreme outliers
    # form their own "cluster" and are trivially far from everything else. Such
    # a split is not a segmentation, so degenerate solutions are rejected in
    # favour of the best silhouette among the balanced ones.
    min_size = max(C.MIN_CLUSTER_SIZE, int(round(C.MIN_CLUSTER_SHARE * n_samples)))
    if "smallest_cluster" in valid.columns:
        usable = valid[valid["smallest_cluster"] >= min_size]
    else:
        usable = valid

    if valid.empty:
        recommended_k = elbow_k if elbow_k else k_min
        reason = (
            "Silhouette could not be computed for any k, so the elbow of the inertia curve is "
            "used instead."
        )
    else:
        if usable.empty:
            fallback = valid.loc[valid["silhouette"].idxmax()]
            recommended_k = int(fallback["k"])
            reason = (
                f"Every candidate k produced a cluster with fewer than {min_size} customers "
                f"({int(fallback['smallest_cluster'])} at the best k), so no split is a usable "
                "segmentation on this data."
            )
            warnings.append(reason)
        else:
            best = usable.loc[usable["silhouette"].idxmax()]
            recommended_k = int(best["k"])
            reason = (
                f"k={recommended_k} has the highest silhouette score "
                f"({float(best['silhouette']):.3f}). The elbow of the inertia curve falls at "
                f"k={elbow_k}."
            )
            if elbow_k != recommended_k:
                reason += (
                    " The two criteria disagree, which usually means the data has a strong dominant "
                    "group plus a diffuse tail - inspect the cluster profile before accepting."
                )
            rejected = valid.drop(index=usable.index)
            rejected = rejected[rejected["silhouette"] >= float(best["silhouette"])]
            if not rejected.empty:
                detail = ", ".join(
                    f"k={int(r['k'])} (silhouette {r['silhouette']:.3f}, smallest cluster "
                    f"{int(r['smallest_cluster'])} customers)"
                    for _, r in rejected.sort_values("k").iterrows()
                )
                reason += (
                    f" A higher raw silhouette was rejected for {detail}, because a cluster of a "
                    "few outlier customers is not a segment."
                )

    return KSearchResult(
        table=table,
        k_range=(k_min, k_max),
        recommended_k=recommended_k,
        elbow_k=int(elbow_k),
        best_silhouette_k=recommended_k,
        reason=reason,
        warnings=warnings,
    )


def _detect_elbow(values: np.ndarray) -> int:
    """Knee of a decreasing curve via the maximum distance to the chord."""
    if len(values) < 3:
        return int(np.argmin(values)) + 2
    x = np.arange(len(values), dtype="float64")
    y = np.asarray(values, dtype="float64")
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return int(np.argmin(values)) + 2
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return int(np.argmax(distances)) + 2


# ==========================================================================
# Training
# ==========================================================================
@dataclass
class KMeansModel:
    """A fitted K-Means model plus the data it was fitted on."""

    kmeans: KMeans
    k: int
    n_features: int
    feature_names: list[str]
    inertia: float
    silhouette: float
    n_samples: int
    trained_on: str = "full dataset"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(X)

    @property
    def centroids(self) -> np.ndarray:
        return self.kmeans.cluster_centers_


def train_kmeans(
    X: np.ndarray,
    *,
    k: int,
    feature_names: Sequence[str] | None = None,
    random_state: int = C.RANDOM_STATE,
    n_init: int = 10,
    trained_on: str = "full dataset",
    compute_silhouette: bool = True,
) -> KMeansModel:
    """Fit K-Means on an already-scaled feature matrix."""
    n_samples = X.shape[0]
    if n_samples < k:
        raise ValueError(
            f"Cannot build {k} clusters from {n_samples} record(s). Choose a smaller k or "
            "provide more data."
        )
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)

    silhouette = float("nan")
    if compute_silhouette and len(np.unique(kmeans.labels_)) in (2, k) and k < n_samples:
        sample = C.SILHOUETTE_SAMPLE_LIMIT if n_samples > C.SILHOUETTE_SAMPLE_LIMIT else None
        silhouette = float(
            silhouette_score(X, kmeans.labels_, sample_size=sample, random_state=random_state)
        )

    return KMeansModel(
        kmeans=kmeans,
        k=k,
        n_features=X.shape[1],
        feature_names=list(feature_names or [f"f{i}" for i in range(X.shape[1])]),
        inertia=float(kmeans.inertia_),
        silhouette=silhouette,
        n_samples=n_samples,
        trained_on=trained_on,
    )


# ==========================================================================
# Cluster description
# ==========================================================================
@dataclass
class ClusterProfile:
    """Per-cluster statistics, raw means and standardised deviations."""

    table: pd.DataFrame  # one row per cluster
    z_table: pd.DataFrame  # cluster x feature z-scores
    feature_means: pd.DataFrame
    feature_stds: pd.DataFrame
    sizes: dict[int, int]
    shares: dict[int, float]
    categorical_modes: dict[str, dict[int, str]]


def generate_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    raw_frame: pd.DataFrame | None = None,
    categorical_features: Sequence[str] = (),
) -> ClusterProfile:
    """Describe each cluster: size, share, feature means and z-score deviations."""
    names = list(feature_names)
    data = pd.DataFrame(X, columns=names, index=range(len(X)))
    data["__cluster__"] = np.asarray(labels, dtype=int)

    cluster_ids = sorted(data["__cluster__"].unique())
    means = data.groupby("__cluster__")[names].mean()
    stds = data[names].std(ddof=0).replace(0, np.nan)
    overall_mean = data[names].mean()
    z = means.sub(overall_mean, axis="columns").div(stds, axis="columns")

    sizes = {int(c): int((data["__cluster__"] == c).sum()) for c in cluster_ids}
    total = max(len(data), 1)
    shares = {c: v / total for c, v in sizes.items()}

    modes: dict[str, dict[int, str]] = {}
    if raw_frame is not None:
        raw = raw_frame.reset_index(drop=True).copy()
        raw["__cluster__"] = np.asarray(labels, dtype=int)
        for col in categorical_features:
            if col not in raw.columns:
                continue
            per_cluster: dict[int, str] = {}
            for c in cluster_ids:
                values = raw.loc[raw["__cluster__"] == c, col].dropna()
                if values.empty:
                    per_cluster[int(c)] = "-"
                else:
                    counts = values.astype(str).value_counts()
                    per_cluster[int(c)] = f"{counts.index[0]} ({counts.iloc[0] / len(values):.0%})"
            modes[col] = per_cluster

    rows: list[dict[str, Any]] = []
    for c in cluster_ids:
        row: dict[str, Any] = {
            "Cluster": int(c),
            "Customers": sizes[int(c)],
            "Share": shares[int(c)],
        }
        for name in names:
            row[name] = float(means.loc[c, name])
        rows.append(row)
    table = pd.DataFrame(rows)

    return ClusterProfile(
        table=table,
        z_table=z,
        feature_means=means,
        feature_stds=stds,
        sizes=sizes,
        shares=shares,
        categorical_modes=modes,
    )


def _rank_tiers(series: pd.Series) -> dict[int, str]:
    """Assign tiers by rank *among the clusters* so labels stay distinct.

    Percentile tiers computed against the whole population can easily put every
    cluster in the same bucket - two clusters can both sit below the 30th
    percentile of a heavily skewed feature while differing substantially from
    each other. Ranking the clusters against one another guarantees that the
    strongest and weakest cluster are always told apart, which is exactly the
    distinction a segment name is supposed to convey.
    """
    usable = series.dropna()
    if len(usable) < 2:
        return {int(cluster): "Mid" for cluster in usable.index}
    n = len(usable)
    ordered = usable.sort_values(ascending=False)
    tiers: dict[int, str] = {}
    for rank, cluster in enumerate(ordered.index):
        if n == 2:
            tiers[int(cluster)] = "High" if rank == 0 else "Low"
        elif rank == 0:
            tiers[int(cluster)] = "High"
        elif rank == n - 1:
            tiers[int(cluster)] = "Low"
        else:
            tiers[int(cluster)] = "Mid"
    return tiers


def _tier(value: float, reference: np.ndarray, *, high: tuple[float, float], low: tuple[float, float], mid: str) -> str:
    """Assign a descriptive tier from a percentile of the overall distribution."""
    ref = np.asarray(reference, dtype="float64")
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(value):
        return mid
    pct = float((ref < value).mean())
    if high[0] <= pct <= high[1]:
        return "High"
    if low[0] <= pct <= low[1]:
        return "Low"
    return "Mid"


def _axis_for(z_table: pd.DataFrame, candidates: Sequence[str], *, min_spread: float = 0.35) -> str | None:
    """Pick the most differentiating feature from a candidate group.

    Selection is based on the largest standardised deviation achieved by any
    cluster, so the axis always reflects something the data actually shows.
    """
    best: tuple[float, str] | None = None
    for name in candidates:
        if name not in z_table.columns:
            continue
        column = z_table[name].dropna()
        if column.empty:
            continue
        score = float(column.abs().max())
        if best is None or score > best[0]:
            best = (score, name)
    if best is None or best[0] < min_spread:
        return None
    return best[1]


def generate_segment_names(
    profile: ClusterProfile,
    raw_frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
) -> dict[int, str]:
    """Build human-readable segment names from the measured cluster statistics.

    Each name is assembled from up to three independent, measurable
    characteristics: value, engagement and recency/tenure. If none of those
    can be established from the data, a neutral ``Segment N`` label is used
    instead of inventing a business claim.
    """
    names = list(feature_names)
    raw = raw_frame.reset_index(drop=True)
    z = profile.z_table

    value_candidates = [n for n in names if _has_any_form(n, C.VALUE_TOKENS) or _has_any_form(n, C.FREQUENCY_TOKENS)]
    value_axis = _axis_for(z, value_candidates)
    value_axis_is_monetary = bool(value_axis and _has_any_form(value_axis, C.VALUE_TOKENS))
    if value_axis is None:
        # No monetary or frequency feature exists. Use the single strongest
        # differentiator, but describe it as engagement rather than value,
        # because nothing in the data supports a value claim.
        spread = z.abs().max().sort_values(ascending=False)
        if len(spread) and float(spread.iloc[0]) >= 0.8:
            value_axis = str(spread.index[0])
            value_axis_is_monetary = False
        else:
            value_axis = None

    engagement_axis = _axis_for(
        z,
        [
            n
            for n in names
            if (_has_any_form(n, C.FREQUENCY_TOKENS) or _has_any_form(n, C.VALUE_TOKENS))
            and n != value_axis
        ],
    )
    risk_axis = _axis_for(
        z,
        [
            n
            for n in names
            if (_has_any_form(n, _RISK_TOKENS) or _has_any_form(n, _TENURE_TOKENS))
            and n != value_axis
        ],
    )

    references = {
        n: raw[n].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        for n in names
        if n in raw.columns
    }

    # Rank each chosen axis across the clusters themselves, so a strong and a
    # weak cluster never collapse onto the same descriptive word.
    rank_tiers = {
        axis: _rank_tiers(profile.table.set_index("Cluster")[axis])
        for axis in (value_axis, engagement_axis, risk_axis)
        if axis
    }

    result: dict[int, str] = {}
    for _, row in profile.table.iterrows():
        cluster = int(row["Cluster"])
        parts: list[str] = []

        if value_axis:
            value = float(row[value_axis])
            reference = references.get(value_axis, np.array([value]))
            if value_axis_is_monetary:
                tier = rank_tiers[value_axis].get(cluster) or _tier(
                    value, reference, high=(0.70, 1.01), low=(0.0, 0.30), mid="Mid"
                )
                parts.append(f"{tier}-Value")
            else:
                tier = rank_tiers[value_axis].get(cluster) or _tier(
                    value, reference, high=(0.70, 1.01), low=(0.0, 0.30), mid=""
                )
                parts.append(
                    {"High": "Highly Engaged", "Low": "Infrequent", "Mid": "Moderately Engaged"}.get(
                        tier, "Moderately Engaged"
                    )
                )

        if engagement_axis:
            value = float(row[engagement_axis])
            tier = rank_tiers[engagement_axis].get(cluster) or _tier(
                value,
                references.get(engagement_axis, np.array([value])),
                high=(0.70, 1.01),
                low=(0.0, 0.30),
                mid="",
            )
            if not parts:
                parts.append(
                    {"High": "Highly Engaged", "Low": "Infrequent"}.get(tier, "Moderately Engaged")
                )

        if risk_axis:
            value = float(row[risk_axis])
            reference = references.get(risk_axis, np.array([value]))
            tier = rank_tiers[risk_axis].get(cluster) or _tier(
                value, reference, high=(0.70, 1.01), low=(0.0, 0.30), mid=""
            )
            if _has_any_form(risk_axis, _RISK_TOKENS):
                # A higher recency value means longer since the last activity.
                if tier == "High":
                    parts.append("At-Risk")
                elif tier == "Low":
                    parts.append("Recent")
            else:
                # Tenure-style: a higher value means a longer relationship.
                if tier == "High":
                    parts.append("Long-Standing")
                elif tier == "Low":
                    parts.append("Newly Onboarded")

        label = " ".join(p for p in parts if p).strip()
        result[cluster] = (
            f"{label} Customers" if label else f"{C.SEGMENT_FALLBACK_PREFIX} {cluster + 1}"
        )

    return _make_unique(result, profile=profile, feature_names=names)


def _make_unique(
    labels: dict[int, str], *, profile: ClusterProfile, feature_names: Sequence[str]
) -> dict[int, str]:
    """Disambiguate clusters that received the same descriptive name.

    Rather than appending an arbitrary "variant 2", the cluster's strongest
    measured differentiator is named, so the two labels still tell the reader
    something true about the data.
    """
    seen: dict[str, list[int]] = {}
    for cluster in sorted(labels):
        seen.setdefault(labels[cluster], []).append(cluster)

    out = dict(labels)
    table = profile.z_table
    for label, clusters in seen.items():
        if len(clusters) < 2:
            continue
        for order, cluster in enumerate(clusters, start=2):
            if cluster not in table.index:
                out[cluster] = f"{label} ({order})"
                continue
            z_row = table.loc[cluster].drop(labels={"Cluster"}).abs()
            # Prefer a feature that actually separates this cluster from the
            # ones it shares a name with.
            competitors = [c for c in clusters if c != cluster and c in table.index]
            if competitors:
                gap = (z_row.reindex(table.index) - 0.0).abs()
                gap.loc[competitors] = 0.0
            else:
                gap = z_row
            gap = gap.reindex(feature_names).dropna()
            if gap.empty or float(gap.max()) <= 0:
                out[cluster] = f"{label} ({order})"
                continue
            best_feature = str(gap.idxmax())
            readable = best_feature.replace("_", " ")
            direction = "highest" if float(table.loc[cluster, best_feature]) > 0 else "lowest"
            out[cluster] = f"{label} ({direction} {readable})"
    return out


# ==========================================================================
# 2D projection
# ==========================================================================
def pca_projection(
    X: np.ndarray, *, n_components: int = 2, random_state: int = C.RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Project the clustering space onto two principal components for plotting."""
    n_components = int(min(n_components, X.shape[1], X.shape[0]))
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    if coords.shape[1] < 2:  # pragma: no cover - degenerate input
        coords = np.column_stack([coords, np.zeros(len(coords))])
    labels = [f"PC{i + 1}" for i in range(coords.shape[1])]
    return coords, pca.explained_variance_ratio_, labels


def top_discriminating_features(z_table: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Rank features by how strongly they separate the clusters."""
    if z_table.empty:
        return pd.DataFrame(columns=["Feature", "Max |z|", "Most separated cluster", "Direction"])
    spread = z_table.abs().max().sort_values(ascending=False).head(limit)
    rows = []
    for name in spread.index:
        row = z_table[name].dropna()
        if row.empty:
            continue
        extreme = row.abs().idxmax()
        rows.append(
            {
                "Feature": name,
                "Max |z|": float(row.loc[extreme]),
                "Most separated cluster": int(extreme),
                "Direction": "above average" if row.loc[extreme] > 0 else "below average",
            }
        )
    return pd.DataFrame(rows).sort_values("Max |z|", key=lambda s: s.abs(), ascending=False)
