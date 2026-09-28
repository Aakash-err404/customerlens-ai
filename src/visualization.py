"""Plotly figures for every chart in the dashboard.

All builders take already-computed data so that no ML logic is duplicated in
the presentation layer. Every figure is returned as a Plotly ``Figure`` and
rendered by Streamlit.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEGMENT_PALETTE = [
    "#2E5EAA",
    "#E4572E",
    "#17A398",
    "#F2A541",
    "#8E5EA2",
    "#4C9F70",
    "#C94C7C",
    "#7A7A7A",
    "#0F8B8D",
    "#B5651D",
]

RISK_COLOURS = {
    "Low": "#2E7D32",
    "Medium": "#F2A541",
    "High": "#C62828",
    "Lower risk": "#2E7D32",
    "Middle": "#8A94A6",
    "Higher risk": "#C62828",
}

#: Left-to-right order used when stacking risk bands.
RISK_ORDER = ["Low", "Medium", "High", "Lower risk", "Middle", "Higher risk"]


#: Matches the Streamlit theme in ``.streamlit/config.toml`` so figures sit inside
#: the page instead of punching a white rectangle through the dark background.
DARK_BACKGROUND = "#161B22"
DARK_TEXT = "#E6EDF3"
DARK_MUTED = "#9AA4B2"

DARK_FIG = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BACKGROUND,
    plot_bgcolor=DARK_BACKGROUND,
    font=dict(color=DARK_TEXT),
)

PLOT_LAYOUT = dict(
    **DARK_FIG,
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=420,
)


def _empty(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=DARK_MUTED),
    )
    fig.update_layout(
        height=height,
        **DARK_FIG,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _segment_colours(labels: Mapping[int, str] | None) -> list[str]:
    n = max((max(labels) + 1) if labels else 0, 1)
    return list(SEGMENT_PALETTE[:n]) + [
        SEGMENT_PALETTE[i % len(SEGMENT_PALETTE)] for i in range(max(n - len(SEGMENT_PALETTE), 0))
    ]


# ==========================================================================
# Dataset analysis
# ==========================================================================
def missing_values_chart(profile_frame: pd.DataFrame, top: int = 25) -> go.Figure:
    """Missing-value share per column."""
    data = profile_frame[profile_frame["Missing"] > 0].copy()
    if data.empty:
        return _empty("No missing values were found in the uploaded dataset.")
    data = data.nlargest(top, "Missing %")
    fig = px.bar(
        data.sort_values("Missing %"),
        x="Missing %",
        y="Column",
        orientation="h",
        color="Missing %",
        color_continuous_scale="OrRd",
        labels={"Missing %": "Missing %"},
    )
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis=dict(automargin=True))
    return fig


def unique_values_chart(profile_frame: pd.DataFrame, top: int = 25) -> go.Figure:
    """Distinct-value counts, split by detected column type."""
    data = profile_frame.copy().sort_values("Unique values", ascending=False).head(top)
    fig = px.bar(
        data,
        x="Unique values",
        y="Column",
        orientation="h",
        color="Type",
        labels={"Unique values": "Distinct values"},
    )
    fig.update_layout(**PLOT_LAYOUT, yaxis=dict(automargin=True))
    return fig


def target_distribution_chart(
    counts: Mapping[Any, int], *, title: str = "Churn target distribution"
) -> go.Figure:
    """Class balance of the churn target."""
    if not counts:
        return _empty("No target distribution is available yet.")
    frame = pd.DataFrame(
        [{"Class": str(k), "Customers": int(v)} for k, v in counts.items()]
    )
    frame["Share"] = frame["Customers"] / frame["Customers"].sum()
    fig = px.bar(
        frame,
        x="Class",
        y="Customers",
        text=frame["Share"].map(lambda v: f"{v:.1%}"),
        color="Class",
        color_discrete_sequence=SEGMENT_PALETTE,
        labels={"Class": "Class (1 = churn)"},
    )
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    return fig


def numeric_distributions(
    frame: pd.DataFrame, columns: Sequence[str], *, n_cols: int = 3
) -> list[go.Figure]:
    """Histogram grid for the selected numeric features."""
    figures: list[go.Figure] = []
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        if values.nunique() <= 12:
            counts = values.value_counts().sort_index()
            fig = go.Figure(
                go.Bar(x=counts.index.astype(str), y=counts.to_numpy(), marker_color=SEGMENT_PALETTE[0])
            )
            fig.update_layout(title=col, xaxis_title="Value", yaxis_title="Customers")
        else:
            fig = px.histogram(values, nbins=40, color_discrete_sequence=[SEGMENT_PALETTE[0]])
            fig.update_layout(title=col, xaxis_title=col, yaxis_title="Customers")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        figures.append(fig)
    return figures


# ==========================================================================
# Segmentation
# ==========================================================================
def elbow_curve(k_table: pd.DataFrame) -> go.Figure:
    """Inertia against k, with the elbow marked."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_table["k"],
            y=k_table["inertia"],
            mode="lines+markers",
            name="Inertia (within-cluster sum of squares)",
            line=dict(color=SEGMENT_PALETTE[0], width=3),
            marker=dict(size=9),
        )
    )
    fig.add_vline(x=k_table["k"].iloc[-1], line_dash="dash", line_color="#999999", annotation_text="max k")
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Elbow curve - lower inertia means tighter clusters",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia",
    )
    return fig


def silhouette_curve(k_table: pd.DataFrame, recommended_k: int) -> go.Figure:
    """Silhouette score against k."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_table["k"],
            y=k_table["silhouette"],
            mode="lines+markers",
            name="Silhouette score",
            line=dict(color=SEGMENT_PALETTE[1], width=3),
            marker=dict(size=9),
        )
    )
    fig.add_vline(
        x=recommended_k,
        line_dash="dot",
        line_color=SEGMENT_PALETTE[3],
        annotation_text=f"recommended k = {recommended_k}",
        annotation_position="top left",
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Silhouette score by k - higher means better separated clusters",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette score",
    )
    return fig


def segment_distribution_chart(
    sizes: Mapping[int, int], shares: Mapping[int, float], names: Mapping[int, str]
) -> go.Figure:
    """How many customers sit in each segment."""
    frame = pd.DataFrame(
        [
            {
                "Cluster": f"C{c}",
                "Segment": names.get(c, f"Segment {c + 1}"),
                "Customers": int(sizes[c]),
                "Share": float(shares.get(c, 0.0)),
            }
            for c in sorted(sizes)
        ]
    )
    fig = px.bar(
        frame,
        x="Segment",
        y="Customers",
        text=frame["Share"].map(lambda v: f"{v:.1%}"),
        color="Cluster",
        color_discrete_sequence=_segment_colours({c: names.get(c, "") for c in sizes}),
        labels={"Customers": "Customers"},
        hover_data={"Share": ":.1%", "Cluster": True},
    )
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def pca_cluster_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    names: Mapping[int, str],
    explained: Sequence[float],
    *,
    hover: pd.DataFrame | None = None,
) -> go.Figure:
    """2D PCA scatter coloured by cluster."""
    frame = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": [f"C{c}" for c in labels],
        }
    )
    if hover is not None:
        for col in hover.columns:
            frame[col] = hover[col].to_numpy()
    frame["Segment"] = [names.get(int(c), "") for c in labels]
    ev = f" (PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} of variance)" if len(explained) > 1 else ""
    fig = px.scatter(
        frame,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_sequence=_segment_colours(names),
        hover_name="Segment",
        hover_data=[c for c in frame.columns if c not in ("PC1", "PC2", "Cluster", "Segment")],
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        **{**PLOT_LAYOUT, "height": 520},
        title=f"Customer segments in principal-component space{ev}",
    )
    return fig


def cluster_profile_heatmap(z_table: pd.DataFrame, names: Mapping[int, str], *, limit: int = 14) -> go.Figure:
    """Standardised cluster x feature deviation heatmap."""
    if z_table.empty:
        return _empty("No cluster profile is available yet.")
    spread = z_table.abs().max().sort_values(ascending=False).head(limit).index
    data = z_table[list(spread)].copy()
    data.index = [f"C{c} - {names.get(int(c), '')}" for c in data.index]
    fig = go.Figure(
        go.Heatmap(
            z=data.to_numpy(),
            x=[str(c) for c in data.columns],
            y=[str(i) for i in data.index],
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Std. dev."),
            hovertemplate="Cluster: %{y}<br>Feature: %{x}<br>Deviation: %{z:.2f} sd<extra></extra>",
        )
    )
    fig.update_layout(
        **{**PLOT_LAYOUT, "height": max(360, 60 * len(data))},
        title="Cluster profile - standardised deviation from the overall customer average",
        xaxis_title="Feature",
        yaxis_title="",
    )
    return fig


def segment_driver_heatmap(
    z_table: pd.DataFrame,
    names: Mapping[int, str],
    coefficients: pd.DataFrame | None = None,
    *,
    limit: int = 8,
) -> go.Figure:
    """What sets each segment apart, annotated with whether it raises or lowers risk.

    Columns are the features that separate the segments most, so a row reads as a
    portrait of one segment. The arrow on each column label is the effect that
    feature has on predicted churn, which is what links the two halves of the
    report: strong colour plus a raising arrow means a segment worth acting on.
    """
    if z_table.empty:
        return _empty("No segment profile is available yet.")

    spread = list(z_table.abs().max().sort_values(ascending=False).head(limit).index)
    effects: dict[str, float] = {}
    if coefficients is not None and not coefficients.empty:
        effects = dict(zip(coefficients["Feature"], coefficients["Coefficient"]))

    labels: list[str] = []
    effect_words: list[str] = []
    for feature in spread:
        weight = effects.get(feature)
        if weight is None:
            labels.append(f"{feature}")
            effect_words.append("no measured link to churn")
        elif weight > 0:
            labels.append(f"{feature}  (raises risk)")
            effect_words.append("more of this feature goes with more predicted churn")
        else:
            labels.append(f"{feature}  (lowers risk)")
            effect_words.append("more of this feature goes with less predicted churn")

    rows = [names.get(int(cluster), f"Segment {cluster}") for cluster in z_table.index]
    values = z_table[spread].to_numpy()
    custom = [[effect_words[column] for column in range(len(spread))] for _ in rows]

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=rows,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="vs average"),
            customdata=custom,
            hovertemplate=(
                "Segment: %{y}<br>Feature: %{x}<br>Farther from average: %{z:.1f}<br>%{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **{**PLOT_LAYOUT, "height": max(320, 74 * len(rows))},
        xaxis=dict(tickangle=-30),
        yaxis=dict(automargin=True),
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return fig


def cluster_feature_comparison(
    profile_table: pd.DataFrame, feature: str, names: Mapping[int, str]
) -> go.Figure:
    """One feature compared across clusters, in real units."""
    frame = profile_table[["Cluster", feature]].copy()
    frame["Segment"] = [names.get(int(c), "") for c in frame["Cluster"]]
    fig = px.bar(
        frame.sort_values(feature, ascending=False),
        x="Segment",
        y=feature,
        color="Cluster",
        color_discrete_sequence=_segment_colours(names),
        labels={feature: "Cluster mean"},
    )
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def centroid_radar(
    z_table: pd.DataFrame, names: Mapping[int, str], *, limit: int = 8
) -> go.Figure:
    """Radial comparison of cluster centroids on the strongest features."""
    if z_table.empty:
        return _empty("No cluster centroids are available yet.")
    features = list(z_table.abs().max().sort_values(ascending=False).head(limit).index)
    theta = features + [features[0]]
    fig = go.Figure()
    colours = _segment_colours(names)
    for cluster in sorted(z_table.index):
        values = [float(z_table.loc[cluster, f]) for f in features]
        label = f"C{cluster} - {names.get(int(cluster), '')}"
        fig.add_trace(
            go.Scatterpolar(r=values + [values[0]], theta=theta, name=label, line=dict(width=2))
        )
    fig.update_layout(
        **DARK_FIG,
        height=460,
        title="Cluster centroids (standardised)",
        polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=70, b=40),
    )
    return fig


# ==========================================================================
# Churn
# ==========================================================================
def churn_rate_by_segment(
    frame: pd.DataFrame, segment_col: str, target_col: str
) -> go.Figure:
    """Churn rate inside each segment."""
    grouped = frame.groupby(segment_col, observed=True)[target_col].agg(["mean", "count"])
    grouped = grouped.reset_index().sort_values("mean", ascending=False)
    fig = px.bar(
        grouped,
        x=segment_col,
        y="mean",
        text=grouped["mean"].map(lambda v: f"{v:.1%}"),
        color="mean",
        color_continuous_scale="RdYlGn_r",
        labels={"mean": "Churn rate"},
    )
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
    return fig


def segment_prediction_vs_actual(
    frame: pd.DataFrame, segment_col: str, probability_col: str, actual_col: str
) -> go.Figure:
    """Per segment: what the model predicted on average, against what really happened.

    This is the clearest picture of whether the two halves of the report agree.
    The two bars are a direct comparison for the same group of people, and the
    gap between them is annotated, so an over- or under-confident segment is
    visible without reading a single number.
    """
    grouped = (
        frame.groupby(segment_col, observed=True)
        .agg(
            predicted=(probability_col, "mean"),
            actual=(actual_col, "mean"),
            size=(probability_col, "size"),
        )
        .reset_index()
        .sort_values("actual", ascending=True)
    )
    if grouped.empty:
        return _empty("Nothing to compare yet.", height=340)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Model's average prediction",
            y=list(grouped[segment_col]),
            x=grouped["predicted"],
            orientation="h",
            marker_color=SEGMENT_PALETTE[0],
            text=[f"{value:.0%}" for value in grouped["predicted"]],
            textposition="outside",
            customdata=grouped["size"],
            hovertemplate="Model predicted %{x:.0%} on average<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Actually churned",
            y=list(grouped[segment_col]),
            x=grouped["actual"],
            orientation="h",
            marker_color=SEGMENT_PALETTE[1],
            text=[f"{value:.0%}" for value in grouped["actual"]],
            textposition="outside",
            hovertemplate="Actually churned: %{x:.0%}<extra></extra>",
        )
    )

    gap = grouped["predicted"] - grouped["actual"]
    for label, delta in zip(grouped[segment_col], gap):
        if abs(delta) < 0.02:
            continue
        fig.add_annotation(
            x=float(max(grouped.loc[grouped[segment_col] == label, "predicted"].iloc[0],
                        grouped.loc[grouped[segment_col] == label, "actual"].iloc[0])),
            y=label,
            text=f"model {abs(delta):.0%} {'optimistic' if delta > 0 else 'pessimistic'}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(size=12, color=DARK_MUTED),
        )

    fig.update_layout(**PLOT_LAYOUT, barmode="group", bargap=0.3, bargroupgap=0.08, xaxis_tickformat=".0%")
    fig.update_xaxes(title="Average share of the group", range=[0, 1])
    fig.update_yaxes(title="")
    return fig


def risk_mix_by_segment(
    frame: pd.DataFrame, segment_col: str, risk_col: str, actual_col: str | None = None
) -> go.Figure:
    """Predicted risk inside each segment, next to the churn that actually happened.

    This is the chart that ties the two halves of the report together: the bars
    are what the model predicted for each segment, the dot is what was observed
    there, so a segment whose bars and dot disagree is one to investigate.
    """
    present = set(frame[risk_col].dropna())
    available = [band for band in RISK_ORDER if band in present]
    if not available:
        return _empty("No predicted risk levels to show yet.", height=360)

    shares = (
        frame.pivot_table(index=segment_col, columns=risk_col, aggfunc="size", observed=True)
        .reindex(columns=available)
        .fillna(0.0)
    )
    shares = shares.div(shares.sum(axis=1).replace(0, float("nan")), axis=0)
    shares = shares.sort_values(available[-1], ascending=False)

    fig = go.Figure()
    for band in reversed(available):
        values = shares[band].to_numpy()
        counts = (
            frame[frame[risk_col] == band].groupby(segment_col, observed=True).size().reindex(shares.index).fillna(0)
        )
        fig.add_trace(
            go.Bar(
                name=band,
                y=list(shares.index),
                x=values,
                orientation="h",
                marker_color=RISK_COLOURS.get(band, DARK_MUTED),
                text=[f"{value:.0%}" for value in values],
                textposition="inside",
                insidetextanchor="middle",
                customdata=counts.to_numpy(),
                hovertemplate=(
                    f"{band} risk<br>%{{x:.0%}} of this segment<br>%{{customdata:,.0f}} customers<extra></extra>"
                ),
            )
        )

    if actual_col is not None and actual_col in frame.columns:
        observed = frame.groupby(segment_col, observed=True)[actual_col].mean().reindex(shares.index)
        fig.add_trace(
            go.Scatter(
                name="Actually churned",
                y=list(shares.index),
                x=observed.to_numpy(),
                mode="markers",
                marker=dict(
                    color=DARK_TEXT, size=15, symbol="line-ns-open", line=dict(width=3, color=DARK_TEXT)
                ),
                hovertemplate="Actually churned<br>%{x:.0%} of this segment<extra></extra>",
            )
        )

    fig.update_layout(**PLOT_LAYOUT, barmode="stack", bargap=0.34, xaxis_tickformat=".0%")
    fig.update_xaxes(title="Share of the segment", range=[0, 1])
    fig.update_yaxes(title="")
    return fig


def probability_distribution(
    probabilities: np.ndarray, *, target: np.ndarray | None = None, threshold: float = 0.5
) -> go.Figure:
    """Histogram of predicted churn probability."""
    frame = pd.DataFrame({"churn_probability": np.asarray(probabilities, dtype="float64")})
    if target is not None:
        frame["Actual outcome"] = np.where(np.asarray(target) == 1, "Churned", "Retained")
    fig = px.histogram(
        frame,
        x="churn_probability",
        color="Actual outcome" if target is not None else None,
        nbins=45,
        color_discrete_sequence=[SEGMENT_PALETTE[0], SEGMENT_PALETTE[1]],
        marginal="box",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#C62828", annotation_text=f"threshold {threshold:.2f}")
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Predicted churn probability")
    return fig


def risk_distribution(risk_levels: Sequence[str]) -> go.Figure:
    """Counts per risk band."""
    frame = pd.DataFrame({"Risk level": list(risk_levels)})
    counts = frame["Risk level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            marker_color=[RISK_COLOURS[i] for i in counts.index],
            text=[f"{int(v):,}" for v in counts.to_numpy()],
            textposition="outside",
        )
    )
    fig.update_layout(**PLOT_LAYOUT, showlegend=False, yaxis_title="Customers")
    return fig


def confusion_matrix_figure(confusion: np.ndarray, title: str = "Confusion matrix") -> go.Figure:
    """Heatmap of the confusion matrix."""
    labels = ["No churn (0)", "Churn (1)"]
    text = [[str(int(v)) for v in row] for row in confusion]
    total = confusion.sum()
    hover = [
        [f"{text[i][j]} of {total:.0%} of all test customers" for j in range(confusion.shape[1])]
        for i in range(confusion.shape[0])
    ]
    fig = go.Figure(
        go.Heatmap(
            z=confusion,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            customdata=hover,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Customers: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        **DARK_FIG,
        height=400,
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def roc_curve_figure(results: Sequence[Any], *, thresholds: Mapping[str, float] | None = None) -> go.Figure:
    """ROC curves for every evaluated model."""
    fig = go.Figure()
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=result.fpr,
                y=result.tpr,
                mode="lines",
                name=f"{result.label} (AUC {result.roc_auc:.3f})",
                line=dict(width=2.5),
            )
        )
        if thresholds and result.label in thresholds:
            t = thresholds[result.label]
            fig.add_trace(
                go.Scatter(
                    x=[t],
                    y=[None],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=14),
                    name=f"{result.label} at threshold {t:.2f}",
                    showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing", line=dict(dash="dash", color="#999999"))
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="ROC curve - how well the model separates churners from retained customers",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    return fig


def precision_recall_figure(results: Sequence[Any]) -> go.Figure:
    """Precision-recall curves."""
    fig = go.Figure()
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=result.recall_curve,
                y=result.precision_curve,
                mode="lines",
                name=result.label,
                line=dict(width=2.5),
            )
        )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Precision-recall curve",
        xaxis_title="Recall (share of churners correctly found)",
        yaxis_title="Precision (share of flagged customers who really churn)",
    )
    return fig


def threshold_tradeoff_figure(sweep: pd.DataFrame, current: float) -> go.Figure:
    """Precision / recall / F1 as the decision threshold moves."""
    fig = go.Figure()
    for column, colour in (("precision", SEGMENT_PALETTE[0]), ("recall", SEGMENT_PALETTE[1]), ("f1", SEGMENT_PALETTE[2])):
        fig.add_trace(
            go.Scatter(
                x=sweep["threshold"],
                y=sweep[column],
                mode="lines",
                name=column.capitalize(),
                line=dict(width=2.5),
            )
        )
    fig.add_vline(x=current, line_dash="dash", line_color="#C62828", annotation_text=f"current {current:.2f}")
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Decision-threshold trade-off (computed on this data only - never on the test set)",
        xaxis_title="Classification threshold",
        yaxis_title="Score",
    )
    return fig


def feature_coefficient_figure(coefficients: pd.DataFrame, *, limit: int = 18) -> go.Figure:
    """Horizontal bar chart of logistic-regression coefficients."""
    data = coefficients.head(limit).iloc[::-1]
    colours = [SEGMENT_PALETTE[1] if v > 0 else SEGMENT_PALETTE[0] for v in data["Coefficient"]]
    fig = go.Figure(
        go.Bar(
            x=data["Coefficient"],
            y=data["Feature"],
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.3f}" for v in data["Coefficient"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#666666", line_width=1)
    fig.update_layout(
        **{**PLOT_LAYOUT, "height": max(380, 26 * len(data))},
        title="Logistic-regression coefficients (association, not causation)",
        xaxis_title="Coefficient",
        yaxis_title="",
    )
    return fig


def univariate_association_figure(univariate: pd.DataFrame, *, limit: int = 15) -> go.Figure:
    """Univariate AUC per feature - the leakage screen."""
    if univariate.empty:
        return _empty("Univariate association is not available yet.")
    data = univariate.head(limit).iloc[::-1]
    colours = [
        SEGMENT_PALETTE[1] if v >= 0.98 else (SEGMENT_PALETTE[3] if v >= 0.7 else SEGMENT_PALETTE[0])
        for v in data["Univariate AUC"]
    ]
    fig = go.Figure(
        go.Bar(
            x=data["Univariate AUC"],
            y=data["Feature"],
            orientation="h",
            marker_color=colours,
            text=[f"{v:.3f}" for v in data["Univariate AUC"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#999999", annotation_text="no signal")
    fig.add_vline(x=0.98, line_dash="dot", line_color="#C62828", annotation_text="leakage risk")
    fig.update_layout(
        **{**PLOT_LAYOUT, "height": max(380, 26 * len(data))},
        title="Univariate separation of the churn label (0.50 = no signal, 1.00 = perfect separation)",
        xaxis_title="Univariate AUC",
        yaxis_title="",
    )
    return fig


def correlation_heatmap(frame: pd.DataFrame, *, limit: int = 18) -> go.Figure:
    """Correlation matrix of the numeric modelling features."""
    numeric = frame.select_dtypes(include=[np.number])
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return _empty("Not enough numeric features to build a correlation matrix.")
    corr = numeric.corr(numeric_only=True)
    if corr.shape[0] > limit:
        keep = corr.abs().sum().sort_values(ascending=False).head(limit).index
        corr = corr.loc[keep, keep]
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=[str(c) for c in corr.columns],
            y=[str(c) for c in corr.index],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **DARK_FIG,
        height=max(420, 26 * len(corr)),
        title="Feature correlation",
        xaxis=dict(tickangle=-45),
    )
    return fig

