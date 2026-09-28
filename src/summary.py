"""Composition of the final, plain-language summary from measured results.

The summary is generated from numbers that were actually computed - never from a
template with placeholders. Every claim in the output is traceable to one of the
values collected in :class:`SummaryFacts`, and where a number is not meaningful
(synthetic target, model that does not beat the baseline, leakage suspects) the
summary says so instead of dressing it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: How many segments and drivers the summary spells out. The full tables stay in
#: the app; the prose stays readable.
MAX_SEGMENTS_IN_TEXT = 6
MAX_DRIVERS_IN_TEXT = 6


@dataclass
class SegmentFact:
    """One segment, reduced to the numbers the summary is allowed to state."""

    name: str
    customers: int
    share: float
    churn_rate: float | None = None
    traits: str = ""


@dataclass
class SummaryFacts:
    """Everything the summary may say, gathered by the caller.

    Kept as plain values on purpose: the summary has no dependency on Streamlit
    or on the app's own dataclasses, so it can be unit-tested and reused.
    """

    # -- what was analysed
    source_name: str = ""
    grain_label: str = ""
    rows_in_file: int = 0
    rows_analysed: int = 0
    customer_count: int = 0
    columns_analysed: int = 0

    # -- the churn target
    target_origin: str = "unavailable"
    target_column: str = ""
    target_description: str = ""
    target_rule: str = ""
    churn_rate: float | None = None

    # -- segmentation
    k: int = 0
    silhouette: float = float("nan")
    k_reason: str = ""
    segments: list[SegmentFact] = field(default_factory=list)

    # -- churn model
    model_label: str = ""
    test_accuracy: float | None = None
    test_auc: float | None = None
    baseline_accuracy: float | None = None
    beats_baseline: bool | None = None
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame)

    # -- honesty checks and provenance
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    excluded_columns: int = 0
    excluded_examples: list[str] = field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0
    test_size: float = 0.2
    random_state: int = 0
    threshold: float = 0.5
    suggested_threshold: float = 0.5

    # -- the one switch that changes the shape of the whole document
    can_model: bool = True

    @property
    def synthetic(self) -> bool:
        return self.target_origin == "synthetic"

    @property
    def observed(self) -> bool:
        return self.target_origin == "observed"

    @property
    def base_rate(self) -> float | None:
        return self.churn_rate

    @property
    def segments_above_base(self) -> list[SegmentFact]:
        """Segments that churn more than the customer base as a whole."""
        if self.base_rate is None:
            return []
        return [s for s in self.segments if s.churn_rate is not None and s.churn_rate > self.base_rate]

    @property
    def priority_segment(self) -> SegmentFact | None:
        """The segment with the most customers at above-average churn risk."""
        if not self.segments_above_base:
            return None
        return max(self.segments_above_base, key=lambda s: s.customers)


# ==========================================================================
# Formatting helpers
# ==========================================================================
def pct(value: float | None, places: int = 1) -> str:
    """A percentage, or ``n/a`` when the number is missing or undefined."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{places}%}"


def pp(value: float | None, places: int = 1) -> str:
    """A difference between two percentages, worded for a non-technical reader."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.{places}f} points"


def num(value: float | None, places: int = 3) -> str:
    """A plain ratio such as an AUC, which is never shown as a percentage."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{places}f}"


def count_phrase(total: int) -> str:
    """``1 column`` / ``4 columns``, without the usual parenthetical."""
    return "1 column" if total == 1 else f"{total} columns"


def _headline(facts: SummaryFacts) -> str:
    if not facts.can_model:
        return (
            f"{facts.customer_count:,} customers were profiled and segmented, but no churn model "
            "could be fitted, so this report stops at descriptive analysis."
        )
    if facts.beats_baseline:
        gap = abs(float(facts.test_accuracy) - float(facts.baseline_accuracy))
        return (
            f"The {facts.customer_count:,} customers fall into {facts.k} clear groups, and the "
            f"churn model gets {pct(facts.test_accuracy)} of customers it had never seen before "
            f"right, {gap * 100:.1f} points better than simply guessing that the "
            f"{pct(facts.baseline_accuracy)} majority stay."
        )
    return (
        f"The {facts.k} segments describe the customer base clearly, but the churn model is no "
        f"better at spotting churn than simply guessing that the {pct(facts.baseline_accuracy)} "
        f"majority stay ({pct(facts.test_accuracy)} on customers it had never seen)."
    )


def _driver_lines(facts: SummaryFacts) -> list[str]:
    if facts.drivers.empty:
        return []
    frame = facts.drivers.head(MAX_DRIVERS_IN_TEXT)
    rows = []
    for _, row in frame.iterrows():
        coefficient = float(row["Coefficient"])
        direction = "more likely to churn" if coefficient > 0 else "less likely to churn"
        rows.append(
            f"`{row['Feature']}` - {direction} (strength {abs(coefficient):.3f})"
        )
    return rows


def _segment_bullets(facts: SummaryFacts) -> list[str]:
    lines = []
    for segment in facts.segments[:MAX_SEGMENTS_IN_TEXT]:
        text = (
            f"**{segment.name}** - {segment.customers:,} customers "
            f"({pct(segment.share)} of the base)"
        )
        if segment.churn_rate is not None:
            comparison = ""
            if facts.base_rate is not None:
                delta = float(segment.churn_rate) - float(facts.base_rate)
                if abs(delta) >= 0.01:
                    comparison = f", {pp(delta)} against the {pct(facts.base_rate)} base rate"
            text += f", churn rate {pct(segment.churn_rate)}{comparison}"
        if segment.traits:
            text += f" - {segment.traits}"
        lines.append(f"- {text}")
    return lines


def _action_lines(facts: SummaryFacts) -> list[str]:
    """Segment-level next steps, each one derived from a measured rate."""
    if not facts.segments:
        return []
    lines = []
    for segment in facts.segments:
        if segment.churn_rate is None or facts.base_rate is None:
            lines.append(
                f"- **{segment.name}** - no churn label is available, so no action can be ranked "
                "from this dataset."
            )
            continue
        delta = float(segment.churn_rate) - float(facts.base_rate)
        if delta >= 0.05:
            lines.append(
                f"- **{segment.name}** - churns at {pct(segment.churn_rate)}, "
                f"{abs(delta) * 100:.1f} points above the base rate across "
                f"{segment.customers:,} customers. This is where retention effort pays off first."
            )
        elif delta <= -0.05:
            lines.append(
                f"- **{segment.name}** - churns at {pct(segment.churn_rate)}, "
                f"{abs(delta) * 100:.1f} points below the base rate. Low priority; protect the "
                "experience rather than spending retention budget here."
            )
        else:
            lines.append(
                f"- **{segment.name}** - churns at {pct(segment.churn_rate)}, in line with the "
                "base rate. Treat it as a steady-state group."
            )
    return lines


# ==========================================================================
# The summary
# ==========================================================================
def build_summary(facts: SummaryFacts) -> str:
    """Compose the final report as markdown, ready to render or download."""
    out: list[str] = ["## Final summary", "", _headline(facts), ""]

    # -- what was analysed ------------------------------------------------
    out.append("### What was analysed")
    out.append("")
    scope = (
        f"`{facts.source_name}` - {facts.grain_label.lower()}, "
        f"{facts.rows_in_file:,} rows in the file, {facts.rows_analysed:,} rows after cleaning, "
        f"reduced to {facts.customer_count:,} customers across {facts.columns_analysed} columns."
    )
    out.append(scope)
    if facts.observed and facts.target_column:
        out.append("")
        out.append(
            f"The churn label is the observed column `{facts.target_column}`, used exactly as "
            f"supplied. {pct(facts.churn_rate)} of customers are labelled as churned."
        )
    elif facts.synthetic:
        out.append("")
        rate_line = f"{pct(facts.churn_rate)} of customers meet that rule."
        out.append(
            f"The churn label is **synthetic**: {facts.target_rule} {rate_line} Performance "
            "against it measures how faithfully the model reproduces a rule that was chosen in "
            "advance - it is not evidence of real-world predictive power."
        )
    else:
        out.append("")
        out.append("No churn label was found and none could be derived, so no churn model was fitted.")
    out.append("")

    # -- segmentation -----------------------------------------------------
    if facts.segments:
        out.append("### Segments found")
        out.append("")
        detail = (
            f"We sorted the customers into {facts.k} groups by who they resemble. A fit quality "
            f"score of {facts.silhouette:.2f} on a 0 to 1 scale (higher is cleaner) says how "
            "distinct those groups are from each other."
        )
        if facts.k_reason:
            detail += f" How many groups to use was settled beforehand, on practice data only: {facts.k_reason}"
        out.append(detail)
        out.append("")
        out.extend(_segment_bullets(facts))
        out.append("")

    # -- churn model ------------------------------------------------------
    if facts.can_model:
        out.append("### Churn model")
        out.append("")
        out.append(
            f"The model was shown {facts.test_rows:,} customers it had never seen while learning, "
            f"having kept {facts.train_rows:,} back for practice. It gets {pct(facts.test_accuracy)} "
            f"of the new customers right, and its ranking quality is {num(facts.test_auc)} out of a "
            "possible 1, where 0.5 would be a coin toss."
        )
        out.append("")
        if facts.beats_baseline:
            out.append(
                f"That is {abs(float(facts.test_accuracy) - float(facts.baseline_accuracy)) * 100:.1f} "
                "points better than guessing that the majority stay, so what we know about these "
                "customers does carry usable signal."
            )
        else:
            out.append(
                "That is no better than guessing that the majority stay, which needs no data at all. "
                "On this dataset the model adds nothing, and no change of cut-off or model type "
                "can honestly change that."
            )
        if facts.warnings:
            out.append("")
            out.extend(f"- {warning}" for warning in facts.warnings)
        out.append("")

        drivers = _driver_lines(facts)
        if drivers:
            out.append("### What makes customers more or less likely to churn")
            out.append("")
            out.append(
                "The details the model leans on most. These are patterns in your data, not proof "
                "that one thing causes another."
            )
            out.append("")
            out.extend(f"- {line}" for line in drivers)
            out.append("")

        actions = _action_lines(facts)
        if actions:
            out.append("### What to do with it")
            out.append("")
            out.extend(actions)
            out.append("")

    # -- limitations ------------------------------------------------------
    out.append("### What this report cannot tell you")
    out.append("")
    if facts.excluded_columns:
        shown = "; ".join(facts.excluded_examples)
        extra = facts.excluded_columns - len(facts.excluded_examples)
        more = "" if extra <= 0 else f", and {count_phrase(extra)} more for the same kind of reason"
        out.append(
            f"{count_phrase(facts.excluded_columns).capitalize()} "
            f"{'was' if facts.excluded_columns == 1 else 'were'} held back from the model{more}: "
            f"{shown}."
        )
    if facts.synthetic:
        out.append("")
        out.append(
            "Because the churn label is synthetic, treat the segment actions above as a test of the "
            "method, not as a business finding."
        )
    if not facts.can_model:
        out.append("")
        out.append(
            "No churn model means no risk ranking is available. Segmentation and data-quality "
            "conclusions still stand."
        )
    for note in facts.notes:
        out.append("")
        out.append(note)
    out.append("")

    # -- provenance -------------------------------------------------------
    out.append("### How these numbers were produced")
    out.append("")
    out.append(
        f"Customers were split into a practice group and a held-back group, "
        f"{facts.test_size:.0%} held back, keeping the proportion of churners the same in both "
        f"so the split could not be luck. The repeat run used seed {facts.random_state}. Filling in "
        "gaps, rescaling, turning words into numbers, grouping customers and fitting the model "
        "were all worked out on the practice customers only. The held-back customers were then "
        "scored once, at the very end, and a customer is called at risk at "
        f"{facts.threshold:.0%} likelihood or higher"
        + (
            f". A different cut-off of {facts.suggested_threshold:.2f} fitted the practice data "
            "slightly better, but no number above was recalculated with it - it is shown for "
            "reference only."
            if abs(float(facts.suggested_threshold) - float(facts.threshold)) > 1e-9
            else "."
        )
    )
    return "\n".join(out).strip() + "\n"


def build_plain_summary(facts: SummaryFacts) -> str:
    """The same report as plain text, for the downloadable copy."""
    text = build_summary(facts)
    keep = (
        "##", "###", "-", "**",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(keep):
            lines.append(stripped.lstrip("#").replace("**", "").lstrip("- ").strip())
        else:
            lines.append(stripped)
    return "\n".join(lines).strip() + "\n"


def summary_frame(facts: SummaryFacts) -> pd.DataFrame:
    """One row per segment, the machine-readable twin of the prose summary."""
    rows = []
    for segment in facts.segments:
        rows.append(
            {
                "Segment": segment.name,
                "Customers": segment.customers,
                "Share of base": segment.share,
                "Churn rate": segment.churn_rate,
                "Base rate": facts.base_rate,
                "Defining traits": segment.traits,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Segment",
            "Customers",
            "Share of base",
            "Churn rate",
            "Base rate",
            "Defining traits",
        ],
    )
