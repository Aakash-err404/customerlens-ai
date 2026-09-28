"""Headless smoke test for app.py.

Drives every page function for every fixture through Streamlit's bare-mode
testing harness, so widget/API mismatches surface as real exceptions instead of
a blank page. Nothing here is part of the app.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app  # noqa: E402

FIXTURES = {
    "transactions": ROOT / "datasets" / "Online_Retail.csv",
    "customer": ROOT / "datasets" / "online_retail_customer_churn.csv",
}
SETTINGS = {
    "test_size": 0.2,
    "cv_folds": 5,
    "k": 0,
    "random_state": 42,
    "n_jobs": -1,
}


def exercise(name: str, payload: bytes) -> None:
    print("=" * 78)
    print(name)
    print("=" * 78)
    start = time.time()
    analysis = app.build_analysis(payload, name)
    print(
        f"  grain={analysis.verdict.label!r} customers={len(analysis.customers)} "
        f"features={len(analysis.selection.all_features)} "
        f"target={analysis.target.origin} rate={analysis.rate_text} "
        f"can_model={analysis.can_model}"
    )
    if not analysis.can_model:
        print("  SKIP: not modellable")
        return

    # call the undecorated function: the cache decorator needs a session context
    fit = getattr(app.run_modelling, "__wrapped__", app.run_modelling)
    modelling = fit(
        analysis.customers,
        analysis.selection,
        analysis.target.y.to_numpy(dtype=int),
        target_origin=analysis.target.origin,
        test_size=SETTINGS["test_size"],
        k_override=None,
        cv_folds=SETTINGS["cv_folds"],
        random_state=SETTINGS["random_state"],
        n_jobs=SETTINGS["n_jobs"],
    )
    print(
        f"  k={modelling.k} silhouette={modelling.kmeans.silhouette:.3f} "
        f"test_acc={modelling.result_a.metrics['Accuracy']:.4f} "
        f"baseline={modelling.baseline.metrics['Accuracy']:.4f}"
    )

    st.session_state["analysis"] = analysis
    st.session_state["modelling"] = modelling
    st.session_state["settings"] = SETTINGS

    for page in app.PAGES:
        started = time.time()
        try:
            app.PAGES[page]()
            print(f"  [ok]   {page} ({time.time() - started:.1f}s)")
        except Exception:
            print(f"  [FAIL] {page}")
            traceback.print_exc()
            raise

    # the summary text is the deliverable: make sure it actually composed
    facts = app.summary_facts(analysis, modelling)
    written = app.sm.build_summary(facts)
    if len(written) < 800 or "Final summary" not in written:
        raise AssertionError("summary text looks truncated")
    print(f"  [ok]   summary composed ({len(written)} chars, {len(facts.segments)} segments)")

    print(f"  total {time.time() - start:.1f}s")


def main() -> int:
    for name, path in FIXTURES.items():
        exercise(name, path.read_bytes())
    print("=" * 78)
    print("all pages rendered without error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
