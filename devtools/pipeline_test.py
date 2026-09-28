"""Development-only: run the full ML pipeline headlessly against both fixtures."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clustering, constants as C, data_detection as dd, data_processing as dp
from src import evaluation as ev
from src import feature_engineering as fe
from src.churn_model import (
    attach_segments,
    fit_segment_feature,
    generate_risk_categories,
    make_stratified_split,
    train_logistic_model,
)

FIXTURES = Path(__file__).resolve().parents[1] / "datasets"


def banner(text: str) -> None:
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def run(path: Path) -> None:
    banner(f"FIXTURE: {path.name}")
    t0 = time.time()
    raw = pd.read_csv(path, encoding="latin-1" if "Online_Retail" in path.name else "utf-8", low_memory=False)
    print(f"raw shape {raw.shape}")

    frame, report = dp.clean_dataframe(raw)
    print(f"cleaned -> {frame.shape}; duplicates removed {report.duplicate_rows_removed:,}")
    frame, coerced, _ = dp.coerce_numeric_columns(frame)
    frame, date_report = dd.infer_datetime_columns(frame)
    print("date columns detected:", list(date_report))
    if coerced:
        print("coerced to numeric:", coerced)

    profile = dd.profile_dataset(frame)
    print(f"numeric={profile.numeric}")
    print(f"categorical={profile.categorical}")
    print(f"boolean={profile.booleans}")
    print(f"datetime={profile.datetimes}")
    print("id columns:", dd.detect_id_columns(profile))
    best, ranked, _ = dd.detect_target_column(frame, profile)
    print("target candidates:", [(n, s) for n, s, _ in ranked], "-> best:", best)

    schema = dd.detect_transaction_columns(frame, profile)
    print("schema:", schema.to_dict())
    verdict = dd.detect_dataset_type(frame, profile, schema)
    print(f"VERDICT: {verdict.label} (score {verdict.score}, {verdict.rows_per_entity})")
    for r in verdict.reasons:
        print("   -", r)

    if verdict.is_transaction:
        print("\nrows without customer id are dropped inside the aggregator")
        agg = fe.aggregate_transactions(frame, schema, profile=profile)
        customers = agg.customers
        print(f"customers: {customers.shape}")
        print(agg.specs_frame().to_string(index=False))
        for note in agg.notes:
            print("   note:", note)
        customer_df = customers
        customer_id = schema.customer_id
        target_column = None
        target_origin = "synthetic"
    else:
        customer_id = dd.detect_id_columns(profile)[0] if dd.detect_id_columns(profile) else None
        customer_df, specs, notes = fe.engineer_customer_features(frame, profile, customer_id=customer_id)
        print("engineered notes:", notes)
        target_column = best
        target_origin = "observed"
        if target_column is None:
            raise SystemExit("no target")

    print("\ncustomer-level columns:", list(customer_df.columns))
    sub_profile = dd.profile_dataset(customer_df)
    selection = fe.select_features(
        customer_df,
        sub_profile,
        customer_id=customer_id,
        target_column=target_column,
        id_columns=[customer_id] if customer_id else [],
    )
    print("SELECTED:", selection.all_features)
    print("EXCLUDED:")
    print(selection.excluded_frame().to_string(index=False))

    # ---- target ------------------------------------------------------
    excluded_features: list[str] = []
    if target_origin == "observed":
        tb = dp.build_observed_target(customer_df, target_column)
    else:
        recency = "recency_days" if "recency_days" in customer_df.columns else "frequency"
        tb = dp.build_synthetic_inactivity_target(
            customer_df,
            customer_col=customer_id,
            recency_col=recency,
            inactivity_days=90,
            snapshot=frame[schema.date].max() + pd.Timedelta(days=1) if schema.date else None,
            frequency_col="frequency",
            require_low_frequency=False,
        )
    excluded_features += tb.excluded_features
    print("\ntarget:", tb.origin, "rate", round(tb.churn_rate, 4))
    print("   ", tb.description)
    for w in tb.warnings:
        print("   warn:", w)

    selection = fe.select_features(
        customer_df,
        sub_profile,
        customer_id=customer_id,
        target_column=target_column,
        id_columns=[customer_id] if customer_id else [],
        extra_excluded=excluded_features,
        extra_reasons={f: "Defines the synthetic churn label - using it would leak the rule." for f in excluded_features},
    )
    print("SELECTED (post guard):", selection.all_features)

    X_frame = customer_df[selection.all_features].copy()
    for c in selection.boolean:
        X_frame[c] = X_frame[c].astype(float)
    pre = fe.build_preprocessor(selection)
    matrix, num_block, names = fe.prepare_matrix(X_frame, selection, pre)
    print("matrix", matrix.shape, "numeric block", num_block.shape)

    # ---- clustering on ALL customers (exploratory) -------------------
    t = time.time()
    search = clustering.find_optimal_k(num_block)
    print("\nk search:\n", search.table.to_string(index=False))
    print("recommended k:", search.recommended_k, "| elbow k:", search.elbow_k)
    print("reason:", search.reason)
    k = search.recommended_k
    model_full = clustering.train_kmeans(num_block, k=k, feature_names=selection.numeric + selection.boolean)
    print(f"kmeans fit {time.time() - t:.1f}s silhouette={model_full.silhouette:.4f} inertia={model_full.inertia:.1f}")
    prof = clustering.generate_cluster_profiles(
        num_block,
        model_full.predict(num_block),
        feature_names=selection.numeric + selection.boolean,
        raw_frame=X_frame,
        categorical_features=selection.categorical,
    )
    seg_names = clustering.generate_segment_names(
        prof, X_frame, feature_names=selection.numeric + selection.boolean
    )
    print("segment names:", seg_names)
    print(prof.table[["Cluster", "Customers", "Share"]].to_string(index=False))
    print("\ntop discriminating:\n", clustering.top_discriminating_features(prof.z_table).to_string(index=False))

    # ---- churn: split, KMeans on TRAIN only, two models ---------------
    y = tb.y.reindex(customer_df.index).to_numpy(dtype=int)
    if target_origin == "observed":
        y = pd.Series(tb.y.to_numpy(dtype=int), index=tb.y.index)
        mask = customer_df[selection.all_features].notna().all(axis=1) | True
        y = y.reindex(customer_df.index).to_numpy(dtype=int)
    split = make_stratified_split(y, test_size=0.2, random_state=42)
    print("\nsplit:", split.n_train, split.n_test, split.train_class_counts, split.test_class_counts)

    X_tr, X_te = matrix[split.train_idx], matrix[split.test_idx]
    N = len(selection.numeric) + len(selection.boolean)
    nblk_tr, nblk_te = num_block[split.train_idx], num_block[split.test_idx]

    t = time.time()
    km_train = clustering.train_kmeans(
        nblk_tr, k=k, feature_names=selection.numeric + selection.boolean, trained_on="training partition only"
    )
    seg_tr = km_train.predict(nblk_tr)
    seg_te = km_train.predict(nblk_te)  # transform only - never refit
    seg_enc = fit_segment_feature(seg_tr, k)
    Xtr_b = attach_segments(X_tr, seg_tr, seg_enc)
    Xte_b = attach_segments(X_te, seg_te, seg_enc)
    print(f"kmeans(train) silhouette={km_train.silhouette:.4f}  segment spread={np.bincount(seg_tr, minlength=k)}")
    print(f"clustering + encoding {time.time() - t:.1f}s")

    t = time.time()
    model_a = train_logistic_model(
        X_tr, split.y_train, label="Logistic Regression", feature_names=names, scoring="roc_auc"
    )
    model_b = train_logistic_model(
        Xtr_b, split.y_train, label="Logistic + K-Means", feature_names=names + seg_enc.column_names,
        scoring="roc_auc", uses_segment=True,
    )
    print(f"model training {time.time() - t:.1f}s")
    for m in (model_a, model_b):
        print(f"{m.label}: best={m.best_params} cv={m.best_score:.4f} trainacc={m.train_accuracy:.4f} cvacc={m.cv_accuracy:.4f}")

    res_a = ev.evaluate_model(split.y_test, model_a.predict_proba(X_te), label=model_a.label,
                              train_accuracy=model_a.train_accuracy, cv_accuracy=model_a.cv_accuracy)
    res_b = ev.evaluate_model(split.y_test, model_b.predict_proba(Xte_b), label=model_b.label,
                              uses_segment=True, train_accuracy=model_b.train_accuracy, cv_accuracy=model_b.cv_accuracy)
    baseline = ev.majority_baseline(split.y_test)
    table = ev.compare_models([res_a, res_b, baseline])
    print("\n", table.to_string(index=False))
    print("\nverdict:", ev.comparison_verdict(table))
    print("baseline:", ev.baseline_verdict(table))
    print("banner:", ev.performance_banner(res_a.metrics["Accuracy"]))

    from src.churn_model import feature_coefficients
    print("\ntop coefficients:\n", feature_coefficients(model_a, names).head(8).to_string(index=False))

    uni = ev.univariate_target_association(X_frame, y)
    print("\nunivariate:\n", uni.head(8).to_string(index=False))
    diag = ev.run_diagnostics(result=res_a, train_accuracy=model_a.train_accuracy,
                              cv_accuracy=model_a.cv_accuracy, univariate=uni,
                              excluded_features=excluded_features, target_origin=target_origin)
    print("signal detected:", diag.signal_detected)
    for w in diag.warnings:
        print("  WARN:", w)

    sweep = ev.threshold_sweep(model_a.oof_probabilities, split.y_train)
    print("\noof threshold sweep head:\n", sweep.head(3).to_string(index=False))
    risk = generate_risk_categories(model_a.predict_proba(X_te))
    print("risk counts:", pd.Series(risk).value_counts().to_dict())
    print(f"\nTOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["online_retail_customer_churn.csv", "Online_Retail.csv"]:
        run(FIXTURES / name)
