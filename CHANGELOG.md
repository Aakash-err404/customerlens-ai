# Changelog

All notable changes to CUSTOMERLENS AI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. Planned for the next iteration:

- Fix the two known loader defects listed under v1.0.0 "Known limitations".
- Replace the fixed risk bands with bands that reflect the score distribution of
  the file being analysed.

## [1.0.0] - 2026-09-28

First public release.

### Added

- Adaptive CSV ingestion: encoding, separator and delimiter sniffing from the raw
  bytes, with no hard-coded filenames, column names or target column.
- Dataset classification that decides whether rows are transactions or customers
  and states the evidence behind the decision.
- Per-customer feature engineering (RFM, tenure, basket behaviour, monetary
  value, returns and category diversity), built only from columns that exist and
  hold usable data.
- Churn target construction from an observed column when one is found, otherwise
  from an explicit inactivity rule that is labelled `synthetic` everywhere it
  appears, with the rule printed in words.
- Feature selection with a stated reason for every excluded column: identifiers,
  the target, constants, near-constants, over-missing columns and high-cardinality
  text.
- K-Means segmentation with `k` chosen from the silhouette score cross-checked
  against the elbow, degenerate splits rejected, and segment names derived from
  rank-based tiers in the data.
- Two tuned logistic regressions (with and without the K-Means segment as a
  feature), compared against a majority-class baseline.
- A two-page Streamlit interface: **Results** and **Final Summary**, with dataset
  upload and settings in the sidebar.
- **How the groups connect to churn**: the model's average prediction per group
  against what actually happened, and the spread of risk inside each group, both
  computed only on customers held back from training.
- **What makes each group different**: a group-by-feature heatmap whose column
  labels repeat whether each feature raises or lowers predicted churn.
- A generated written report, composed only from measured values, plus markdown,
  plain-text and CSV downloads.
- Dark theme throughout, including all charts.
- Plain-language wording on the pages and in the report, with the technical
  statements kept in a collapsed evidence section.
- Three headless test harnesses under `devtools/`.

### Security and correctness

- Strict train/test discipline: the split happens first, and imputation, scaling,
  encoding, K-Means and model tuning are fitted on the training partition only.
  The test partition is transformed and scored exactly once, at the end.
- The customer identifier, the churn target and any column derived from the
  target are excluded from the predictors, with a self-check that reports the
  exclusion.
- The decision threshold is fixed at 0.50 for every reported metric. An
  F1-optimal threshold is computed and shown for reference only, and no reported
  metric is recomputed at it.
- The headline verdict is decided against a majority-class baseline rather than
  an absolute benchmark, because on an imbalanced file an absolute bar can be
  cleared by a model that uses no features at all.
- No runtime writes, no hard-coded absolute paths and no network calls, so the
  app runs on a read-only host such as Streamlit Community Cloud.

### Known limitations

These are open defects in v1.0.0, listed rather than hidden. Both affect
arbitrary uploaded files; neither affects the two bundled fixtures, which run
clean.

1. **Whole-number columns can raise an error.** A column of integers with no
   missing values survives numeric coercion as `int64`, and the skew check then
   calls `float.is_integer` on it, which raises `TypeError`. The affected
   dataset cannot be analysed.
2. **Alphanumeric identifiers can be read as numbers.** The currency-stripping
   step removes every character outside a numeric alphabet, so a code such as
   `C00001`, `SKU0001` or `R0` becomes `1`, `1` and `0`. Such a column is then
   treated as a numeric feature, which is wrong and can silently distort the
   model.
3. **Column-role detection can pick the target.** When no column name matches a
   product-like term, the fallback for choosing a product column can select a
   low-cardinality column, including the detected churn target.
4. **Aggregate-only features are not reported.** For transaction-grain files,
   features that require summing across several rows are built for the model but
   left out of the segment profiles, so the reported group averages cover only
   per-row features.

[Unreleased]: https://github.com/Aakash-err404/customerlens-ai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Aakash-err404/customerlens-ai/releases/tag/v1.0.0
