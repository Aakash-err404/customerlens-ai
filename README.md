# CUSTOMERLENS AI

**Customer segmentation and churn prediction that reads any CSV.**

Point it at a spreadsheet and it works out what the columns mean, sorts customers
into groups, and fits a churn model — then shows the evidence next to every number
it reports. No column names, target column or file format is hard-coded.

- **Two pages.** Pick a file in the sidebar, read the results, read the summary.
- **Two questions answered.** Who the customers are (segmentation), and who is
  likely to leave (churn), with the two drawn together.
- **Honest by construction.** The headline verdict is decided against a
  majority-class baseline, not an absolute target, and the app says plainly when
  a model adds nothing.
- **Written for non-analysts.** Every headline, caption and table heading
  explains what the number means in ordinary words. The technical statements are
  still there, one click away.
- **Deployable.** No runtime writes, no absolute paths, no network calls.

---

## Contents

- [Quick start](#quick-start)
- [What you get](#what-you-get)
- [Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud)
- [How it works](#how-it-works)
- [Measured results](#measured-results)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

---

## Quick start

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, usually <http://localhost:8501>. The interface
opens in a dark theme with no dataset loaded.

To see it work immediately, load one of the two bundled files from the sidebar
under **Dataset → Bundled example**:

| Example | Rows | What it is |
| --- | --- | --- |
| `Online_Retail.csv` | 541,909 transactions | A real sales log. Rows are events, so the app aggregates to 4,372 customers and builds 31 features. No churn column exists, so the target is derived from a 90-day inactivity rule and labelled **synthetic** everywhere it appears. |
| `online_retail_customer_churn.csv` | 1,000 customers | A customer-level table with a real `Target_Churn` column, so the label is used exactly as supplied. |

To use your own data, choose **Upload a CSV** in the sidebar. Processing runs
entirely in memory: the file is never written to disk or sent anywhere.

## What you get

### Page 1 — Results

| Section | What it answers |
| --- | --- |
| Headline row | How many customers, what share churn, how many groups, how often the model is right on customers it has never seen, and how that compares with always guessing the majority stay. |
| **The customer groups** | Every customer sorted into groups, with each group's size, share and the share that actually churned. |
| **How the groups connect to churn** | Two charts drawn only from customers held back from training: the model's average prediction per group against what really happened, and how risk is spread inside each group. |
| **Churn prediction** | Whether the model is worth using, in one sentence, plus the full scoreboard and the ROC curve. |
| **What makes customers more or less likely to churn** | The details the model leans on most, and in which direction. |
| **What makes each group different** | A group-by-feature heatmap whose column labels repeat whether each feature raises or lowers risk, so a group worth acting on is visible at a glance. |
| **Customers most likely to churn** | The highest-risk customers, ranked. |
| Evidence | Collapsed: data quality, detected schema, every excluded column and why, the reasoning behind the number of groups, and the technical version of each claim above. |

### Page 2 — Final Summary

A written report generated from the numbers above — what was analysed, how the
churn target was obtained, the groups and their churn rates, whether the model is
worth using, the strongest drivers, what to do next, and an explicit list of what
the report cannot tell you. Followed by the group profiles and every download,
including the report as markdown, plain text and CSV.

Charts are limited to the five that change a decision: churn rate by group,
prediction against reality per group, risk spread within each group, the ROC
curve, and the driver bars. The underlying numbers for everything else are in
the downloads.

## Deploy to Streamlit Community Cloud

Push the repository, create the app in [Streamlit Community Cloud](https://share.streamlit.io)
and point it at the repo root:

| Setting | Value |
| --- | --- |
| Main file path | `app.py` |
| Python version | 3.12 or newer (3.14.4 used for the numbers below) |
| Requirements file | `requirements.txt` (auto-detected) |

`datasets/` is committed so the bundled examples appear straight away. The app
works without it — the example list is simply empty — and the repository is
around 42 MB because of `Online_Retail.csv`.

One caveat for a free Cloud instance: the 541,909-row example peaks at roughly
1.05 GB of resident memory, which is at or slightly over the 1 GB free-tier
allowance. The 1,000-row example is comfortably within it. A paid instance, or
the smaller file, avoids the risk.

## How it works

1. **Read** — encoding and separator are detected from the bytes (`utf-8`,
   `utf-8-sig`, `cp1252`, `latin-1` × `,` `;` tab `|`), never assumed.
2. **Clean** — normalise names and text, drop duplicate rows, all-null and
   constant columns, coerce numeric-looking columns, parse dates, replace
   infinities. Every action is reported.
3. **Profile and classify** — each column gets a semantic kind, an identifier
   score with reasons, and a target score with reasons. A verdict states whether
   the rows are transactions or customers, with the evidence.
4. **Aggregate** — for a transaction grain, per-customer features are derived
   (RFM, tenure, basket behaviour, value, returns, category diversity). A feature
   is only built when the columns it needs exist and hold usable data.
5. **Build the target** — a real churn-like column is used if one is found, with
   the encoding mapping shown. Otherwise a target is derived from an explicit
   inactivity rule, labelled `synthetic` everywhere, with the rule printed in
   words. Columns used to build the target are excluded from the features.
6. **Select features** — identifiers, the target, constants, near-constants,
   over-missing columns and high-cardinality text are removed, each with a stated
   reason.
7. **Segment** — `k` is chosen from the silhouette score cross-checked against the
   elbow; a split where a couple of outliers form their own "cluster" is rejected
   as degenerate. Group names come from rank-based tiers computed from the data.
8. **Model** — two logistic regressions, tuned and compared. See below.
9. **Report** — metrics appear as measured, beside a majority-class baseline, with
   a plain-language note whenever a result is not meaningful.

### Train/test discipline

The order of operations is the point:

```
stratified split  →  fit imputer / scaler / encoder on TRAIN  →  K-Means on TRAIN
                  →  one-hot segment encoder on TRAIN labels  →  tune both models on TRAIN
                  →  transform TEST and score ONCE
```

The customer identifier, the target and anything derived from the target never
reach the models.

The number of groups is chosen once, from the training partition only. That same
`k` is then used twice: the segmentation shown on the pages is refitted on all
customers and labelled *descriptive*, because it is a description rather than a
prediction, so every customer appears in the group table and in the risk table;
the segmentation that feeds the churn model is the training-only fit, so the
held-back rows never influence the model.

### Two models, compared

- **Model A** — logistic regression on the preprocessed features.
- **Model B** — the same, with the K-Means segment one-hot encoded and appended.

Both are tuned with `GridSearchCV` under stratified cross-validation, scored on
ROC-AUC, and reported next to a **majority-class baseline** — a model that uses no
features at all — so the numbers can be judged honestly.

## Measured results

Reproduced from the bundled fixtures (Python 3.14.4, pandas 3.0.2, scikit-learn
1.8.0, 20% stratified test share, seed 42, 5 CV folds):

| Fixture | Grain | Customers | Target | k | Test accuracy | Majority baseline | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Online_Retail.csv` | transactions | 4,372 | synthetic (90-day inactivity) | 2 | **50.97%** | 66.51% | worse than guessing |
| `online_retail_customer_churn.csv` | customers | 1,000 | observed `Target_Churn` | 2 | **53.00%** | 52.50% | ties guessing |

**These results are reported as measured, not as a success.** Neither model beats
the majority-class baseline by enough to be useful, and the app says so in plain
language instead of presenting the number as an achievement.

An 85% benchmark exists in the code as a reference constant but is deliberately
not shown as a badge: on a file where 85% of customers churn, always predicting
"churned" clears 85% using no features at all. The verdict is decided against the
baseline, and there are three outcomes — better than baseline, only tied, or worse.

The reason is visible rather than hidden. On the transaction fixture the churn
labels are *synthetic*, defined by the same recency figure the model would have to
predict, so once that column is excluded there is nothing left to learn. That is a
property of the data, not a defect in the model, and the diagnostic panel reports
it. On the customer fixture no individual attribute separates leavers from
stayers at all, which the univariate check states in plain words.

Both fixtures run warning-free. The 541,909-row file profiles, aggregates,
segments and models in roughly 15–30 seconds on a laptop.

## Design decisions worth knowing

**`log1p` before standardising.** Transaction features are right-skewed by
construction — a handful of wholesale customers sit dozens of standard deviations
from the mean. `SkewedLogStandardScaler` learns from the training partition which
non-negative columns are skewed enough to compress, then applies `log1p` before
`StandardScaler`. Without it K-Means degenerates into "outliers vs everyone else"
(a 4,370/2 split); with it the split is 2,491/1,881.

**No `saga`.** The solver grid covers `liblinear` and `lbfgs` only. `saga` on a
wide one-hot matrix made a single fit unusably slow, and the search found the same
hyperparameters without it.

**Version-aware parameter grid.** scikit-learn 1.8 changed `penalty`/`l1_ratio`
handling, so the grid is built for the installed version and the app emits no
deprecation warnings.

**The threshold never moves itself.** The default is 0.50 and is used for every
reported metric. The F1-optimal threshold from out-of-fold predictions is computed
and shown for reference only; no metric is recomputed at it, and there is no
control to move it.

**Risk bands are relative, not absolute.** Fixed probability cut-offs put nearly
every customer in one band whenever a model barely separates churners, which
hides the ordering the model found. The risk chart therefore splits this file's
own predictions into thirds.

**Upload failures surface as messages.** A file that cannot be read produces an
error in the interface, not a stack trace.

## Known limitations

v1.0.0 has four open defects, all affecting arbitrary uploaded files rather than
the two bundled fixtures. They are listed here rather than hidden, and the first
two are scheduled for the next release.

1. **Whole-number columns can raise an error.** An integer column with no missing
   values survives numeric coercion as `int64`, and the skew check then calls
   `float.is_integer` on it, raising `TypeError`. Such a file cannot be analysed.
2. **Alphanumeric identifiers can be read as numbers.** The currency-stripping
   step removes every character outside a numeric alphabet, so `C00001`, `SKU0001`
   and `R0` become `1`, `1` and `0`. The column is then treated as a numeric
   feature, which is wrong and can silently distort the model.
3. **Column-role detection can pick the target.** With no product-like column name
   to match, the fallback that chooses a product column can select a
   low-cardinality column, including the detected churn target.
4. **Aggregate-only features are not reported.** For transaction-grain files,
   features requiring a sum across rows are built for the model but omitted from
   the group profiles, so reported group averages cover per-row features only.

The app is also memory-hungry on very large files; see the Cloud note above.

## Project layout

```
app.py                     Streamlit interface, session state, page routing
src/
  constants.py             vocabularies, thresholds, defaults, page list, example discovery
  data_detection.py        CSV reading, datetime inference, profiling, role scoring, grain verdict
  data_processing.py       cleaning, coercion, target construction (observed + synthetic)
  feature_engineering.py   aggregation, feature selection, preprocessing, SkewedLogStandardScaler
  clustering.py            k search, K-Means, cluster profiles, segment names, PCA
  churn_model.py           split, logistic grid, out-of-fold probabilities, coefficients, risk bands
  evaluation.py            held-out metrics, baseline, diagnostics, threshold sweep, verdicts
  summary.py               the written final summary, composed from measured values only
  visualization.py         every Plotly figure; the app renders five of them
datasets/                  sample data, unmodified
devtools/                  headless test harnesses, not part of the app
.streamlit/config.toml     dark theme and upload size
```

`summary.py` holds no Streamlit import and no dependency on the app's own
dataclasses. It takes a `SummaryFacts` record of measured values and returns the
report as markdown, plain text, or a one-row-per-group table, which is why every
sentence in the summary is traceable to a number.

## Tests

Three harnesses, no test framework required:

```bash
python devtools/pipeline_test.py     # end-to-end pipeline on both fixtures
python devtools/app_smoke.py         # both page functions, both fixtures, headless
python devtools/app_apptest.py       # the whole app in a real Streamlit runtime
```

The first prints the full pipeline for each fixture, the second renders both pages
for both fixtures, and the third drives the real app through Streamlit's
`AppTest`: cold start, loading an example, automatic fitting, both pages, the
summary, reset, and the empty-upload state.

## Contributing

Issues and pull requests are welcome. Please open an issue before starting
anything large, and keep these rules intact — they are what makes the numbers
trustworthy:

- Split before fitting. Nothing derived from the test partition may reach a
  fitted component.
- Never hard-code a filename, column name or target column.
- Every excluded column, chosen role and derived target needs a stated reason,
  and it has to be shown in the interface.
- Report a metric as measured, next to a baseline. No absolute target badges.
- Add or update a harness in `devtools/` for any change to the pipeline.

## License

MIT — see [LICENSE](LICENSE).

The bundled datasets keep their own provenance and are included for
demonstration only.
