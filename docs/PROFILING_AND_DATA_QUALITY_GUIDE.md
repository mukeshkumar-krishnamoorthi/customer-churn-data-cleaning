# Data Profiling & Data Quality — A Practical Guide

This document explains *how* and *why* `src/profiling/profiler.py` and
`src/validation/data_quality.py` were built the way they were, so you can
apply the same method to any dataset in the future — not just this one.

It's organized as: concepts first, then the decision process, then a
step-by-step recipe you can reuse.

---

## 1. Two different jobs: Profiling vs. Quality Validation

People often mix these up. They answer different questions:

|                      | Data Profiling                                  | Data Quality Validation                         |
|----------------------|-------------------------------------------------|-------------------------------------------------|
| Question it answers  | "What does this data actually look like?"       | "Does this data meet my expectations?"          |
| Needs rules upfront? | No — it's purely descriptive                    | Yes — you compare against a schema/ruleset      |
| Output               | Stats: min, max, mean, missing %, top values... | Pass/fail issues with severity, and a score     |
| When you run it      | First, on unfamiliar data                       | After profiling, and on every pipeline run      |
| Analogy              | A doctor taking your vitals                     | A doctor checking vitals against healthy ranges |

**You always profile before you validate.** You can't write a sensible rule
like "`tenure` should be between 0 and 100" until you've *seen* the data and
confirmed that's a reasonable range. Profiling is how you discover the rules;
validation is how you enforce them forever after.

---

## 2. Why the original code wasn't "production grade"

Before the rewrite, `profiler.py` had this:

```python
summary = pd.DataFrame({
    "Column": self.df.columns,
    "Min": self.df.min(),
    "Max": self.df.max(),
    ...
})
```

This looks fine but is a real bug: `self.df.columns` is a plain array (no
index), while `self.df.min()` returns a **pandas Series indexed by column
name**. When you hand pandas a mix of plain arrays and indexed Series in a
`DataFrame(...)` constructor, it aligns everything by index — silently
reshuffling rows. Running it against the real churn data actually produced a
column called `Memory` where every row showed the *same* value (the whole
dataframe's memory, not each column's) instead of per-column values.

The lesson: **a script that "runs without errors" is not the same as a
script that's correct.** Production-grade code needs to survive:

- Edge cases (empty dataframe, all-null column, single unique value)
- Mixed dtypes (numeric, text, boolean, dates all in one dataset)
- Pandas/library version differences (e.g. `dtype` printing `"str"` vs
  `"object"` depending on pandas version — see §4)
- Being run unattended, with **no one reading the console output** —
  it needs to fail loudly and structurally (exit codes, files), not just
  print something a human might miss.

---

## 3. Designing the Profiler (`profiler.py`)

### 3.1 Two levels of stats

Production profilers report at two levels, because they answer different
questions:

- **Dataset-level** (`dataset_summary()`): row/column counts, overall
  missing %, duplicate row %, memory footprint, dtype mix. This is your
  "at a glance, is this dataset healthy" view.
- **Column-level** (`profile_column()`): one column at a time, in depth.
  This is where the real signal is.

### 3.2 How to decide *what stats* to compute per column

The trick is: **branch by what kind of data the column holds**, because the
same stat means different things (or is meaningless) across types.

```
is_bool_dtype?      → true/false counts, true %
is_numeric_dtype?    → min/max/mean/median/std/quartiles/outliers/zeros/negatives
is_datetime_dtype?   → min/max/range in days
else (text/category) → top value, frequency, blank-string count, "looks numeric %"
```

Why these specific stats, not others?

- **Numeric**: mean/median/std tell you the shape (skew shows up as
  mean ≠ median). Quartiles + IQR let you flag outliers *without hardcoding
  a threshold* — the data defines its own "normal range." Zeros and
  negatives are called out separately because they're common real-world bugs
  (e.g. a negative `MonthlyCharges` is a red flag, but "negative" isn't
  automatically wrong for every numeric column, so we just report it and let
  the *quality* layer decide if it's an error).
- **Categorical/text**: `unique_pct` tells you if this is a true category
  (`gender` → 2 values) vs. free text or an ID (`unique_pct` near 100%).
  `blank_strings` catches a very common failure mode CSV files have: a
  "missing" value that isn't `NaN`, just an empty string — `isna()` will
  **not** catch that, so we check for it explicitly.
  `looks_numeric_pct` is the single most valuable field in this whole
  profiler: it's how you *discover* that `TotalCharges` is secretly a number
  stored as text (99.84% of values parse as numbers) — that finding is what
  drove the `numeric_strings` rule in the quality config later.
- **Outliers via IQR** (`Q1 - 1.5×IQR` to `Q3 + 1.5×IQR`) is the standard,
  distribution-agnostic method — it doesn't assume a bell curve like a
  z-score approach would, so it works even for skewed business data like
  charges or tenure.

### 3.3 Robustness rules that make it "production" rather than "notebook"

1. **Reject bad input early and loudly**: `__init__` raises `ValueError` on
   an empty dataframe instead of computing garbage stats silently.
2. **Guard every division**: every `x / n` in this codebase is written as
   `x / n if n else 0.0`. A dataset profiler that crashes on a one-row
   dataframe is not production-ready.
3. **Never assume a dtype string**: don't compare `str(series.dtype) ==
   "object"` — pandas versions differ (this project's pandas 3.0 reports
   `"str"`, older pandas reports `"object"`). Instead use the *behavioral*
   checks: `pd.api.types.is_numeric_dtype(series)`,
   `is_bool_dtype`, `is_datetime64_any_dtype`. These ask "does this column
   behave like a number/bool/date" rather than "does the label match a
   string I guessed," so they survive library upgrades.
4. **Everything returns plain dicts/DataFrames, not just prints**. A
   `print()`-only report is useless in a pipeline — nothing downstream can
   consume it. `to_report()` returns a JSON-serializable dict specifically
   so it can be written to `outputs/reports/profile_report.json` and read
   by another tool, a dashboard, or a future run for comparison ("did our
   missing % go up since last week?").
5. **Numpy types are not JSON types.** `numpy.int64` will crash
   `json.dump`. Every numeric stat is explicitly cast with `int(...)` /
   `float(...)` before being put in a dict, and `FileWriter.json` also has a
   `default=` fallback as a safety net (see `src/utils/writer.py`).

---

## 4. Designing the Data Quality engine (`data_quality.py`)

### 4.1 The core design decision: rules as *config*, not *code*

The old version had this hardcoded inside the class:

```python
EXPECTED_SCHEMA = {
    "gender": "object",
    "SeniorCitizen": "int64",
    ...
}
```

This means every time the rules change, you edit Python and redeploy. In
production, **the ruleset changes far more often than the engine does** —
new columns, new business constraints, a new allowed category. So the fix is
to separate:

- **The engine** (`data_quality.py`) — generic code that knows *how* to run
  a "range check" or a "category check," but has zero knowledge of this
  specific dataset.
- **The ruleset** (`config/quality_rules.json`) — a plain data file that
  says *what* the rules are for *this* dataset.

This is the same pattern used by real tools (Great Expectations, dbt tests,
Deequ): a small number of generic *check types*, applied to many
dataset-specific *rule instances* defined in config. It means:

- A non-engineer can update `quality_rules.json` without touching code.
- The same engine works on a completely different dataset — just swap the
  JSON.
- Rules are diffable/reviewable in a PR, same as code.

### 4.2 The check types, and why each one exists

| Check | What it catches | Real example from this dataset |
|---|---|---|
| `schema` | Column missing, or wrong "kind" (numeric/string/bool/datetime) | `TotalCharges` expected numeric but stored as text |
| `not_null` | Nulls in columns that must always be populated | none currently, but guards against upstream breakage |
| `missing_values` (warn) | High missing % in non-critical columns | — |
| `duplicate_rows` | Exact-duplicate rows (usually an ingestion bug) | 22 duplicate rows found |
| `numeric_strings` | A column that's semantically numeric but stored as text with junk values | 11 blank `TotalCharges` values |
| `range` | Values outside a plausible min/max | negative charges, tenure > 100 |
| `category` | Values outside an allowed set | a `gender` value that isn't Male/Female |
| `uniqueness` | Duplicate values in a column that should be an ID | — (no ID column in this dataset) |
| `consistency` | Cross-column business-logic mismatches | `TotalCharges` should ≈ `tenure × MonthlyCharges`; 151 rows deviate >15% |

Each check is intentionally **narrow and composable** — one check, one job
— rather than one giant "validate everything" function. That's what lets
you enable/disable or tune individual checks per dataset via config.

### 4.3 Severity: error vs. warning

Not every quality problem is equally bad. Two severities keep the signal
useful:

- **error** — the data is *provably wrong* or violates a hard constraint
  (a required column is null, a numeric column has unparseable text, a
  supposedly-unique ID repeats). These should usually block a pipeline.
- **warning** — the data is *suspicious* but not provably wrong (a schema
  drift, a duplicate row, a value that doesn't reconcile with another
  column within tolerance). These get logged and reviewed, not necessarily
  blocked on.

Deciding which bucket a check belongs in is a judgment call, and it belongs
in the config layer, not hardcoded — e.g. `duplicates_allowed: true/false`
in `quality_rules.json` controls whether duplicate rows are an error or a
warning, because "are duplicates okay" genuinely varies by dataset.

### 4.4 The quality score

```python
penalty = len(errors) * 15 + len(warnings) * 5
score = max(0, 100 - penalty)
```

Kept deliberately simple and *deterministic* rather than a fancy weighted
formula: it's easy to explain to a non-engineer ("15 points per hard error,
5 per warning"), easy to unit test, and easy to reason about when it
changes. A gate then compares this to a threshold:

```python
if quality_report.score < QUALITY_GATE_MIN_SCORE:
    sys.exit(1)   # fail the pipeline
```

This "quality gate" pattern — compute a score, fail the build/pipeline
below a threshold — is exactly how production data pipelines stop bad data
from silently flowing downstream (same idea as a CI test suite blocking a
merge).

---

## 5. How the schema and rules were actually decided

This is the part that's hardest to teach from code alone, because it's a
*process*, not a formula. Here's the actual sequence used on this dataset:

1. **Load and profile first, with zero assumptions.** Ran `DataProfiler`
   over the raw CSV before writing a single rule.
2. **Read the profiler output for anomalies**, specifically:
   - Any `dtype` that seems wrong for the column's meaning (`TotalCharges`
     showed `dtype: str` — a "Charges" column should be numeric → schema
     rule).
   - Any `looks_numeric_pct` under 100% on a column that should be fully
     numeric → tells you exactly how bad the coercion problem is (99.84%
     here → 11 bad rows, not a mass failure).
   - `blank_strings > 0` on a column with `missing == 0` → the classic
     "missing values disguised as empty strings" trap. This is *why*
     `check_missing_values()` alone isn't enough — it only sees `NaN`.
   - `unique` counts on categorical columns (`gender` → 2, `Churn` → 2) →
     directly became the `categories` allow-lists.
3. **Bring in domain knowledge the data alone can't tell you.** No amount
   of profiling tells you that `TotalCharges` *should* roughly equal
   `tenure × MonthlyCharges` — that's a business/domain fact about how
   billing works. This became the one `consistency` rule. In general:
   *statistical* rules (ranges, categories, missing %) come from profiling;
   *business-logic* rules (this equals that times that) come from
   understanding what the columns mean.
4. **Set thresholds from what's actually observed, not guessed.** The
   `tolerance_pct: 15` on the consistency check and `missing_warn_pct: 5`
   weren't picked arbitrarily — they came from running the check at a
   stricter threshold first, seeing how many rows it flagged, and loosening
   it until only genuinely inconsistent rows remained (some deviation is
   expected from prorated first/last months, discounts, etc. — you don't
   want your quality gate crying wolf on legitimate business noise).
5. **Decide error vs. warning by asking "would I block a deploy on this?"**
   A required column being null → yes, block (`error`). 22 duplicate rows
   out of 7000 → worth knowing, not worth blocking (`warning`).
6. **Only encode what you can currently observe or justify** — resist the
   urge to add speculative rules "just in case." `unique: []` (no ID
   uniqueness rule) is empty on purpose: this dataset has no ID column, so
   there's nothing to validate. Don't invent rules for columns that don't
   exist.

---

## 6. The reusable recipe (for any new dataset)

1. Load the data. Run a profiler exactly like `DataProfiler` over it with
   **no rules yet** — just look.
2. For every column, ask three questions:
   - What *kind* is this (numeric / category / free text / boolean / date)?
   - What dtype does pandas think it is, and does that match what the
     column *means*?
   - What's implausible in the min/max/top-values/blank-string output?
3. Turn each anomaly you find into exactly one rule, in config, in the
   narrowest check type that fits (schema / not_null / range / category /
   numeric_strings / uniqueness).
4. Add any *business-logic* rules a domain expert would know but the data
   alone wouldn't reveal (cross-column consistency).
5. Decide severity per rule: hard constraint → `error`; "worth
   investigating" → `warning`.
6. Run it, read the actual issue counts, and tune thresholds so real
   problems are flagged and normal noise isn't.
7. Wire it into your pipeline with a score threshold that fails the run
   when data quality drops — don't just log and move on.
8. Write tests for the engine using small, hand-built DataFrames where you
   *know* the expected violations (see `tests/test_data_quality.py`) — this
   is what lets you safely add new rules later without silently breaking
   old ones.

---

## 7. Where to look in this repo

| Concept | File |
|---|---|
| Profiler implementation | `src/profiling/profiler.py` |
| Quality engine implementation | `src/validation/data_quality.py` |
| This dataset's actual rules | `config/quality_rules.json` |
| Pipeline wiring + quality gate | `src/main.py` |
| Tests showing expected behavior | `tests/test_profiler.py`, `tests/test_data_quality.py` |
| Generated reports (run `python src/main.py`) | `outputs/reports/*.json`, `outputs/reports/*.csv` |
