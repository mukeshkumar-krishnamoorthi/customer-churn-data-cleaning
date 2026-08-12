from dataclasses import asdict, dataclass, field

import pandas as pd

DEFAULT_MISSING_WARN_PCT = 5.0

_KIND_CHECKS = {
    "numeric": pd.api.types.is_numeric_dtype,
    "string": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "boolean": pd.api.types.is_bool_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
}


@dataclass
class QualityIssue:
    check: str
    column: str | None
    severity: str  # "error" | "warning"
    message: str
    count: int = 0
    pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityReport:
    score: float
    rows: int
    columns: int
    error_count: int
    warning_count: int
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "rows": self.rows,
            "columns": self.columns,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["check", "column", "severity", "message", "count", "pct"])

        return pd.DataFrame(issue.to_dict() for issue in self.issues)


class DataQuality:
    """
    Rule-driven data-quality engine.

    Rules are supplied as a plain dict (typically loaded from a JSON config)
    so quality expectations can evolve per-dataset without touching this engine.
    Supported keys:
      - schema: {column: "numeric" | "string" | "boolean" | "datetime"}
      - not_null: [column, ...]
      - missing_warn_pct: float
      - duplicates_allowed: bool
      - numeric_strings: [column, ...]  (numeric values stored as text)
      - ranges: {column: {"min": x, "max": y}}
      - categories: {column: [allowed values]}
      - unique: [column, ...]
      - consistency: [{"type": "product_match", "a": col, "b": col, "target": col, "tolerance_pct": float}]
    """

    def __init__(self, df: pd.DataFrame, rules: dict | None = None):
        if df is None or df.empty:
            raise ValueError("DataQuality requires a non-empty DataFrame")

        self.df = df
        self.rules = rules or {}

    def run(self) -> QualityReport:
        issues: list[QualityIssue] = []

        issues += self._check_schema()
        issues += self._check_missing_values()
        issues += self._check_duplicate_rows()
        issues += self._check_numeric_strings()
        issues += self._check_ranges()
        issues += self._check_categories()
        issues += self._check_uniqueness()
        issues += self._check_consistency()

        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]

        return QualityReport(
            score=self._compute_score(errors, warnings),
            rows=self.df.shape[0],
            columns=self.df.shape[1],
            error_count=len(errors),
            warning_count=len(warnings),
            issues=issues,
        )

    def _compute_score(self, errors: list[QualityIssue], warnings: list[QualityIssue]) -> float:
        penalty = len(errors) * 15 + len(warnings) * 5
        return max(0.0, round(100 - penalty, 2))

    def _check_schema(self) -> list[QualityIssue]:
        issues = []

        for column, kind in self.rules.get("schema", {}).items():
            if column not in self.df.columns:
                issues.append(QualityIssue(
                    "schema", column, "error",
                    f"Expected column '{column}' not found in dataset",
                ))
                continue

            check_fn = _KIND_CHECKS.get(kind)

            if check_fn is not None and not check_fn(self.df[column]):
                issues.append(QualityIssue(
                    "schema", column, "warning",
                    f"Column '{column}' expected kind '{kind}' but has dtype '{self.df[column].dtype}'",
                ))

        return issues

    def _check_missing_values(self) -> list[QualityIssue]:
        issues = []
        not_null_cols = set(self.rules.get("not_null", []))
        warn_pct = self.rules.get("missing_warn_pct", DEFAULT_MISSING_WARN_PCT)
        n = len(self.df)

        for column in self.df.columns:
            missing = int(self.df[column].isna().sum())

            if missing == 0:
                continue

            pct = round(missing / n * 100, 2)

            if column in not_null_cols:
                issues.append(QualityIssue(
                    "not_null", column, "error",
                    f"Column '{column}' must not contain nulls", missing, pct,
                ))
            elif pct > warn_pct:
                issues.append(QualityIssue(
                    "missing_values", column, "warning",
                    f"Column '{column}' has {pct}% missing values", missing, pct,
                ))

        return issues

    def _check_duplicate_rows(self) -> list[QualityIssue]:
        duplicate_count = int(self.df.duplicated().sum())

        if duplicate_count == 0:
            return []

        pct = round(duplicate_count / len(self.df) * 100, 2)
        severity = "error" if self.rules.get("duplicates_allowed", True) is False else "warning"

        return [QualityIssue(
            "duplicate_rows", None, severity,
            f"{duplicate_count} duplicate row(s) found ({pct}%)", duplicate_count, pct,
        )]

    def _check_numeric_strings(self) -> list[QualityIssue]:
        issues = []

        for column in self.rules.get("numeric_strings", []):
            if column not in self.df.columns:
                continue

            series = self.df[column]
            coerced = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
            bad_mask = coerced.isna() & series.notna()
            bad_count = int(bad_mask.sum())

            if bad_count:
                pct = round(bad_count / len(series) * 100, 2)
                issues.append(QualityIssue(
                    "numeric_string", column, "error",
                    f"Column '{column}' should be numeric but has {bad_count} non-numeric value(s)",
                    bad_count, pct,
                ))

        return issues

    def _check_ranges(self) -> list[QualityIssue]:
        issues = []

        for column, bounds in self.rules.get("ranges", {}).items():
            if column not in self.df.columns:
                continue

            values = pd.to_numeric(self.df[column], errors="coerce")
            mask = pd.Series(False, index=values.index)

            if "min" in bounds:
                mask |= values < bounds["min"]
            if "max" in bounds:
                mask |= values > bounds["max"]

            count = int(mask.sum())

            if count:
                pct = round(count / len(values) * 100, 2)
                issues.append(QualityIssue(
                    "range", column, "error",
                    f"Column '{column}' has {count} value(s) outside expected range {bounds}",
                    count, pct,
                ))

        return issues

    def _check_categories(self) -> list[QualityIssue]:
        issues = []

        for column, allowed in self.rules.get("categories", {}).items():
            if column not in self.df.columns:
                continue

            series = self.df[column].dropna()
            mask = ~series.isin(allowed)
            count = int(mask.sum())

            if count:
                pct = round(count / len(series) * 100, 2) if len(series) else 0.0
                bad_values = sorted(series[mask].unique().tolist(), key=str)[:5]
                issues.append(QualityIssue(
                    "category", column, "warning",
                    f"Column '{column}' has {count} value(s) outside allowed set {allowed}: {bad_values}",
                    count, pct,
                ))

        return issues

    def _check_uniqueness(self) -> list[QualityIssue]:
        issues = []

        for column in self.rules.get("unique", []):
            if column not in self.df.columns:
                continue

            duplicate_count = int(self.df[column].duplicated().sum())

            if duplicate_count:
                pct = round(duplicate_count / len(self.df) * 100, 2)
                issues.append(QualityIssue(
                    "uniqueness", column, "error",
                    f"Column '{column}' expected to be unique but has {duplicate_count} duplicate value(s)",
                    duplicate_count, pct,
                ))

        return issues

    def _check_consistency(self) -> list[QualityIssue]:
        issues = []

        for rule in self.rules.get("consistency", []):
            if rule.get("type") != "product_match":
                continue

            a, b, target = rule.get("a"), rule.get("b"), rule.get("target")

            if not all(col in self.df.columns for col in (a, b, target)):
                continue

            tolerance = rule.get("tolerance_pct", 5) / 100
            left = pd.to_numeric(self.df[a], errors="coerce")
            right = pd.to_numeric(self.df[b], errors="coerce")
            actual = pd.to_numeric(self.df[target], errors="coerce")
            expected = left * right

            valid = actual.notna() & expected.notna() & (expected != 0)
            deviation = (actual - expected).abs() / expected
            mismatched = valid & (deviation > tolerance)
            count = int(mismatched.sum())

            if count:
                valid_count = int(valid.sum())
                pct = round(count / valid_count * 100, 2) if valid_count else 0.0
                issues.append(QualityIssue(
                    "consistency", target, "warning",
                    f"'{target}' deviates from '{a}' * '{b}' by more than "
                    f"{rule.get('tolerance_pct', 5)}% in {count} row(s)",
                    count, pct,
                ))

        return issues
