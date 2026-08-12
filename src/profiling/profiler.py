import pandas as pd

from utils.logger import Logger

CATEGORICAL_TOP_N = 5
IQR_MULTIPLIER = 1.5


class DataProfiler:
    """Computes dataset- and column-level statistics for data-quality review."""

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DataProfiler requires a non-empty DataFrame")

        self.df = df

    def dataset_summary(self) -> dict:
        n_rows, n_cols = self.df.shape
        n_cells = n_rows * n_cols

        missing_cells = int(self.df.isna().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum())

        return {
            "rows": n_rows,
            "columns": n_cols,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0.0,
            "missing_cells": missing_cells,
            "missing_cells_pct": round(missing_cells / n_cells * 100, 2) if n_cells else 0.0,
            "memory_kb": round(self.df.memory_usage(deep=True).sum() / 1024, 2),
            "dtypes": {str(k): int(v) for k, v in self.df.dtypes.astype(str).value_counts().items()},
        }

    def profile_column(self, column: str) -> dict:
        if column not in self.df.columns:
            raise KeyError(f"Column not found: {column}")

        series = self.df[column]
        n = len(series)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        profile = {
            "column": column,
            "dtype": str(series.dtype),
            "count": n,
            "missing": missing,
            "missing_pct": round(missing / n * 100, 2) if n else 0.0,
            "unique": unique,
            "unique_pct": round(unique / n * 100, 2) if n else 0.0,
            "memory_kb": round(series.memory_usage(deep=True) / 1024, 2),
        }

        non_null = series.dropna()

        if pd.api.types.is_bool_dtype(series):
            profile.update(self._profile_boolean(non_null))
        elif pd.api.types.is_numeric_dtype(series):
            profile.update(self._profile_numeric(non_null))
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._profile_datetime(non_null))
        else:
            profile.update(self._profile_categorical(non_null))

        return profile

    def profile_columns(self) -> dict:
        return {column: self.profile_column(column) for column in self.df.columns}

    def column_summary(self) -> pd.DataFrame:
        rows = []

        for column, profile in self.profile_columns().items():
            rows.append({
                "Column": column,
                "Type": profile["dtype"],
                "Missing": profile["missing"],
                "Missing %": profile["missing_pct"],
                "Unique": profile["unique"],
                "Unique %": profile["unique_pct"],
                "Min": profile.get("min", profile.get("top_value", "")),
                "Max": profile.get("max", ""),
                "Mean": profile.get("mean", ""),
                "Memory (KB)": profile["memory_kb"],
            })

        return pd.DataFrame(rows)

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame | None:
        numeric_df = self.df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            return None

        return numeric_df.corr(method=method)

    def to_report(self) -> dict:
        correlation = self.correlation_matrix()

        return {
            "dataset": self.dataset_summary(),
            "columns": self.profile_columns(),
            "correlation": correlation.round(3).to_dict() if correlation is not None else None,
        }

    def print_report(self) -> None:
        summary = self.dataset_summary()
        Logger.info("Dataset Profile", **{k: v for k, v in summary.items() if k != "dtypes"})

        for column, profile in self.profile_columns().items():
            Logger.info(f"Column: {column}", **{k: v for k, v in profile.items() if k != "column"})

    def _profile_numeric(self, series: pd.Series) -> dict:
        if series.empty:
            return {}

        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 2),
            "median": float(median),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "q1": float(q1),
            "q3": float(q3),
            "zeros": int((series == 0).sum()),
            "negatives": int((series < 0).sum()),
            "outliers_iqr": int(len(outliers)),
            "outliers_iqr_pct": round(len(outliers) / len(series) * 100, 2),
        }

    def _profile_categorical(self, series: pd.Series) -> dict:
        if series.empty:
            return {}

        value_counts = series.value_counts()
        as_text = series.astype(str).str.strip()
        looks_numeric = pd.to_numeric(as_text, errors="coerce")

        return {
            "top_value": value_counts.index[0],
            "top_freq": int(value_counts.iloc[0]),
            "top_freq_pct": round(value_counts.iloc[0] / len(series) * 100, 2),
            "blank_strings": int((as_text == "").sum()),
            "looks_numeric_pct": round(looks_numeric.notna().sum() / len(series) * 100, 2),
            "top_values": {str(k): int(v) for k, v in value_counts.head(CATEGORICAL_TOP_N).items()},
        }

    def _profile_boolean(self, series: pd.Series) -> dict:
        if series.empty:
            return {}

        true_count = int(series.sum())

        return {
            "true_count": true_count,
            "false_count": int(len(series) - true_count),
            "true_pct": round(true_count / len(series) * 100, 2),
        }

    def _profile_datetime(self, series: pd.Series) -> dict:
        if series.empty:
            return {}

        return {
            "min": str(series.min()),
            "max": str(series.max()),
            "range_days": (series.max() - series.min()).days,
        }
