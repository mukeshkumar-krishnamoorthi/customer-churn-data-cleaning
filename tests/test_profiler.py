import pandas as pd
import pytest

from profiling.profiler import DataProfiler


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 30, 30, None, 200],
        "gender": ["Male", "Female", "Female", "Male", "Male"],
        "flag": [True, False, True, True, False],
    })


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        DataProfiler(pd.DataFrame())


def test_dataset_summary_counts(df):
    summary = DataProfiler(df).dataset_summary()

    assert summary["rows"] == 5
    assert summary["columns"] == 4
    assert summary["missing_cells"] == 1
    assert summary["duplicate_rows"] == 0


def test_numeric_column_profile_flags_outlier(df):
    profile = DataProfiler(df).profile_column("age")

    assert profile["missing"] == 1
    assert profile["min"] == 25.0
    assert profile["max"] == 200.0
    assert profile["outliers_iqr"] >= 1


def test_categorical_column_profile(df):
    profile = DataProfiler(df).profile_column("gender")

    assert profile["top_value"] in {"Male", "Female"}
    assert profile["top_values"]["Male"] == 3


def test_boolean_column_profile(df):
    profile = DataProfiler(df).profile_column("flag")

    assert profile["true_count"] == 3
    assert profile["false_count"] == 2


def test_profile_column_unknown_column_raises(df):
    with pytest.raises(KeyError):
        DataProfiler(df).profile_column("does_not_exist")


def test_column_summary_shape(df):
    summary = DataProfiler(df).column_summary()

    assert list(summary["Column"]) == list(df.columns)
    assert len(summary) == df.shape[1]


def test_correlation_matrix_none_when_single_numeric_column():
    single_numeric = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    assert DataProfiler(single_numeric).correlation_matrix() is None


def test_to_report_is_json_shape(df):
    report = DataProfiler(df).to_report()

    assert set(report.keys()) == {"dataset", "columns", "correlation"}
    assert set(report["columns"].keys()) == set(df.columns)
