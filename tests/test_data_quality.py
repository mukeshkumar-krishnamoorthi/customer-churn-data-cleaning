import pandas as pd
import pytest

from validation.data_quality import DataQuality


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "tenure": [1, 34, 2],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", "1936.3", "107.7"],
        "Churn": ["No", "No", "Yes"],
    })


@pytest.fixture
def rules():
    return {
        "schema": {
            "gender": "string",
            "tenure": "numeric",
            "MonthlyCharges": "numeric",
            "TotalCharges": "string",
        },
        "not_null": ["gender", "tenure", "MonthlyCharges", "TotalCharges"],
        "numeric_strings": ["TotalCharges"],
        "ranges": {"MonthlyCharges": {"min": 0}, "tenure": {"min": 0}},
        "categories": {"gender": ["Male", "Female"], "Churn": ["Yes", "No"]},
        "unique": [],
        "consistency": [
            {"type": "product_match", "a": "tenure", "b": "MonthlyCharges", "target": "TotalCharges", "tolerance_pct": 15},
        ],
    }


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        DataQuality(pd.DataFrame())


def test_clean_dataframe_scores_perfectly(clean_df, rules):
    report = DataQuality(clean_df, rules).run()

    assert report.error_count == 0
    assert report.score == 100.0


def test_not_null_violation_is_error(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "gender"] = None

    report = DataQuality(df, rules).run()

    checks = {issue.check for issue in report.issues}
    assert "not_null" in checks
    assert report.error_count >= 1
    assert report.score < 100.0


def test_numeric_string_blank_flagged_as_error(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "TotalCharges"] = " "

    report = DataQuality(df, rules).run()

    numeric_string_issues = [i for i in report.issues if i.check == "numeric_string"]
    assert len(numeric_string_issues) == 1
    assert numeric_string_issues[0].column == "TotalCharges"
    assert numeric_string_issues[0].severity == "error"


def test_range_violation_flagged(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "MonthlyCharges"] = -5

    report = DataQuality(df, rules).run()

    range_issues = [i for i in report.issues if i.check == "range"]
    assert len(range_issues) == 1
    assert range_issues[0].column == "MonthlyCharges"


def test_category_violation_flagged(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "gender"] = "Other"

    report = DataQuality(df, rules).run()

    category_issues = [i for i in report.issues if i.check == "category"]
    assert len(category_issues) == 1


def test_uniqueness_violation_flagged(clean_df, rules):
    rules["unique"] = ["gender"]
    df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)

    report = DataQuality(df, rules).run()

    uniqueness_issues = [i for i in report.issues if i.check == "uniqueness"]
    assert len(uniqueness_issues) == 1


def test_consistency_violation_flagged(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "TotalCharges"] = "9999"

    report = DataQuality(df, rules).run()

    consistency_issues = [i for i in report.issues if i.check == "consistency"]
    assert len(consistency_issues) == 1


def test_missing_schema_column_is_error(clean_df, rules):
    rules["schema"]["customerID"] = "string"

    report = DataQuality(clean_df, rules).run()

    schema_issues = [i for i in report.issues if i.check == "schema" and i.column == "customerID"]
    assert len(schema_issues) == 1
    assert schema_issues[0].severity == "error"


def test_report_to_dataframe_matches_issue_count(clean_df, rules):
    df = clean_df.copy()
    df.loc[0, "gender"] = "Other"

    report = DataQuality(df, rules).run()

    assert len(report.to_dataframe()) == len(report.issues)


def test_no_rules_yields_only_generic_checks(clean_df):
    report = DataQuality(clean_df, rules=None).run()

    assert report.error_count == 0
