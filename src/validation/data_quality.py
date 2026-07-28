import pandas as pd


class DataQuality:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self):
        report = {
            "Missing Values": self.check_missing_values(),
            "Duplicate Rows": self.check_duplicate_rows(),
            "Data Types": self.check_data_types(),
            "Invalid Numeric Values": self.check_numeric_values(),
            "Invalid Categories": self.check_categories(),
        }

        return report

    def check_missing_values(self):
        return self.df.isna().sum()

    def check_duplicate_rows(self):
        return self.df.duplicated().sum()

    def check_data_types(self):
        errors = {}

        EXPECTED_SCHEMA = {
            "gender": "object",
            "SeniorCitizen": "int64",
            "tenure": "int64",
            "MonthlyCharges": "float64",
            "TotalCharges": "float64",
        }

        for column, dtype in EXPECTED_SCHEMA.items():

            actual = str(self.df[column].dtype)

            if actual != dtype:
                errors[column] = {
                    "Expected": dtype,
                    "Actual": actual
                }

        return errors

    def check_numeric_values(self):

        issues = {}

        if (self.df["MonthlyCharges"] < 0).any():
            issues["MonthlyCharges"] = "Negative values found"

        if (self.df["tenure"] < 0).any():
            issues["tenure"] = "Negative values found"

        return issues

    def check_categories(self):

        issues = {}

        valid_gender = {"Male", "Female"}

        invalid = self.df[
            ~self.df["gender"].isin(valid_gender)
        ]

        if len(invalid):

            issues["gender"] = len(invalid)

        return issues
