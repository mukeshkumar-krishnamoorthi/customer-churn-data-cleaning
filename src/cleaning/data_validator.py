import pandas as pd


class DataValidator:

    def __init__(self, df, rules):
        if df is None or df.empty:
            raise ValueError(
                "DataValidator requires a non-empty DataFrame"
            )

        self.df = df
        self.rules = rules

        self.errors = []
        self.warnings = []
        self.results = []

    def validate_schema(self):
        schema = self.rules.get("schema", {})

        for column, expected_type in schema.items():

            if column not in self.df.columns:
                self.errors.append({
                    "rule": "schema",
                    "column": column,
                    "message": "Column is missing"
                })

                self.results.append({
                    "rule": "schema",
                    "column": column,
                    "status": "FAIL"
                })

                continue

            actual_type = self.df[column].dtype

            if not self._is_valid_dtype(
                self.df[column],
                expected_type
            ):
                self.errors.append({
                    "rule": "schema",
                    "column": column,
                    "expected": expected_type,
                    "actual": str(actual_type),
                    "message": "Invalid data type"
                })

                self.results.append({
                    "rule": "schema",
                    "column": column,
                    "status": "FAIL",
                    "expected": expected_type,
                    "actual": str(actual_type)
                })

            else:
                self.results.append({
                    "rule": "schema",
                    "column": column,
                    "status": "PASS",
                    "expected": expected_type,
                    "actual": str(actual_type)
                })

        return self

    def _is_valid_dtype(self, series, expected_type):

        if expected_type == "string":
            return (
                pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            )

        if expected_type == "integer":
            return pd.api.types.is_integer_dtype(series)

        if expected_type in ("float", "numeric"):
            return pd.api.types.is_numeric_dtype(series)

        if expected_type == "boolean":
            return pd.api.types.is_bool_dtype(series)

        if expected_type == "datetime":
            return pd.api.types.is_datetime64_any_dtype(series)

        return False
