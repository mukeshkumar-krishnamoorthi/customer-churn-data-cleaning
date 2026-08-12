import pandas as pd
import re


class DataCleaner:

    def __init__(self, df, rules):
        if df is None or df.empty:
            raise ValueError(
                "DataCleaner requires a non-empty DataFrame"
            )

        self.df = df.copy()
        self.rules = rules
        self.removed = []

    # 1. Normalize column names
    def normalize_column_names(self):
        config = self.rules.get("column_names", {})

        self.df.columns = [
            self._clean_string(column, config)
            for column in self.df.columns
        ]

        return self

    # 2. Clean string values
    def clean_string_values(self):
        config = self.rules.get("strings", {})

        for column in self.df.columns:

            if not (
                pd.api.types.is_object_dtype(self.df[column])
                or pd.api.types.is_string_dtype(self.df[column])
            ):
                continue

            self.df[column] = self._clean_string_series(
                self.df[column],
                config
            )

        return self

    # 3. Standardize categories
    def standardize_categories(self):
        category_mappings = self.rules.get(
            "category_mappings", {}
        )

        for column, mappings in category_mappings.items():

            if column not in self.df.columns:
                continue

            self.df[column] = self.df[column].replace(mappings)

            reverse_mapping = {}

            for standard_value, aliases in mappings.items():
                for alias in aliases:
                    reverse_mapping[alias] = standard_value

            self.df[column] = (
                self.df[column]
                .replace(reverse_mapping)
            )

        return self

    # 4. Convert data types
    def convert_data_types(self):
        data_types = self.rules.get("data_types", {})

        for column, dtype in data_types.items():

            if column not in self.df.columns:
                continue

            if dtype in ("integer", "int"):
                self.df[column] = pd.to_numeric(
                    self.df[column],
                    errors="coerce"
                ).astype("Int64")

            elif dtype in ("float", "numeric"):
                self.df[column] = pd.to_numeric(
                    self.df[column],
                    errors="coerce"
                )

            elif dtype == "string":
                self.df[column] = self.df[column].astype("string")

            elif dtype == "datetime":
                self.df[column] = pd.to_datetime(
                    self.df[column],
                    errors="coerce"
                )

            elif dtype == "boolean":
                self.df[column] = self.df[column].astype("boolean")

            else:
                self.df[column] = self.df[column].astype(dtype)

        return self

    # 5. Handle configured missing values
    def handle_missing_values(self):
        config = self.rules.get(
            "missing_values",
            {}
        )

        values = config.get(
            "replace_with_null",
            []
        )

        if not values:
            return self

        self.df = self.df.replace(
            values,
            pd.NA
        )

        return self

    # Helper for column names
    def _clean_string(self, value, config):
        value = str(value)

        if config.get("strip_whitespace", False):
            value = value.strip()

        if config.get("camelcase_to_underscore", False):
            value = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                "_",
                value
            )

        if config.get("spaces_to_underscore", False):
            value = re.sub(r"\s+", "_", value)

        if config.get("remove_special_characters", False):
            value = re.sub(r"[^a-zA-Z0-9_]", "", value)

        if config.get("remove_duplicate_underscores", False):
            value = re.sub(r"_+", "_", value)

        if config.get("lowercase", False):
            value = value.lower()

        if config.get("strip_underscores", False):
            value = value.strip("_")

        return value
        # Helper for DataFrame string values

    def _clean_string_series(self, series, config):
        series = series.astype("string")

        if config.get("strip_whitespace", False):
            series = series.str.strip()

        if config.get("collapse_whitespace", False):
            series = series.str.replace(
                r"\s+",
                " ",
                regex=True
            )

        if config.get("lowercase", False):
            series = series.str.lower()

        if config.get("empty_to_null", False):
            series = series.mask(
                series.str.strip().eq(""),
                pd.NA
            )

        return series

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

        return self.df

    def apply_business_rules(self):
        violations = []

        # Rule 1
        invalid_phone = self.df[
            (self.df["phone_service"] == "No") &
            (self.df["multiple_lines"] != "No phone service")
        ]

        if not invalid_phone.empty:
            violations.append(
                "phone_service='No' requires "
                "multiple_lines='No phone service'"
            )

        # Rule 2
        internet_dependent_columns = [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support"
        ]

        for column in internet_dependent_columns:
            invalid = self.df[
                (self.df["internet_service"] == "No") &
                (self.df[column] != "No internet service")
            ]

            if not invalid.empty:
                violations.append(
                    f"internet_service='No' requires "
                    f"{column}='No internet service'"
                )

        if violations:
            raise ValueError(
                "Business rule violations:\n"
                + "\n".join(violations)
            )

        print(f"Violations : {violations}")

        return self.df

    # Run complete cleaning pipeline
    def clean(self):

        self.normalize_column_names()

        self.clean_string_values()

        self.handle_missing_values()

        self.standardize_categories()

        self.convert_data_types()

        self.remove_duplicates()

        self.apply_business_rules()

        return self.df
