import pandas as pd
from utils.logger import Logger


class DataProfiler:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile_dataset(self):

        Logger.info(
            "Dataset Profile",
            Rows=self.df.shape[0],
            Columns=self.df.shape[1],
            Duplicate_Rows=self.df.duplicated().sum(),
            Missing_Values=self.df.isna().sum().sum(),
            Memory=f"{self.df.memory_usage(deep=True).sum()/1024:.2f} KB"
        )

    def column_summary(self) -> pd.DataFrame:

        summary = pd.DataFrame({
            "Column": self.df.columns,
            "Type": self.df.dtypes.astype(str).values,
            "Missing": self.df.isna().sum().values,
            "Duplicates": self.df.duplicated().sum(),
            "Unique": self.df.nunique().values,
            "Min":self.df.min(),
            "Max":self.df.max(),
            "Memory":f"{self.df.memory_usage(deep=True).sum()/1024:.2f} KB",
        })

        return summary

    def save_as_csv(self, summary: pd.DataFrame) -> None:
        summary.to_csv(
            "outputs/reports/column_summary.csv",
            index=False,
        )

    def profile_columns(self):

        report = []

        for column in self.df.columns:

            series = self.df[column]

            report.append("\n" + "-" * 60)
            report.append(f"Column : {column}")
            report.append("-" * 60)

            report.append(f"Datatype         : {series.dtype}")
            report.append(f"Missing Values   : {series.isna().sum()}")
            report.append(f"Unique Values    : {series.nunique()}")
            report.append(f"Duplicate Values : {series.duplicated().sum()}")
            report.append(
                f"Memory Usage     : {series.memory_usage(deep=True)/1024:.2f} KB")

            if pd.api.types.is_numeric_dtype(series):

                report.append(f"Minimum          : {series.min()}")
                report.append(f"Maximum          : {series.max()}")
                report.append(f"Average          : {series.mean():.2f}")
                report.append(f"Median           : {series.median():.2f}")
                report.append(f"Mode             : {series.mode().iloc[0]}")

        return "\n".join(report)
