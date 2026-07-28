from pathlib import Path

import pandas as pd

from utils.logger import Logger


class CSVLoader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a CSV file, got: {self.file_path.suffix}")

        Logger.info(f"Loading '{self.file_path.name}'...")

        df = pd.read_csv(self.file_path)

        Logger.info(
            "CSV Loaded",
            File=self.file_path.name,
            Rows=len(df),
            Columns=len(df.columns),
        )

        return df
