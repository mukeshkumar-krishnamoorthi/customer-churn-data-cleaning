from pathlib import Path
import json
import numpy as np
import pandas as pd

from utils.logger import Logger


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    return str(value)


class FileWriter:
    """
    Generic file writer for exporting data.
    """

    @staticmethod
    def csv(df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)

        Logger.info(
            "CSV Exported",
            File=path.name,
            Rows=df.shape[0],
            Columns=df.shape[1],
        )

    @staticmethod
    def json(data: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=_json_default)

        Logger.info(
            "JSON Exported",
            File=path.name,
        )

    @staticmethod
    def excel(df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_excel(path, index=False)

        Logger.info(
            "Excel Exported",
            File=path.name,
        )

    @staticmethod
    def text(text: str, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(text, encoding="utf-8")

        Logger.info(
            "Text File Exported",
            File=path.name,
        )
