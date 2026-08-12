import json
from pathlib import Path


def load_json_config(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
