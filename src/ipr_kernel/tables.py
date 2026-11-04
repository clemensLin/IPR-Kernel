"""Read and write the balanced-accuracy tables that the figure script and the manuscript use."""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPO_ROOT

TABLES_DIR = REPO_ROOT / "figures" / "tables"


METRICS = ("global_bal_acc_mean", "global_bal_acc_std", "local_bal_acc_mean", "local_bal_acc_std")


def svm_rows(params, results: dict, digits: int = 4) -> list[list[float]]:
    """Rows of (parameter, global mean, global std, local mean, local std), rounded as in the manuscript.

    `params` holds the swept values (system sizes or inverse temperatures) and `results` the matching
    balanced-accuracy lists returned by the SVM experiments.
    """
    return [[p, *(round(float(results[m][i]), digits) for m in METRICS)] for i, p in enumerate(params)]


def write_table(path: Path, tables: dict, source: str) -> Path:
    """Write a JSON table with one row per line, so diffs between runs stay readable."""

    def dump(value, indent: str) -> str:
        if isinstance(value, dict):
            inner = ",\n".join(f'{indent}  "{k}": {dump(v, indent + "  ")}' for k, v in value.items())
            return "{\n" + inner + "\n" + indent + "}"
        if isinstance(value, list) and value and isinstance(value[0], list):
            rows = (",\n" + indent + " ").join(json.dumps(row) for row in value)
            return "[" + rows + "]"
        return json.dumps(value)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump({"_source": source, **tables}, "") + "\n", encoding="utf-8")
    return path
