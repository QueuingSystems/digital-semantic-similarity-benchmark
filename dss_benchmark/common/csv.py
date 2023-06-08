import csv
import os
from typing import Any, Dict

__all__ = ["append_to_csv"]


def append_to_csv(path: str, contents: Dict[str, Any]):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=contents.keys())
            writer.writeheader()
    with open(path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=contents.keys())
        writer.writerow(contents)
