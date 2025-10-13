"""
Utility functions for property testing.
"""
import json
import os
from pathlib import Path


def append_jsonl(path, obj):
    """
    Append an object to a JSONL file.
    Creates parent directories if needed.

    Args:
        path: Path to the JSONL file
        obj: Object to serialize and append
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def load_jsonl(path):
    """
    Load all objects from a JSONL file.

    Args:
        path: Path to the JSONL file

    Returns:
        List of deserialized objects
    """
    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                objects.append(json.loads(line))
    return objects
