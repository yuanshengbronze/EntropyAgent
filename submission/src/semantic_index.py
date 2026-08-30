"""Shared catalog-concept helpers for offline and runtime semantic retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path


MAX_DOCUMENT_CONCEPTS = 16


def values(value: object) -> list[str]:
    """Normalize catalog values without changing their meaning."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    return [str(value)] if value != "" else []


def concepts_for_item(item: dict) -> list[str]:
    """Return the bounded concept representation used by SemRank scoring."""
    concepts: list[str] = []
    for category in values(item.get("category", item.get("categories"))):
        concepts.append(f"category: {category}")
    attributes = {
        "material": item.get("material", item.get("materials")),
        "color": item.get("color"),
        "size": item.get("size"),
        "style": item.get("style"),
        "brand": item.get("brand"),
        "budget": item.get("budget", item.get("budget_price")),
        "feature": item.get("feature"),
        "use_case": item.get("use_case"),
        "other": item.get("other"),
    }
    for attribute, raw_values in attributes.items():
        for value in values(raw_values):
            concepts.append(f"{attribute}: {value}")
    return list(dict.fromkeys(concepts))[:MAX_DOCUMENT_CONCEPTS]


def file_sha256(path: str | Path) -> str:
    """Content fingerprint used to prevent stale persisted indexes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
