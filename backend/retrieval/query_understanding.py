
from __future__ import annotations

from typing import Any


SUPPORTED_FILTERS = (
    "brand",
    "color",
    "material",
    "category",
    "product_type",
)


def understand_query(
    query: str,
    explicit_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prepare a product query without hardcoded attribute vocabularies.

    Natural-language attributes remain part of the search query so the
    semantic and lexical retrievers can use the complete user intent.
    Explicit filters supplied by the user remain strict constraints.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()

    if not query:
        return {
            "original_query": query,
            "search_query": query,
            "intent": "product_search",
            "filters": {},
        }

    if explicit_filters is None:
        explicit_filters = {}

    if not isinstance(explicit_filters, dict):
        raise TypeError("explicit_filters must be a dictionary")

    filters: dict[str, str] = {}

    for field in SUPPORTED_FILTERS:
        value = explicit_filters.get(field)

        if value is None:
            continue

        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Filter '{field}' must be a string or number"
            )

        normalized = str(value).strip()

        if normalized:
            filters[field] = normalized

    return {
        "original_query": query,
        "search_query": query,
        "intent": "product_search",
        "filters": filters,
    }
