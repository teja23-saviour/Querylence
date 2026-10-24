from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DIVERSITY_THRESHOLD = 0.88
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256

TAXONOMY_WEIGHT = 0.50
TITLE_WEIGHT = 0.25
DETAIL_WEIGHT = 0.25

MODEL_SCORE_WEIGHT = 0.70
IDENTITY_COVERAGE_WEIGHT = 0.20
CATALOG_COVERAGE_WEIGHT = 0.10
ITEM_TYPE_ALIGNMENT_WEIGHT = 2.0
ATTRIBUTE_ALIGNMENT_WEIGHT = 1.25
TITLE_ATTRIBUTE_BONUS_WEIGHT = 0.25

_QUERY_PREPOSITIONS = {
    "for", "with", "without", "in", "on", "at", "under",
    "over", "near", "from", "to", "by", "of", "about",
    "during", "and",
}

_QUERY_FILLERS = {
    "a", "an", "the", "some", "please", "find", "looking",
    "need", "want", "buy", "show", "get", "search", "best",
    "good", "cheap", "me", "i", "am", "can", "you", "use",
    "my", "is", "are", "s",
}

RELATED_PRODUCT_HEADS = {
    "organizer", "organiser", "stacker", "rack", "hanger",
    "cover", "case", "bag", "holder", "storage", "protector",
    "carrier", "cleaner", "polish", "stand", "shelf", "caddy",
    "accessory", "accessories", "replacement", "attachment",
    "mount", "bracket", "container", "sleeve", "box", "bin",
    "pouch", "mat", "hook", "clip", "insert", "liner",
    "dispenser", "display", "tray", "tote", "drawer",
    "cabinet", "wrapping",
}

ATTRIBUTE_BLOCKING_REASONS = {
    "no_item_type_evidence",
    "collection_or_prefix_only",
    "related_product_not_requested_item",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _normalize_token(token: str) -> str:
    """Normalize common English plural endings and simple -ing forms."""
    token = token.casefold()

    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]

        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def _normalized_tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in _tokens(text)
    }


def _query_analysis(query: str) -> tuple[str, list[str], list[str]]:
    """
    Infer the item type and remaining query attributes.

    Examples:
        blue cotton shirt
            -> shirt; blue, cotton

        comfortable running shoes
            -> running shoe; comfortable

        gardening tools for outdoor use
            -> gardening tool; outdoor

        lightweight backpack for travel
            -> backpack; lightweight, travel
    """
    raw_tokens = _tokens(query)
    intent_tokens: list[str] = []

    for token in raw_tokens:
        normalized = _normalize_token(token)

        if normalized in _QUERY_PREPOSITIONS:
            break

        if normalized not in _QUERY_FILLERS:
            intent_tokens.append(token)

    if not intent_tokens:
        return "", [], []

    # Preserve compound item types such as running shoes and gardening tools.
    if (
        len(intent_tokens) >= 2
        and intent_tokens[-2].endswith("ing")
    ):
        item_type_terms = [
            _normalize_token(intent_tokens[-2]),
            _normalize_token(intent_tokens[-1]),
        ]

        item_type = (
            f"{intent_tokens[-2]} "
            f"{_normalize_token(intent_tokens[-1])}"
        )
    else:
        item_type_terms = [
            _normalize_token(intent_tokens[-1])
        ]

        item_type = item_type_terms[0]

    type_term_set = set(item_type_terms)
    attributes: list[str] = []

    for token in raw_tokens:
        normalized = _normalize_token(token)

        if (
            normalized in _QUERY_PREPOSITIONS
            or normalized in _QUERY_FILLERS
            or normalized in type_term_set
        ):
            continue

        if normalized and normalized not in attributes:
            attributes.append(normalized)

    return item_type, item_type_terms, attributes


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    if not terms:
        return False

    available = _normalized_tokens(text)
    return all(term in available for term in terms)


def _primary_title(title: str) -> str:
    """Remove misleading Amazon collection prefixes where possible."""
    cleaned = title.strip()

    if " - " not in cleaned:
        return cleaned

    prefix, suffix = cleaned.split(" - ", 1)
    prefix_lower = prefix.casefold().strip()

    if prefix_lower in {"amazon brand", "amazon basics brand"}:
        return suffix.strip() or cleaned

    if prefix_lower.startswith("amazon") and "collection" in prefix_lower:
        return suffix.strip() or cleaned

    return cleaned


def _is_related_item_mention(
    text: str,
    item_type_terms: list[str],
) -> bool:
    """
    Detect an accessory referring to the requested item.

    Inspect only the immediately following word. A garden tool set
    remains a tool set even when its title later mentions a tote bag.
    """
    words = [
        _normalize_token(token)
        for token in _tokens(text)
    ]

    width = len(item_type_terms)

    if not width or len(words) < width:
        return False

    for index in range(len(words) - width + 1):
        if words[index:index + width] != item_type_terms:
            continue

        next_index = index + width

        if next_index < len(words):
            if words[next_index] in RELATED_PRODUCT_HEADS:
                return True

    return False


class CrossEncoderReranker:
    """Rerank products using item identity, attributes, and model scores."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Any | None = None,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if (
            isinstance(diversity_threshold, bool)
            or not isinstance(diversity_threshold, (int, float))
            or not 0.0 <= diversity_threshold <= 1.0
        ):
            raise ValueError(
                "diversity_threshold must be between 0.0 and 1.0"
            )

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")

        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")

        self.model_name = model_name.strip()
        self._model = model
        self.diversity_threshold = float(diversity_threshold)
        self.batch_size = batch_size
        self.max_length = max_length

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )

        return self._model

    def warmup(self) -> None:
        self._get_model().predict(
            [("blue cotton shirt", "Blue cotton T-shirt")],
            batch_size=1,
            show_progress_bar=False,
        )

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, dict):
            return " ".join(
                str(item)
                for item in value.values()
                if item is not None
            )

        if isinstance(value, (list, tuple, set)):
            return " ".join(
                str(item)
                for item in value
                if item is not None
            )

        return str(value).strip()

    @classmethod
    def _title(cls, product: dict[str, Any]) -> str:
        return cls._as_text(product.get("title")) or "Untitled product"

    @classmethod
    def _taxonomy_text(cls, product: dict[str, Any]) -> str:
        parts = []

        for field in ("category", "category_paths"):
            value = cls._as_text(product.get(field))

            if value:
                parts.append(f"{field}: {value}")

        return " | ".join(parts) or "unknown catalog category"

    @classmethod
    def _category_path_text(cls, product: dict[str, Any]) -> str:
        return cls._as_text(product.get("category_paths"))

    @classmethod
    def _product_text(cls, product: dict[str, Any]) -> str:
        """Construct bounded product evidence for cross-encoder scoring."""
        parts = [
            f"title: {_primary_title(cls._title(product))}"
        ]

        for field in (
            "category",
            "category_paths",
            "brand",
            "color",
            "material",
            "fabric_type",
            "product_type",
            "keywords",
            "description",
        ):
            value = cls._as_text(product.get(field))

            if not value:
                continue

            if field == "keywords":
                value = value[:250]
            elif field == "description":
                value = value[:350]

            parts.append(f"{field}: {value}")

        return " | ".join(parts)[:1800]

    @classmethod
    def _attribute_evidence(
        cls,
        product: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Return title evidence and product metadata separately.

        Category paths are excluded from direct color/material evidence.
        """
        title = _primary_title(cls._title(product))

        fields = (
            "brand",
            "color",
            "material",
            "fabric_type",
            "product_type",
            "keywords",
            "description",
        )

        metadata_parts = []

        for field in fields:
            value = cls._as_text(product.get(field))

            if not value:
                continue

            if field == "keywords":
                value = value[:250]
            elif field == "description":
                value = value[:350]

            metadata_parts.append(value)

        return title, " ".join(metadata_parts)

    @staticmethod
    def _standardize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        deviation = float(np.std(scores))

        if deviation < 1e-8:
            return np.zeros_like(scores)

        return (scores - float(np.mean(scores))) / deviation

    @staticmethod
    def _coverage(query: str, text: str) -> float:
        query_terms = {
            _normalize_token(token)
            for token in _tokens(query)
            if _normalize_token(token) not in _QUERY_FILLERS
            and _normalize_token(token) not in _QUERY_PREPOSITIONS
        }

        text_terms = _normalized_tokens(text)

        if not query_terms:
            return 0.0

        return len(query_terms & text_terms) / len(query_terms)

    @staticmethod
    def _attribute_matches(
        attributes: list[str],
        text: str,
    ) -> list[str]:
        available = _normalized_tokens(text)

        return [
            attribute
            for attribute in attributes
            if attribute in available
        ]

    @staticmethod
    def _item_type_alignment(
        primary_title: str,
        full_title: str,
        category_paths: str,
        item_type_terms: list[str],
    ) -> tuple[float, str]:
        if not item_type_terms:
            return 0.0, "no_item_type_in_query"

        if _is_related_item_mention(primary_title, item_type_terms):
            return -1.50, "related_product_not_requested_item"

        if _contains_all_terms(primary_title, item_type_terms):
            return 1.0, "primary_title_match"

        if _contains_all_terms(full_title, item_type_terms):
            return -2.0, "collection_or_prefix_only"

        if _contains_all_terms(category_paths, item_type_terms):
            return 0.25, "category_path_match"

        return -2.0, "no_item_type_evidence"

    @staticmethod
    def _normalize_title(title: Any) -> str:
        if not isinstance(title, str):
            return ""

        return re.sub(
            r"[^a-z0-9]+",
            " ",
            title.casefold(),
        ).strip()

    def _is_near_duplicate(
        self,
        title: Any,
        selected_titles: list[str],
    ) -> bool:
        normalized = self._normalize_title(title)

        if not normalized:
            return False

        return any(
            SequenceMatcher(None, normalized, selected).ratio()
            >= self.diversity_threshold
            for selected in selected_titles
        )

    def _select_diverse(
        self,
        ranked: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        selected_titles: list[str] = []

        for product in ranked:
            title = product.get("title", "")

            if self._is_near_duplicate(title, selected_titles):
                deferred.append(product)
                continue

            selected.append(product)
            normalized = self._normalize_title(title)

            if normalized:
                selected_titles.append(normalized)

            if len(selected) >= top_k:
                return selected

        selected.extend(deferred[:max(0, top_k - len(selected))])
        return selected

    def rerank(
        self,
        query: str,
        products: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query or not products:
            return []

        if top_k is not None and (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")

        candidates = [
            product
            for product in products
            if isinstance(product, dict)
        ]

        if not candidates:
            return []

        title_texts = [
            self._title(product)
            for product in candidates
        ]

        primary_title_texts = [
            _primary_title(title)
            for title in title_texts
        ]

        taxonomy_texts = [
            self._taxonomy_text(product)
            for product in candidates
        ]

        category_path_texts = [
            self._category_path_text(product)
            for product in candidates
        ]

        detail_texts = [
            self._product_text(product)
            for product in candidates
        ]

        item_type, item_type_terms, query_attributes = (
            _query_analysis(query)
        )

        model = self._get_model()

        def predict(pairs: list[tuple[str, str]]) -> np.ndarray:
            scores = np.asarray(
                model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ),
                dtype=np.float64,
            ).reshape(-1)

            if len(scores) != len(candidates):
                raise RuntimeError(
                    "Cross-encoder returned an unexpected number of scores"
                )

            if not np.isfinite(scores).all():
                raise RuntimeError(
                    "Cross-encoder returned non-finite relevance scores"
                )

            return scores

        title_scores = predict([
            (query, title)
            for title in primary_title_texts
        ])

        taxonomy_scores = predict([
            (query, taxonomy)
            for taxonomy in taxonomy_texts
        ])

        detail_scores = predict([
            (query, details)
            for details in detail_texts
        ])

        model_scores = (
            TAXONOMY_WEIGHT * self._standardize(taxonomy_scores)
            + TITLE_WEIGHT * self._standardize(title_scores)
            + DETAIL_WEIGHT * self._standardize(detail_scores)
        )

        ranked: list[dict[str, Any]] = []

        for index, product in enumerate(candidates):
            result = dict(product)

            title_evidence, metadata_evidence = (
                self._attribute_evidence(product)
            )

            matched_title_attributes = self._attribute_matches(
                query_attributes,
                title_evidence,
            )

            matched_catalog_attributes = self._attribute_matches(
                query_attributes,
                metadata_evidence,
            )

            matched_attributes = list(dict.fromkeys(
                matched_title_attributes + matched_catalog_attributes
            ))

            missing_attributes = [
                attribute
                for attribute in query_attributes
                if attribute not in matched_attributes
            ]

            if query_attributes:
                attribute_coverage = (
                    len(matched_attributes) / len(query_attributes)
                )

                title_attribute_coverage = (
                    len(matched_title_attributes) / len(query_attributes)
                )
            else:
                attribute_coverage = 0.0
                title_attribute_coverage = 0.0

            identity_coverage = self._coverage(
                query,
                f"{primary_title_texts[index]} "
                f"{category_path_texts[index]}",
            )

            catalog_coverage = self._coverage(
                query,
                detail_texts[index],
            )

            item_type_alignment, item_type_reason = (
                self._item_type_alignment(
                    primary_title=primary_title_texts[index],
                    full_title=title_texts[index],
                    category_paths=category_path_texts[index],
                    item_type_terms=item_type_terms,
                )
            )

            # Critical fix: matching color/material keywords cannot rescue
            # a candidate whose actual product type does not match.
            attribute_bonus_allowed = (
                not item_type_terms
                or item_type_reason not in ATTRIBUTE_BLOCKING_REASONS
            )

            effective_attribute_coverage = (
                attribute_coverage if attribute_bonus_allowed else 0.0
            )

            effective_title_attribute_coverage = (
                title_attribute_coverage
                if attribute_bonus_allowed else 0.0
            )

            final_score = (
                MODEL_SCORE_WEIGHT * float(model_scores[index])
                + IDENTITY_COVERAGE_WEIGHT * identity_coverage
                + CATALOG_COVERAGE_WEIGHT * catalog_coverage
                + ITEM_TYPE_ALIGNMENT_WEIGHT * item_type_alignment
                + ATTRIBUTE_ALIGNMENT_WEIGHT * effective_attribute_coverage
                + TITLE_ATTRIBUTE_BONUS_WEIGHT
                * effective_title_attribute_coverage
            )

            # Keep RRF as the public score. The reranker sorts using
            # final_relevance_score instead.
            original_score = result.get(
                "rrf_score",
                result.get("score"),
            )

            if original_score is not None:
                result["score"] = float(original_score)
                result["rrf_score"] = float(original_score)
            else:
                result["score"] = final_score

            result["title_relevance_score"] = float(title_scores[index])
            result["product_type_score"] = float(taxonomy_scores[index])
            result["detail_relevance_score"] = float(detail_scores[index])
            result["reranker_score"] = float(detail_scores[index])

            result["identity_coverage"] = identity_coverage
            result["catalog_coverage"] = catalog_coverage

            result["query_item_type"] = item_type
            result["query_item_type_terms"] = item_type_terms
            result["query_attributes"] = query_attributes
            result["matched_query_attributes"] = matched_attributes
            result["matched_title_attributes"] = matched_title_attributes
            result["matched_catalog_attributes"] = matched_catalog_attributes
            result["missing_query_attributes"] = missing_attributes
            result["attribute_coverage"] = attribute_coverage
            result["title_attribute_coverage"] = title_attribute_coverage

            result["effective_attribute_coverage"] = (
                effective_attribute_coverage
            )

            result["item_type_alignment"] = item_type_alignment
            result["item_type_reason"] = item_type_reason
            result["primary_product_title"] = primary_title_texts[index]

            result["final_relevance_score"] = final_score
            result["retrieval_method"] = "hybrid_reranked"

            ranked.append(result)

        ranked.sort(
            key=lambda item: item["final_relevance_score"],
            reverse=True,
        )

        if top_k is None:
            return ranked

        return self._select_diverse(ranked, top_k)