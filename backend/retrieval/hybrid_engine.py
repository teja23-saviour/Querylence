from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any


class HybridSearchEngine:
    """Hybrid BM25 + semantic retrieval with RRF and optional reranking."""

    # Only fill metadata that is missing. Never replace an existing
    # non-empty title, product ID, color, or material with a conflicting
    # value from another retrieval engine.
    MERGEABLE_LIST_FIELDS = {
        "category_paths",
        "keywords",
        "other_image_ids",
    }

    def __init__(
        self,
        bm25_engine,
        semantic_engine,
        rrf_k: int = 60,
        reranker=None,
    ) -> None:
        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
            or rrf_k < 1
        ):
            raise ValueError("rrf_k must be a positive integer")

        self.bm25_engine = bm25_engine
        self.semantic_engine = semantic_engine
        self.rrf_k = rrf_k
        self.reranker = reranker

        self.rerank_candidate_limit = self._positive_env_int(
            "QUERYLENCE_RERANK_CANDIDATES", 20
        )
        self.recovery_multiplier = self._positive_env_int(
            "QUERYLENCE_RECOVERY_MULTIPLIER", 2
        )

    @staticmethod
    def _positive_env_int(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
            return value if value > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _enabled(name: str) -> bool:
        return os.getenv(name, "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        if isinstance(value, str):
            return not value.strip()

        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0

        return False

    @classmethod
    def _merge_product_records(
        cls,
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enrich a product record using another record with the same ID.

        Existing non-empty scalar values take precedence. This avoids
        blindly replacing a product's title or attributes. Selected lists
        are combined while preserving their original order.
        """
        merged = dict(existing)

        for field, incoming_value in incoming.items():
            if field == "product_id":
                continue

            existing_value = merged.get(field)

            if cls._is_missing(existing_value):
                if not cls._is_missing(incoming_value):
                    merged[field] = incoming_value
                continue

            if (
                field in cls.MERGEABLE_LIST_FIELDS
                and isinstance(existing_value, list)
                and isinstance(incoming_value, list)
            ):
                combined = list(existing_value)

                for value in incoming_value:
                    if value not in combined:
                        combined.append(value)

                merged[field] = combined

        return merged

    @staticmethod
    def _product_ids(results: list[dict[str, Any]]) -> set[Any]:
        return {
            product["product_id"]
            for product in results
            if isinstance(product, dict)
            and product.get("product_id") is not None
        }

    def _retrieve_candidates(
        self,
        query: str,
        candidate_count: int,
        filters: dict[str, Any],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve candidates and fuse rankings with Reciprocal Rank Fusion."""
        lexical = self.bm25_engine.search(
            query=query,
            top_k=candidate_count,
            filters=filters,
        ) or []

        semantic = self.semantic_engine.search(
            query=query,
            top_k=candidate_count,
            filters=filters,
        ) or []

        scores: dict[Any, float] = defaultdict(float)
        products: dict[Any, dict[str, Any]] = {}
        components: dict[Any, dict[str, Any]] = {}

        for method, rankings in (
            ("bm25", lexical),
            ("semantic", semantic),
        ):
            # Prevent a duplicate ID in one engine's results from
            # contributing more than once to that engine's RRF score.
            seen_in_engine: set[Any] = set()

            for rank, product in enumerate(rankings, start=1):
                if not isinstance(product, dict):
                    continue

                pid = product.get("product_id")

                if pid is None or pid in seen_in_engine:
                    continue

                seen_in_engine.add(pid)

                contribution = 1.0 / (self.rrf_k + rank)
                scores[pid] += contribution

                components.setdefault(pid, {})[method] = {
                    "rank": rank,
                    "contribution": contribution,
                }

                if pid not in products:
                    products[pid] = dict(product)
                else:
                    products[pid] = self._merge_product_records(
                        products[pid],
                        product,
                    )

        # Stable sorting preserves insertion order when RRF scores tie.
        ranked_ids = sorted(
            scores,
            key=lambda pid: scores[pid],
            reverse=True,
        )

        candidates: list[dict[str, Any]] = []

        for pid in ranked_ids:
            result = dict(products[pid])
            rrf_score = float(scores[pid])

            result["score"] = rrf_score
            result["rrf_score"] = rrf_score
            result["rrf_components"] = components[pid]
            result["retrieval_method"] = "hybrid"

            candidates.append(result)

        lexical_top = self._product_ids(lexical[:top_k])
        semantic_top = self._product_ids(semantic[:top_k])

        intersection = lexical_top & semantic_top
        union = lexical_top | semantic_top

        lexical_ids = self._product_ids(lexical)
        semantic_ids = self._product_ids(semantic)

        full_intersection = lexical_ids & semantic_ids
        full_union = lexical_ids | semantic_ids

        metrics = {
            "top_overlap_count": len(intersection),
            "top_overlap_ratio": (
                len(intersection) / len(union)
                if union else 0.0
            ),
            "full_overlap_ratio": (
                len(full_intersection) / len(full_union)
                if full_union else 0.0
            ),
            "bm25_candidate_count": len(lexical_ids),
            "semantic_candidate_count": len(semantic_ids),
            "unique_fused_candidate_count": len(candidates),
        }

        return candidates, metrics

    def _get_reranker(self):
        """Create the cross-encoder only when reranking is requested."""
        if self.reranker is None:
            from retrieval.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker()

        return self.reranker

    def warmup(self) -> None:
        """Load the cross-encoder before measuring search latency."""
        if self._enabled("QUERYLENCE_RERANKING"):
            reranker = self._get_reranker()

            if hasattr(reranker, "warmup"):
                reranker.warmup()

    def _rank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        enabled: bool,
    ) -> list[dict[str, Any]]:
        """Rerank candidates while preserving each product's original RRF score."""
        if not candidates:
            return []

        if not enabled:
            return candidates[:top_k]

        # Always include enough candidates to satisfy top_k.
        pool_size = max(top_k, self.rerank_candidate_limit)
        rerank_pool = candidates[:pool_size]

        original_rrf_scores = {
            product["product_id"]: float(
                product.get("rrf_score", product.get("score", 0.0))
            )
            for product in rerank_pool
            if product.get("product_id") is not None
        }

        reranked = self._get_reranker().rerank(
            query=query,
            products=rerank_pool,
            top_k=top_k,
        )

        results: list[dict[str, Any]] = []

        for item in reranked or []:
            if not isinstance(item, dict):
                continue

            result = dict(item)
            pid = result.get("product_id")

            original_rrf = original_rrf_scores.get(pid)

            if original_rrf is None:
                original_rrf = float(
                    result.get("rrf_score", result.get("score", 0.0))
                )

            result["score"] = original_rrf
            result["rrf_score"] = original_rrf

            if result.get("reranker_score") is not None:
                result["reranker_score"] = float(
                    result["reranker_score"]
                )

            result["retrieval_method"] = "hybrid_reranked"
            results.append(result)

        return results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates, optionally recover, then rerank."""
        total_start = time.perf_counter()

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")

        if query is None:
            return []

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            return []

        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")

        filters = filters or {}

        reranking = self._enabled("QUERYLENCE_RERANKING")
        recovery_enabled = self._enabled("QUERYLENCE_RECOVERY")

        candidate_count = max(top_k * 5, 20)

        retrieval_start = time.perf_counter()

        candidates, metrics = self._retrieve_candidates(
            query=query,
            candidate_count=candidate_count,
            filters=filters,
            top_k=top_k,
        )

        initial_retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        insufficient = len(candidates) < top_k

        disagreement = (
            metrics["top_overlap_count"] == 0
            and metrics["full_overlap_ratio"] < 0.10
        )

        recovery_triggered = recovery_enabled and (
            insufficient or disagreement
        )

        recovery_reason = None
        recovery_ms = 0.0
        candidate_count_used = candidate_count

        if recovery_triggered:
            recovery_reason = (
                "insufficient_candidates"
                if insufficient
                else "severe_retrieval_disagreement"
            )

            recovery_start = time.perf_counter()
            expanded_count = candidate_count * self.recovery_multiplier

            expanded, expanded_metrics = self._retrieve_candidates(
                query=query,
                candidate_count=expanded_count,
                filters=filters,
                top_k=top_k,
            )

            recovery_ms = (
                time.perf_counter() - recovery_start
            ) * 1000

            current_ids = {
                product.get("product_id")
                for product in candidates
                if product.get("product_id") is not None
            }

            expanded_ids = {
                product.get("product_id")
                for product in expanded
                if product.get("product_id") is not None
            }

            # Keep the larger unique candidate pool.
            if len(expanded_ids) > len(current_ids):
                candidates = expanded
                metrics = expanded_metrics
                candidate_count_used = expanded_count

        rerank_start = time.perf_counter()

        results = self._rank_candidates(
            query=query,
            candidates=candidates,
            top_k=top_k,
            enabled=reranking,
        )

        rerank_ms = (
            time.perf_counter() - rerank_start
        ) * 1000

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        for product in results:
            product["recovery_triggered"] = recovery_triggered
            product["recovery_reason"] = recovery_reason

            product["lexical_semantic_overlap"] = round(
                metrics["top_overlap_ratio"],
                4,
            )

            product["top_k_overlap_count"] = (
                metrics["top_overlap_count"]
            )

            product["full_candidate_overlap"] = round(
                metrics["full_overlap_ratio"],
                4,
            )

            product["bm25_candidate_count"] = (
                metrics["bm25_candidate_count"]
            )

            product["semantic_candidate_count"] = (
                metrics["semantic_candidate_count"]
            )

            product["unique_fused_candidate_count"] = (
                metrics["unique_fused_candidate_count"]
            )

            product["initial_candidate_count"] = candidate_count
            product["candidate_count_used"] = candidate_count_used

            product["initial_retrieval_latency_ms"] = round(
                initial_retrieval_ms,
                3,
            )

            product["recovery_retrieval_latency_ms"] = round(
                recovery_ms,
                3,
            )

            product["reranking_latency_ms"] = round(
                rerank_ms,
                3,
            )

            product["total_latency_ms"] = round(
                total_ms,
                3,
            )

        return results