
from collections import defaultdict


class HybridSearchEngine:
    """Combine lexical and semantic search rankings using RRF."""

    def __init__(
        self,
        bm25_engine,
        semantic_engine,
        rrf_k: int = 60,
    ):
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.bm25_engine = bm25_engine
        self.semantic_engine = semantic_engine
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query = (query or "").strip()
        if not query:
            return []

        candidate_count = max(top_k * 5, 20)

        lexical_results = self.bm25_engine.search(
            query,
            top_k=candidate_count,
            filters=filters,
        )

        semantic_results = self.semantic_engine.search(
            query,
            top_k=candidate_count,
            filters=filters,
        )

        fused_scores = defaultdict(float)
        products = {}

        self._add_rankings(lexical_results, fused_scores, products)
        self._add_rankings(semantic_results, fused_scores, products)

        ranked_ids = sorted(
            fused_scores,
            key=lambda product_id: fused_scores[product_id],
            reverse=True,
        )

        results = []

        for product_id in ranked_ids[:top_k]:
            product = dict(products[product_id])
            product["score"] = float(fused_scores[product_id])
            product["retrieval_method"] = "hybrid"
            results.append(product)

        return results

    def _add_rankings(self, rankings, fused_scores, products):
        for rank, product in enumerate(rankings, start=1):
            product_id = product.get("product_id")

            if not product_id:
                continue

            fused_scores[product_id] += 1.0 / (self.rrf_k + rank)

            if product_id not in products:
                products[product_id] = product
