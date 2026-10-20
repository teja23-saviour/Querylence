import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# Resolve the dataset path relative to this Python file.
DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "products.jsonl"
)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25SearchEngine:
    """Lexical product retrieval using the BM25 ranking algorithm."""

    def __init__(self, dataset_path: Path = DATASET_PATH):
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Product catalog not found: {dataset_path}. "
                "Run scripts/prepare_abo.py first."
            )

        self.products: list[dict] = []

        # Load products from the JSONL dataset.
        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    product = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} "
                        f"of {dataset_path}"
                    ) from error

                searchable_text = str(
                    product.get("searchable_text") or ""
                ).strip()

                if not searchable_text:
                    continue

                product["searchable_text"] = searchable_text
                self.products.append(product)

        if not self.products:
            raise ValueError(
                "The product catalog contains no searchable products."
            )

        # Tokenize the catalog once when the engine starts.
        self.tokenized_corpus = [
            tokenize(product["searchable_text"])
            for product in self.products
        ]

        # Vocabulary supports a quick check for completely unknown queries.
        self.vocabulary = {
            token
            for document in self.tokenized_corpus
            for token in document
        }

        # Build the BM25 index once and reuse it for searches.
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(
            f"BM25 indexed {len(self.products):,} products "
            f"with {len(self.vocabulary):,} unique tokens."
        )

    @staticmethod
    def _matches_filter(product: dict, filters: dict) -> bool:
        """Return True when a product satisfies all supported filters."""
        brand_filter = filters.get("brand")
        if brand_filter:
            brand = str(product.get("brand") or "")
            if str(brand_filter).casefold() not in brand.casefold():
                return False

        color_filter = filters.get("color")
        if color_filter:
            color = str(product.get("color") or "")
            if str(color_filter).casefold() not in color.casefold():
                return False

        category_filter = filters.get("category")
        if category_filter:
            category_values = [
                str(product.get("category") or ""),
                *[
                    str(value)
                    for value in (product.get("category_paths") or [])
                ],
            ]
            category_query = str(category_filter).casefold()
            if not any(
                category_query in value.casefold()
                for value in category_values
            ):
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return positively scored products matching query and filters."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query or "")
        if not query_tokens:
            return []

        filters = filters or {}

        # No known query terms means there cannot be a lexical match.
        if not any(token in self.vocabulary for token in query_tokens):
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Keep only positive-score products before ranking. This avoids
        # filling results with products that have no positive lexical score.
        # Use partial selection to avoid sorting the entire catalog.
        import heapq

        candidate_count = min(len(scores), max(top_k * 20, top_k))
        candidate_indices = heapq.nlargest(
            candidate_count,
            range(len(scores)),
            key=lambda index: scores[index],
        )

        results = []
        for index in candidate_indices:
            score = float(scores[index])
            if score <= 0:
                continue

            product = self.products[index]
            if not self._matches_filter(product, filters):
                continue

            results.append(
                {
                    "product_id": product.get("product_id"),
                    "title": product.get("title"),
                    "description": product.get("description"),
                    "brand": product.get("brand"),
                    "category": product.get("category"),
                    "color": product.get("color"),
                    "main_image_id": product.get("main_image_id"),
                    "marketplace": product.get("marketplace"),
                    "domain_name": product.get("domain_name"),
                    "score": round(score, 4),
                    "retrieval_method": "bm25",
                }
            )

            if len(results) >= top_k:
                break

        return results
