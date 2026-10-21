
import heapq
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "products.jsonl"
)

CACHE_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25SearchEngine:
    """Lexical product retrieval with a persistent BM25 index."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        cache_path: Path | None = None,
    ):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Product catalog not found: {self.dataset_path}. "
                "Run scripts/prepare_abo.py first."
            )

        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else self.dataset_path.with_name("bm25_index.pkl")
        )

        self.dataset_signature = self._get_dataset_signature()

        if self._load_cache():
            print(
                f"Loaded cached BM25 index: "
                f"{len(self.products):,} products."
            )
        else:
            self._build_index()
            self._save_cache()
            print(
                f"Built BM25 index: {len(self.products):,} products, "
                f"{len(self.vocabulary):,} unique tokens."
            )

    def _get_dataset_signature(self) -> dict:
        """Detect whether the source catalog has changed."""
        stat = self.dataset_path.stat()

        return {
            "version": CACHE_VERSION,
            "dataset_path": str(self.dataset_path.resolve()),
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }

    def _load_cache(self) -> bool:
        """Load a matching, previously saved index."""
        if not self.cache_path.exists():
            return False

        try:
            with self.cache_path.open("rb") as file:
                cached = pickle.load(file)

            if cached.get("signature") != self.dataset_signature:
                return False

            self.products = cached["products"]
            self.tokenized_corpus = cached["tokenized_corpus"]
            self.vocabulary = cached["vocabulary"]
            self.bm25 = cached["bm25"]

            return True

        except Exception as error:
            # A missing, outdated, or unreadable cache can be rebuilt.
            print(f"BM25 cache unavailable; rebuilding: {error}")
            return False

    def _build_index(self) -> None:
        """Read the catalog and construct the BM25 index."""
        self.products = []

        with self.dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    product = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} "
                        f"of {self.dataset_path}"
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

        self.tokenized_corpus = [
            tokenize(product["searchable_text"])
            for product in self.products
        ]

        self.vocabulary = {
            token
            for document in self.tokenized_corpus
            for token in document
        }

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _save_cache(self) -> None:
        """Persist the index for reuse after process restarts."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached = {
            "signature": self.dataset_signature,
            "products": self.products,
            "tokenized_corpus": self.tokenized_corpus,
            "vocabulary": self.vocabulary,
            "bm25": self.bm25,
        }

        temporary_path = self.cache_path.with_suffix(".pkl.tmp")

        try:
            with temporary_path.open("wb") as file:
                pickle.dump(
                    cached,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            temporary_path.replace(self.cache_path)

        except OSError as error:
            # Searching can still work if disk caching fails.
            print(f"Could not save BM25 cache: {error}")

            if temporary_path.exists():
                temporary_path.unlink()

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
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be at least 1")

        query_tokens = tokenize(query or "")

        if not query_tokens:
            return []

        filters = filters or {}

        if not any(token in self.vocabulary for token in query_tokens):
            return []

        scores = self.bm25.get_scores(query_tokens)

        candidate_count = min(
            len(scores),
            max(top_k * 20, top_k),
        )

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
