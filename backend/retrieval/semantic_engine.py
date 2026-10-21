
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "products.jsonl"
)

EMBEDDING_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "semantic_embeddings.npz"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(text: str) -> str:
    """Normalize whitespace without removing useful product words."""
    return re.sub(r"\s+", " ", text).strip()


class SemanticSearchEngine:
    """Semantic product retrieval using pretrained text embeddings."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        cache_path: Path = EMBEDDING_CACHE_PATH,
        model_name: str = MODEL_NAME,
        model=None,
    ):
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Product catalog not found: {dataset_path}. "
                "Run scripts/prepare_abo.py first."
            )

        self.dataset_path = Path(dataset_path)
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)

        self.products = self._load_products()

        if not self.products:
            raise ValueError("The product catalog contains no searchable products.")

        self.product_texts = [
            normalize_text(str(product.get("searchable_text") or product.get("title") or ""))
            for product in self.products
        ]

        self.embeddings = self._load_or_create_embeddings()

    def _load_products(self) -> list[dict]:
        products = []

        with self.dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    product = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.dataset_path}"
                    ) from error

                searchable_text = normalize_text(
                    str(product.get("searchable_text") or product.get("title") or "")
                )

                if not searchable_text:
                    continue

                products.append(product)

        return products

    def _cache_is_valid(self) -> bool:
        if not self.cache_path.exists():
            return False

        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                return (
                    str(cached["dataset_path"].item()) == str(self.dataset_path.resolve())
                    and int(cached["dataset_size"].item()) == self.dataset_path.stat().st_size
                    and str(cached["model_name"].item()) == self.model_name
                    and int(cached["product_count"].item()) == len(self.products)
                )
        except (OSError, ValueError, KeyError, EOFError):
            return False

    def _load_or_create_embeddings(self) -> np.ndarray:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self._cache_is_valid():
            with np.load(self.cache_path, allow_pickle=False) as cached:
                embeddings = cached["embeddings"].astype(np.float32)

            if embeddings.shape[0] == len(self.products):
                return embeddings

        print(f"Creating semantic embeddings for {len(self.products)} products...")
        print("The first run can take a while.")

        embeddings = self.model.encode(
            self.product_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        temporary_path = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            embeddings=embeddings,
            dataset_path=np.array(str(self.dataset_path.resolve())),
            dataset_size=np.array(self.dataset_path.stat().st_size),
            model_name=np.array(self.model_name),
            product_count=np.array(len(self.products)),
        )
        temporary_path.replace(self.cache_path)

        return embeddings

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query = normalize_text(query or "")
        if not query:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        scores = self.embeddings @ query_embedding
        candidate_count = min(len(self.products), max(top_k * 20, top_k))
        candidate_indices = np.argpartition(
            scores, -candidate_count
        )[-candidate_count:]
        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results = []
        filters = filters or {}

        for index in candidate_indices:
            product = self.products[int(index)]

            if not self._matches_filters(product, filters):
                continue

            result = dict(product)
            result["score"] = float(scores[index])
            result["retrieval_method"] = "semantic"
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filters(product: dict, filters: dict) -> bool:
        for field in ("brand", "color", "category"):
            expected = filters.get(field)
            if expected is None:
                continue

            actual_values = [product.get(field, "")]

            if field == "category":
                actual_values.extend(product.get("category_paths") or [])

            actual = " ".join(str(value) for value in actual_values).casefold()

            if str(expected).casefold() not in actual:
                return False

        return True
