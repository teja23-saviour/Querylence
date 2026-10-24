
"""Semantic product retrieval using Sentence Transformers."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BACKEND_DIR / "data" / "processed" / "products.jsonl"
DEFAULT_CACHE_PATH = BACKEND_DIR / "data" / "processed" / "semantic_embeddings.npz"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(value: Any) -> str:
    """Convert strings and metadata values to normalized lowercase text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value if item is not None)
    return re.sub(r"\s+", " ", str(value).strip().lower())


def contains_phrase(text: Any, phrase: Any) -> bool:
    """Match a complete phrase without matching it inside another word."""
    text = normalize_text(text)
    phrase = normalize_text(phrase)
    if not text or not phrase:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )
    return re.search(pattern, text) is not None


class SemanticSearchEngine:
    """Rank products using dense semantic embeddings."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.cache_path = Path(cache_path).resolve()
        self.model_name = model_name

        # Preserve dependency injection so tests can use a fake model
        # without downloading or loading a real transformer.
        self.products = self._load_products()
        self.model = (
            model if model is not None
            else SentenceTransformer(self.model_name)
        )
        self.embeddings = self._load_or_create_embeddings()

        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

        if (
            self.embeddings.ndim != 2
            or self.embeddings.shape[0] != len(self.products)
            or self.embeddings.shape[1] == 0
        ):
            raise ValueError(
                f"Invalid embedding matrix shape: {self.embeddings.shape}"
            )

    def _load_products(self) -> list[dict[str, Any]]:
        """Load the processed product catalog from JSONL."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Product catalog not found: {self.dataset_path}"
            )

        products: list[dict[str, Any]] = []

        with self.dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    product = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: "
                        f"{self.dataset_path}"
                    ) from exc

                if not isinstance(product, dict):
                    continue

                searchable_text = normalize_text(
                    product.get("searchable_text")
                )

                if not searchable_text:
                    searchable_text = " ".join(
                        normalize_text(product.get(field))
                        for field in (
                            "title",
                            "description",
                            "brand",
                            "category",
                            "category_paths",
                            "color",
                            "material",
                            "fabric_type",
                            "keywords",
                        )
                        if product.get(field) is not None
                    ).strip()

                product["searchable_text"] = searchable_text
                products.append(product)

        return products

    def _cache_is_valid(self) -> bool:
        """Accept compatible old caches without dataset_mtime_ns."""
        if not self.cache_path.exists():
            return False

        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                required = {
                    "embeddings",
                    "dataset_path",
                    "dataset_size",
                    "model_name",
                    "product_count",
                }

                if not required.issubset(cached.files):
                    return False

                stat = self.dataset_path.stat()

                matches = (
                    str(cached["dataset_path"].item())
                    == str(self.dataset_path.resolve())
                    and int(cached["dataset_size"].item()) == stat.st_size
                    and str(cached["model_name"].item()) == self.model_name
                    and int(cached["product_count"].item()) == len(self.products)
                )

                if not matches:
                    return False

                # Validate the timestamp when present. Older caches do not
                # have it, so don't reject them only for that reason.
                if "dataset_mtime_ns" in cached.files:
                    if (
                        int(cached["dataset_mtime_ns"].item())
                        != stat.st_mtime_ns
                    ):
                        return False

                embeddings = cached["embeddings"]
                return (
                    embeddings.ndim == 2
                    and embeddings.shape[0] == len(self.products)
                    and embeddings.shape[1] > 0
                )

        except (OSError, ValueError, KeyError, EOFError, TypeError):
            return False

    def _load_or_create_embeddings(self) -> np.ndarray:
        """Load valid embeddings or create and cache them."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self._cache_is_valid():
            with np.load(self.cache_path, allow_pickle=False) as cached:
                embeddings = cached["embeddings"].astype(
                    np.float32, copy=True
                )

            if (
                embeddings.ndim == 2
                and embeddings.shape[0] == len(self.products)
                and embeddings.shape[1] > 0
            ):
                print("Loaded cached semantic embeddings:", embeddings.shape)
                return embeddings

        print(
            "Compatible semantic cache not found. "
            "Generating product embeddings."
        )

        texts = [p.get("searchable_text", "") for p in self.products]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(self.products)
            or embeddings.shape[1] == 0
        ):
            raise ValueError(
                f"Model returned invalid embedding shape: {embeddings.shape}"
            )

        stat = self.dataset_path.stat()
        temporary_path = self.cache_path.with_name(
            self.cache_path.stem + ".tmp.npz"
        )

        try:
            np.savez_compressed(
                temporary_path,
                embeddings=embeddings,
                dataset_path=np.array(str(self.dataset_path.resolve())),
                dataset_size=np.array(stat.st_size),
                dataset_mtime_ns=np.array(stat.st_mtime_ns),
                model_name=np.array(self.model_name),
                product_count=np.array(len(self.products)),
            )
            temporary_path.replace(self.cache_path)
        finally:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        print("Saved semantic embedding cache:", self.cache_path)
        return embeddings

    @staticmethod
    def _matches_filters(
        product: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        """Apply supported product metadata filters."""
        if not filters:
            return True

        brand = product.get("brand")
        color = product.get("color")
        material = (
            f"{normalize_text(product.get('material'))} "
            f"{normalize_text(product.get('fabric_type'))}"
        ).strip()
        category_text = (
            f"{normalize_text(product.get('category'))} "
            f"{normalize_text(product.get('category_paths'))}"
        ).strip()
        product_text = (
            f"{normalize_text(product.get('title'))} "
            f"{category_text}"
        ).strip()

        for key, value in filters.items():
            if value is None or not str(value).strip():
                continue

            expected = normalize_text(value)

            if key == "brand":
                if not contains_phrase(brand, expected):
                    return False
            elif key == "color":
                if not contains_phrase(color, expected):
                    return False
            elif key == "material":
                if not contains_phrase(material, expected):
                    return False
            elif key == "product_type":
                if not contains_phrase(product_text, expected):
                    return False
            elif key == "category":
                if not contains_phrase(category_text, expected):
                    return False
            elif key in {"min_price", "max_price"}:
                raise ValueError(
                    f"Price filtering is not supported by the semantic "
                    f"engine: {key}"
                )
            else:
                raise ValueError(f"Unsupported semantic search filter: {key}")

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring matching products."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Preserve the established API behavior for blank queries.
        if not isinstance(query, str) or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
            or query_embedding.shape[1] != self.embeddings.shape[1]
        ):
            raise ValueError(
                "Query embedding dimensions do not match product embeddings."
            )

        scores = self.embeddings @ query_embedding[0]
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []

        for index in ranked_indices:
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
