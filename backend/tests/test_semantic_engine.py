
import json

import numpy as np
import pytest

from retrieval.semantic_engine import SemanticSearchEngine


class FakeEmbeddingModel:
    """Deterministic embedding model for fast unit tests."""

    def __init__(self):
        self.encode_calls = 0

    def encode(
        self,
        texts,
        batch_size=None,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        self.encode_calls += 1
        vectors = []

        for text in texts:
            text = text.lower()
            vector = np.array(
                [
                    sum(word in text for word in ("garden", "gardening", "trowel")),
                    sum(word in text for word in ("shoe", "shoes", "running")),
                    sum(word in text for word in ("blue", "cotton")),
                ],
                dtype=np.float32,
            )

            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm

            vectors.append(vector)

        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def catalog_files(tmp_path):
    products = [
        {
            "product_id": "P001",
            "title": "Gardening Trowel",
            "brand": "GardenCo",
            "color": "Green",
            "category": "OUTDOOR_LIVING",
            "category_paths": ["/Gardening/Hand Tools"],
            "searchable_text": "Gardening Trowel GardenCo Green",
        },
        {
            "product_id": "P002",
            "title": "Running Shoes",
            "brand": "SportCo",
            "color": "Blue",
            "category": "FOOTWEAR",
            "category_paths": ["/Shoes/Running"],
            "searchable_text": "Running Shoes SportCo Blue",
        },
        {
            "product_id": "P003",
            "title": "Blue Cotton Shirt",
            "brand": "ClothCo",
            "color": "Blue",
            "category": "CLOTHING",
            "category_paths": ["/Clothing/Shirts"],
            "searchable_text": "Blue Cotton Shirt ClothCo",
        },
    ]

    dataset_path = tmp_path / "products.jsonl"
    cache_path = tmp_path / "embeddings.npz"

    with dataset_path.open("w", encoding="utf-8") as file:
        for product in products:
            file.write(json.dumps(product) + "\n")

    return dataset_path, cache_path


def make_engine(catalog_files, model=None):
    dataset_path, cache_path = catalog_files

    return SemanticSearchEngine(
        dataset_path=dataset_path,
        cache_path=cache_path,
        model_name="test-model",
        model=model or FakeEmbeddingModel(),
    )


def test_semantic_search_ranks_relevant_product_first(catalog_files):
    engine = make_engine(catalog_files)

    results = engine.search("garden trowel", top_k=3)

    assert results
    assert results[0]["product_id"] == "P001"
    assert results[0]["retrieval_method"] == "semantic"
    assert isinstance(results[0]["score"], float)


def test_brand_filter(catalog_files):
    engine = make_engine(catalog_files)

    results = engine.search(
        "garden trowel",
        top_k=3,
        filters={"brand": "GardenCo"},
    )

    assert results
    assert all(item["brand"] == "GardenCo" for item in results)


def test_color_filter(catalog_files):
    engine = make_engine(catalog_files)

    results = engine.search(
        "shirt",
        top_k=3,
        filters={"color": "Blue"},
    )

    assert all(item["color"] == "Blue" for item in results)


def test_category_filter_matches_category_path(catalog_files):
    engine = make_engine(catalog_files)

    results = engine.search(
        "garden trowel",
        top_k=3,
        filters={"category": "Hand Tools"},
    )

    assert results
    assert all(
        "Hand Tools" in item["category_paths"][0]
        for item in results
    )


def test_empty_query_returns_no_results(catalog_files):
    engine = make_engine(catalog_files)

    assert engine.search("   ") == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_raises_error(catalog_files, top_k):
    engine = make_engine(catalog_files)

    with pytest.raises(ValueError, match="top_k"):
        engine.search("garden", top_k=top_k)


def test_embeddings_are_reused_from_cache(catalog_files):
    first_model = FakeEmbeddingModel()
    first_engine = make_engine(catalog_files, model=first_model)

    assert first_model.encode_calls == 1
    assert first_engine.cache_path.exists()

    second_model = FakeEmbeddingModel()
    second_engine = make_engine(catalog_files, model=second_model)

    # Cached product embeddings mean no product encoding on reload.
    assert second_model.encode_calls == 0
    assert len(second_engine.embeddings) == 3


def test_missing_catalog_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Product catalog not found"):
        SemanticSearchEngine(
            dataset_path=tmp_path / "missing.jsonl",
            cache_path=tmp_path / "embeddings.npz",
            model=FakeEmbeddingModel(),
        )
