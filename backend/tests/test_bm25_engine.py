
import json

import pytest

from retrieval.bm25_engine import BM25SearchEngine, tokenize


@pytest.fixture
def engine(tmp_path):
    products = [
        {
            "product_id": "P1",
            "title": "Gardener gift set",
            "description": "Garden pruner and trowel tools",
            "brand": "Fiskars",
            "category": "OUTDOOR_LIVING",
            "category_paths": ["/Gardening/Hand Tools"],
            "color": "orange",
            "searchable_text": (
                "Fiskars gardener gift set garden pruner trowel tools"
            ),
        },
        {
            "product_id": "P2",
            "title": "Garden shears",
            "description": "Pruning shears for plants",
            "brand": "Acme",
            "category": "GARDEN",
            "category_paths": ["/Gardening/Pruning"],
            "color": "green",
            "searchable_text": "Acme garden shears pruning plants green",
        },
        {
            "product_id": "P3",
            "title": "Coffee mug",
            "description": "Ceramic mug for hot drinks",
            "brand": "HomeCo",
            "category": "KITCHEN",
            "category_paths": ["/Kitchen/Drinkware"],
            "color": "blue",
            "searchable_text": "HomeCo coffee mug ceramic hot drinks blue",
        },
    ]

    dataset = tmp_path / "products.jsonl"
    dataset.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )

    return BM25SearchEngine(dataset)


def test_tokenize():
    assert tokenize("Garden-Tools 123!") == ["garden", "tools", "123"]


def test_relevant_product_ranks_first(engine):
    results = engine.search("gardener gift set pruner trowel", top_k=2)

    assert results
    assert results[0]["product_id"] == "P1"


def test_brand_filter(engine):
    results = engine.search(
        "gardener gift set",
        filters={"brand": "Fiskars"},
        top_k=5,
    )

    assert results
    assert all(item["brand"] == "Fiskars" for item in results)


def test_color_filter(engine):
    results = engine.search(
        "garden shears",
        filters={"color": "green"},
        top_k=5,
    )

    assert results
    assert all(item["color"] == "green" for item in results)


def test_category_path_filter(engine):
    results = engine.search(
        "gardener gift set",
        filters={"category": "Hand Tools"},
        top_k=5,
    )

    assert results
    assert all(item["product_id"] == "P1" for item in results)


def test_unknown_query_returns_no_results(engine):
    assert engine.search("zzzxxyy qqqvvv") == []


def test_empty_query_returns_no_results(engine):
    assert engine.search("") == []
    assert engine.search("!!! ---") == []


def test_top_k_limit(engine):
    results = engine.search("garden", top_k=1)
    assert len(results) <= 1


def test_invalid_top_k_raises_error(engine):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        engine.search("garden", top_k=0)
