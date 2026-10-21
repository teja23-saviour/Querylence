
import pytest

from retrieval.hybrid_engine import HybridSearchEngine


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_filters = None

    def search(self, query, top_k=10, filters=None):
        self.last_query = query
        self.last_filters = filters
        return self.results[:top_k]


def product(product_id, title):
    return {
        "product_id": product_id,
        "title": title,
        "score": 0.9,
    }


def test_fuses_results_from_both_engines():
    bm25 = FakeEngine([
        product("A", "Blue running shoes"),
        product("B", "Running shirt"),
    ])
    semantic = FakeEngine([
        product("B", "Running shirt"),
        product("C", "Sports footwear"),
    ])

    engine = HybridSearchEngine(bm25, semantic)
    results = engine.search("running shoes", top_k=3)

    ids = [item["product_id"] for item in results]

    assert ids[0] == "B"
    assert set(ids) == {"A", "B", "C"}
    assert all(item["retrieval_method"] == "hybrid" for item in results)


def test_product_found_by_both_engines_gets_fused_score():
    bm25 = FakeEngine([product("A", "Blue shoes")])
    semantic = FakeEngine([product("A", "Blue shoes")])

    engine = HybridSearchEngine(bm25, semantic)
    results = engine.search("blue shoes")

    expected = 2 / 61

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(expected)


def test_passes_filters_to_both_engines():
    filters = {"brand": "Nike"}
    bm25 = FakeEngine([product("A", "Running shoes")])
    semantic = FakeEngine([product("A", "Running shoes")])

    engine = HybridSearchEngine(bm25, semantic)
    engine.search("running shoes", filters=filters)

    assert bm25.last_filters == filters
    assert semantic.last_filters == filters


def test_empty_query_returns_no_results():
    engine = HybridSearchEngine(FakeEngine([]), FakeEngine([]))

    assert engine.search("   ") == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_raises_error(top_k):
    engine = HybridSearchEngine(FakeEngine([]), FakeEngine([]))

    with pytest.raises(ValueError, match="top_k"):
        engine.search("shoes", top_k=top_k)


def test_respects_top_k():
    bm25 = FakeEngine([
        product("A", "Shoes"),
        product("B", "Shirt"),
    ])
    semantic = FakeEngine([
        product("C", "Jacket"),
        product("D", "Hat"),
    ])

    engine = HybridSearchEngine(bm25, semantic)
    results = engine.search("clothing", top_k=2)

    assert len(results) == 2
