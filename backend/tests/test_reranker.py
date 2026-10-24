from retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        scores = {
            "Blue cotton shirt": 0.1,
            "Running shoes": 0.9,
        }

        return [
            scores["Blue cotton shirt"]
            if "Blue cotton shirt" in pair[1]
            else scores["Running shoes"]
            for pair in pairs
        ]


class ConstantCrossEncoder:
    """Return identical scores to isolate explicit ranking features."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [0.0 for _ in pairs]


def test_reranker_orders_by_model_score_and_preserves_rrf_score():
    products = [
        {
            "product_id": "A",
            "title": "Blue cotton shirt",
            "score": 1 / 61,
            "rrf_score": 1 / 61,
        },
        {
            "product_id": "B",
            "title": "Running shoes",
            "score": 1 / 62,
            "rrf_score": 1 / 62,
        },
    ]

    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "product query",
        products,
        top_k=2,
    )

    assert [item["product_id"] for item in results] == ["B", "A"]

    for item in results:
        assert item["score"] == item["rrf_score"]

    assert results[0]["reranker_score"] == 0.9
    assert results[1]["reranker_score"] == 0.1


def test_reranker_returns_empty_for_blank_query_without_loading_model():
    reranker = CrossEncoderReranker()

    assert reranker.rerank(
        "   ",
        [{"product_id": "A"}],
    ) == []


def test_requested_shirt_ranks_above_shirt_organizer():
    products = [
        {
            "product_id": "organizer",
            "title": "Amazon Brand - Solimo Fabric Shirt Stacker Organiser",
            "category": "HOME",
            "category_paths": [
                "Home Storage and Organization/Clothes Covers"
            ],
            "color": "Blue",
            "material": "Fabric",
            "score": 0.02,
            "rrf_score": 0.02,
        },
        {
            "product_id": "shirt",
            "title": "Blue Cotton Crew Neck Shirt",
            "category": "CLOTHING",
            "category_paths": ["Clothing/Shirts"],
            "color": "Blue",
            "material": "Cotton",
            "score": 0.01,
            "rrf_score": 0.01,
        },
    ]

    reranker = CrossEncoderReranker(model=ConstantCrossEncoder())

    results = reranker.rerank(
        "blue cotton shirt",
        products,
        top_k=2,
    )

    assert results[0]["product_id"] == "shirt"

    organizer = next(
        item
        for item in results
        if item["product_id"] == "organizer"
    )

    assert organizer["item_type_alignment"] == -1.5
    assert (
        organizer["item_type_reason"]
        == "related_product_not_requested_item"
    )


def test_collection_prefix_does_not_count_as_primary_item_match():
    products = [
        {
            "product_id": "wagon",
            "title": (
                "AmazonBasics Garden Tool Collection - Collapsible Folding "
                "Outdoor Garden Utility Wagon with Cover Bag"
            ),
            "category": "OUTDOOR_LIVING",
            "category_paths": [
                "Gardening and Lawn Care/Hand Tools/Tool Sets"
            ],
            "score": 0.02,
            "rrf_score": 0.02,
        },
        {
            "product_id": "tool-set",
            "title": "Strathwood 5-Piece Garden Tool Set in Tote Bag",
            "category": "OUTDOOR_LIVING",
            "category_paths": [
                "Gardening and Lawn Care/Hand Tools/Tool Sets"
            ],
            "score": 0.01,
            "rrf_score": 0.01,
        },
    ]

    reranker = CrossEncoderReranker(model=ConstantCrossEncoder())

    results = reranker.rerank(
        "gardening tools for outdoor use",
        products,
        top_k=2,
    )

    assert results[0]["product_id"] == "tool-set"

    wagon = next(
        item
        for item in results
        if item["product_id"] == "wagon"
    )

    assert wagon["item_type_alignment"] == -2.0
    assert (
        wagon["item_type_reason"]
        == "collection_or_prefix_only"
    )


def test_reranker_preserves_rrf_public_score():
    products = [
        {
            "product_id": "A",
            "title": "Running Shoes",
            "score": 1 / 61,
            "rrf_score": 1 / 61,
        }
    ]

    reranker = CrossEncoderReranker(model=ConstantCrossEncoder())

    result = reranker.rerank(
        "comfortable running shoes",
        products,
        top_k=1,
    )[0]

    assert result["score"] == result["rrf_score"] == 1 / 61
    assert result["query_item_type"] == "running shoe"