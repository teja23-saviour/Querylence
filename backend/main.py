
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from retrieval.bm25_engine import BM25SearchEngine
from retrieval.semantic_engine import SemanticSearchEngine
from retrieval.hybrid_engine import HybridSearchEngine
from retrieval.query_understanding import understand_query
from schemas.search import SearchRequest


app = FastAPI(
    title="Querylence API",
    description=(
        "An intent-aware, self-correcting multimodal "
        "e-commerce retrieval engine."
    ),
    version="0.3.0",
)


@lru_cache(maxsize=1)
def get_search_engine() -> HybridSearchEngine:
    """Initialize and cache the retrieval engines."""
    bm25_engine = BM25SearchEngine()
    semantic_engine = SemanticSearchEngine()

    return HybridSearchEngine(
        bm25_engine=bm25_engine,
        semantic_engine=semantic_engine,
    )


@app.get("/")
def root():
    return {
        "message": "Querylence API is running",
        "version": "0.3.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Querylence API",
    }


@app.post("/search/text")
def search_text(request: SearchRequest):
    """Understand the query and perform hybrid product retrieval."""

    if (
        request.filters.min_price is not None
        or request.filters.max_price is not None
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Price filtering is not supported by the "
                "current dataset because reliable price data "
                "is unavailable."
            ),
        )

    try:
        engine = get_search_engine()

        explicit_filters = request.filters.model_dump(
            exclude_none=True
        )

        understood = understand_query(
            query=request.query,
            explicit_filters=explicit_filters,
        )

        results = engine.search(
            query=understood["search_query"],
            top_k=request.top_k,
            filters=understood["filters"],
        )

        return {
            "query": request.query,
            "intent": understood["intent"],
            "applied_filters": understood["filters"],
            "retrieval_method": "hybrid",
            "result_count": len(results),
            "results": results,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Product catalog not found. "
                "Run the ABO dataset preparation script first."
            ),
        ) from error

    except Exception as error:
        print(f"Querylence search error: {error!r}")

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred during product search."
            ),
        ) from error
