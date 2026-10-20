
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from retrieval.bm25_engine import BM25SearchEngine
from schemas.search import SearchRequest


app = FastAPI(
    title="Querylence API",
    description=(
        "An intent-aware, self-correcting multimodal "
        "e-commerce retrieval engine."
    ),
    version="0.1.0",
)


# --------------------------------------------------
# Search engine initialization
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_search_engine() -> BM25SearchEngine:
    """Load and cache the BM25 search engine."""
    return BM25SearchEngine()


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Querylence API is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Querylence API",
    }


# --------------------------------------------------
# Text search endpoint
# --------------------------------------------------

@app.post("/search/text")
def search_text(request: SearchRequest):
    """Search the product catalog using BM25."""

    # The current catalog does not provide reliable prices.
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

        filters = request.filters.model_dump(
            exclude_none=True
        )

        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters,
        )

        return {
            "query": request.query,
            "retrieval_method": "bm25",
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
        # Log the underlying exception in the server terminal.
        print(f"Querylence search error: {error!r}")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during product search.",
        ) from error
