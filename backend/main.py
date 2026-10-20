
from fastapi import FastAPI

from schemas.search import SearchRequest

app = FastAPI(
    title="Querylence API",
    description=(
        "An intent-aware, self-correcting "
        "multimodal e-commerce retrieval engine."
    ),
    version="0.2.0",
)


@app.get("/")
def root():
    return {
        "project": "Querylence",
        "message": "API is running",
        "version": "0.2.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "querylence-backend",
    }


@app.post("/search/text")
def search_text(request: SearchRequest):
    return {
        "status": "request_validated",
        "message": (
            "Search request validated. "
            "Retrieval is not implemented yet."
        ),
        "query": request.query,
        "filters": request.filters.model_dump(),
        "top_k": request.top_k,
    }
