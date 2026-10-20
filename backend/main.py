
from fastapi import FastAPI

app = FastAPI(
    title="Querylence API",
    description=(
        "An intent-aware, self-correcting "
        "multimodal e-commerce retrieval engine."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "Querylence",
        "message": "API is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "querylence-backend",
    }
