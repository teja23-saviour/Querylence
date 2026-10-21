
import sys
from pathlib import Path

# Add the backend directory to Python's import path.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.bm25_engine import BM25SearchEngine


def main():
    print("\n--- First initialization ---")
    engine1 = BM25SearchEngine()

    print("\n--- Second initialization ---")
    engine2 = BM25SearchEngine()

    query = "comfortable running shoes"

    results1 = engine1.search(query, top_k=5)
    results2 = engine2.search(query, top_k=5)

    ids1 = [item["product_id"] for item in results1]
    ids2 = [item["product_id"] for item in results2]

    assert ids1 == ids2, "Cached results differ from original results"

    print("\nCache test passed.")
    print(f"Matching results: {len(ids1)}")


if __name__ == "__main__":
    main()
