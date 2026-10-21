

import json
import sys
import time
from pathlib import Path

# Ensure Python can import modules from the backend root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.bm25_engine import BM25SearchEngine
from retrieval.semantic_engine import SemanticSearchEngine
from retrieval.hybrid_engine import HybridSearchEngine


QUERIES = [
    "comfortable running shoes",
    "blue cotton shirt",
    "gardening tools for outdoor use",
    "lightweight backpack for travel",
]

TOP_K = 5


def run_search(engine, query):
    start = time.perf_counter()
    results = engine.search(query, top_k=TOP_K)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return results, elapsed_ms


def get_product_id(result):
    return result.get("product_id", "")


def display_results(method, results, elapsed_ms):
    print(f"\n  {method} — {elapsed_ms:.1f} ms")

    if not results:
        print("    No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"    {rank}. [{get_product_id(result)}] "
            f"{result.get('title', 'Untitled')}"
        )


def get_rank_map(results):
    return {
        get_product_id(item): rank
        for rank, item in enumerate(results, start=1)
        if get_product_id(item)
    }


def calculate_rrf_score(bm25_rank, semantic_rank, rrf_k=60):
    """Calculate expected RRF score from ranks when present."""
    score = 0.0

    if bm25_rank is not None:
        score += 1.0 / (rrf_k + bm25_rank)

    if semantic_rank is not None:
        score += 1.0 / (rrf_k + semantic_rank)

    return score


def main():
    print("=" * 90)
    print("QUERYLENCE RETRIEVAL EVALUATION")
    print(f"Top K: {TOP_K}")
    print("=" * 90)

    print("\nLoading retrieval engines...")

    bm25 = BM25SearchEngine()
    semantic = SemanticSearchEngine()
    hybrid = HybridSearchEngine(bm25, semantic)

    engines = {
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid,
    }

    report = []

    for query in QUERIES:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")

        query_report = {
            "query": query,
            "methods": {},
            "comparison": {},
            "rrf_diagnostics": [],
        }

        rankings = {}
        timings = {}

        # Run all three retrieval methods.
        for method, engine in engines.items():
            results, elapsed_ms = run_search(engine, query)

            rankings[method] = results
            timings[method] = elapsed_ms

            display_results(method, results, elapsed_ms)

            query_report["methods"][method] = {
                "elapsed_ms": round(elapsed_ms, 2),
                "results": [
                    {
                        "rank": rank,
                        "product_id": get_product_id(item),
                        "title": item.get("title", ""),
                        "score": item.get("score"),
                    }
                    for rank, item in enumerate(results, start=1)
                ],
            }

        # Compare product IDs in the displayed top-K results.
        bm25_ids = set(get_rank_map(rankings["bm25"]))
        semantic_ids = set(get_rank_map(rankings["semantic"]))
        hybrid_ids = set(get_rank_map(rankings["hybrid"]))

        shared_ids = bm25_ids & semantic_ids

        print("\n  RANKING COMPARISON")
        print(
            f"    BM25 / semantic overlap: "
            f"{len(shared_ids)} of {TOP_K}"
        )
        print(
            f"    Hybrid products in BM25 top {TOP_K}: "
            f"{len(hybrid_ids & bm25_ids)} of {len(hybrid_ids)}"
        )
        print(
            f"    Hybrid products in semantic top {TOP_K}: "
            f"{len(hybrid_ids & semantic_ids)} of {len(hybrid_ids)}"
        )

        query_report["comparison"] = {
            "bm25_semantic_overlap_count": len(shared_ids),
            "bm25_semantic_overlap_ids": sorted(shared_ids),
            "hybrid_bm25_overlap_count": len(hybrid_ids & bm25_ids),
            "hybrid_semantic_overlap_count": len(
                hybrid_ids & semantic_ids
            ),
        }

        # Verify RRF scores against ranks visible in the top-K lists.
        bm25_ranks = get_rank_map(rankings["bm25"])
        semantic_ranks = get_rank_map(rankings["semantic"])

        print("\n  RRF SCORE VERIFICATION")
        print(
            "    Note: ranks shown here are from each engine's "
            f"top {TOP_K} only."
        )

        for item in rankings["hybrid"]:
            pid = get_product_id(item)
            bm25_rank = bm25_ranks.get(pid)
            semantic_rank = semantic_ranks.get(pid)

            visible_rank_score = calculate_rrf_score(
                bm25_rank,
                semantic_rank,
                rrf_k=hybrid.rrf_k,
            )

            actual_score = item.get("score", 0.0)

            if bm25_rank is None and semantic_rank is None:
                explanation = (
                    "Not in either displayed top-K; "
                    "may have ranked deeper in candidate lists."
                )
            elif bm25_rank is not None and semantic_rank is not None:
                explanation = "Appears in both displayed rankings."
            else:
                explanation = (
                    "Appears in one displayed ranking; "
                    "check the other engine's deeper candidates."
                )

            print(
                f"    {pid}: BM25 rank={bm25_rank}, "
                f"semantic rank={semantic_rank}, "
                f"actual RRF={actual_score:.6f}, "
                f"visible-rank estimate={visible_rank_score:.6f}"
            )
            print(f"      {explanation}")

            query_report["rrf_diagnostics"].append({
                "product_id": pid,
                "bm25_top_k_rank": bm25_rank,
                "semantic_top_k_rank": semantic_rank,
                "actual_rrf_score": actual_score,
                "visible_rank_score_estimate": round(
                    visible_rank_score, 8
                ),
                "in_bm25_top_k": bm25_rank is not None,
                "in_semantic_top_k": semantic_rank is not None,
            })

        # Latency summary.
        print("\n  LATENCY SUMMARY")
        for method, elapsed_ms in timings.items():
            print(f"    {method}: {elapsed_ms:.1f} ms")

        report.append(query_report)

    # Save the complete report outside Git-tracked source files.
    output_path = (
        BACKEND_ROOT
        / "data"
        / "processed"
        / "retrieval_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print(f"Report saved to:\n{output_path}")
    print(
        "\nNote: timings exclude initial engine construction, "
        "but may include runtime overhead such as cache and CPU effects."
    )
    print(
        "RRF verification uses visible top-K ranks as a diagnostic; "
        "it does not prove the full candidate contribution by itself."
    )


if __name__ == "__main__":
    main()
