
import json
import math
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_ROOT / "data" / "processed"

JUDGMENTS_PATH = PROCESSED_DIR / "relevance_judgments.json"
BASELINE_PATH = PROCESSED_DIR / "baseline_comparison.json"
RERANKED_PATH = PROCESSED_DIR / "reranked_comparison.json"
OUTPUT_PATH = PROCESSED_DIR / "ranking_quality_report.json"

TOP_K = 5


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the retrieval evaluation and save the baseline "
            "and reranked reports first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank(results, judgments, k=TOP_K):
    """Return reciprocal rank of the first relevant result."""
    for rank, product in enumerate(results[:k], start=1):
        product_id = product.get("product_id")
        if judgments.get(product_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results, judgments, k=TOP_K):
    """Calculate normalized discounted cumulative gain."""
    gains = []

    for product in results[:k]:
        relevance = judgments.get(product.get("product_id"), 0)
        gains.append((2 ** relevance - 1) / math.log2(len(gains) + 2))

    dcg = sum(gains)

    ideal_relevances = sorted(
        judgments.values(),
        reverse=True,
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_method(reports, judgments_by_query, result_key):
    query_results = []

    for report in reports:
        query = report["query"]
        judgments = judgments_by_query.get(query)

        if not judgments:
            continue

        results = report.get(result_key, [])

        query_results.append({
            "query": query,
            "mrr_at_5": reciprocal_rank(results, judgments),
            "ndcg_at_5": ndcg_at_k(results, judgments),
        })

    if not query_results:
        return {
            "queries_evaluated": 0,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "per_query": [],
        }

    count = len(query_results)

    return {
        "queries_evaluated": count,
        "mrr_at_5": sum(
            item["mrr_at_5"] for item in query_results
        ) / count,
        "ndcg_at_5": sum(
            item["ndcg_at_5"] for item in query_results
        ) / count,
        "per_query": query_results,
    }


def main():
    judgments = load_json(JUDGMENTS_PATH)
    baseline = load_json(BASELINE_PATH)
    reranked = load_json(RERANKED_PATH)

    baseline_methods = {
        "BM25": (baseline, "bm25_top5"),
        "Semantic": (baseline, "semantic_top5"),
        "Hybrid RRF": (baseline, "hybrid_top5"),
        "Hybrid + Cross-Encoder": (reranked, "hybrid_top5"),
    }

    report = {
        "top_k": TOP_K,
        "note": (
            "Preliminary metrics depend on the supplied relevance "
            "judgments. Unjudged products are treated as zero relevance."
        ),
        "methods": {},
    }

    print("=" * 72)
    print("QUERYLENCE RANKING QUALITY EVALUATION")
    print(f"Metrics: MRR@{TOP_K}, nDCG@{TOP_K}")
    print("=" * 72)

    for method, (results, result_key) in baseline_methods.items():
        metrics = evaluate_method(
            results,
            judgments,
            result_key,
        )
        report["methods"][method] = metrics

        print(f"\n{method}")
        print(f"  Queries evaluated: {metrics['queries_evaluated']}")
        print(f"  MRR@{TOP_K}:  {metrics['mrr_at_5']:.4f}")
        print(f"  nDCG@{TOP_K}: {metrics['ndcg_at_5']:.4f}")

        for item in metrics["per_query"]:
            print(
                f"    {item['query']}: "
                f"MRR={item['mrr_at_5']:.4f}, "
                f"nDCG={item['ndcg_at_5']:.4f}"
            )

    baseline_hybrid = report["methods"]["Hybrid RRF"]
    reranked_hybrid = report["methods"]["Hybrid + Cross-Encoder"]

    report["reranker_delta"] = {
        "mrr_at_5": (
            reranked_hybrid["mrr_at_5"]
            - baseline_hybrid["mrr_at_5"]
        ),
        "ndcg_at_5": (
            reranked_hybrid["ndcg_at_5"]
            - baseline_hybrid["ndcg_at_5"]
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nRERANKER CHANGE VS. BASELINE HYBRID")
    print(
        f"  MRR@{TOP_K} delta: "
        f"{report['reranker_delta']['mrr_at_5']:+.4f}"
    )
    print(
        f"  nDCG@{TOP_K} delta: "
        f"{report['reranker_delta']['ndcg_at_5']:+.4f}"
    )
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
