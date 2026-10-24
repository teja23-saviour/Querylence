from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.bm25_engine import BM25SearchEngine
from retrieval.semantic_engine import SemanticSearchEngine
from retrieval.hybrid_engine import HybridSearchEngine


TOP_K = 5
CANDIDATE_COUNT = max(TOP_K * 5, 20)
RRF_K = 60

QUERIES = [
    "comfortable running shoes",
    "blue cotton shirt",
    "gardening tools for outdoor use",
    "lightweight backpack for travel",
]

OUTPUT_PATH = (
    BACKEND_ROOT / "data" / "processed" / "retrieval_comparison.json"
)


def env_enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def product_id(product: dict[str, Any]) -> str:
    return str(product.get("product_id", product.get("id", "")))


def product_title(product: dict[str, Any]) -> str:
    return str(product.get("title", "Untitled product"))


def normalize_for_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def rank_results(
    engine: Any,
    query: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = engine.search(query, top_k=top_k)
    elapsed = (time.perf_counter() - started) * 1000

    if not isinstance(results, list):
        raise TypeError(
            f"{type(engine).__name__}.search() must return a list."
        )

    return results, round(elapsed, 2)


def result_ids(results: list[dict[str, Any]]) -> list[str]:
    return [
        product_id(item)
        for item in results
        if product_id(item)
    ]


def calculate_overlap(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> dict[str, Any]:
    first_ids = set(result_ids(first))
    second_ids = set(result_ids(second))

    intersection = first_ids & second_ids
    union = first_ids | second_ids

    return {
        "overlap_count": len(intersection),
        "union_count": len(union),
        "jaccard": round(len(intersection) / len(union), 4)
        if union else 0.0,
        "shared_product_ids": sorted(intersection),
    }


def verify_rrf(results: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    mismatches = []
    score_mismatches = []

    for item in results:
        components = item.get("rrf_components")
        rrf_score = item.get("rrf_score")

        if not isinstance(components, dict) or rrf_score is None:
            continue

        contributions = []
        valid = True

        for component in components.values():
            if not isinstance(component, dict):
                valid = False
                break

            try:
                rank = int(component["rank"])
                contribution = float(component["contribution"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break

            if (
                rank < 1
                or abs(
                    contribution - 1.0 / (RRF_K + rank)
                ) > 1e-8
            ):
                valid = False
                break

            contributions.append(contribution)

        if not valid or not contributions:
            mismatches.append({
                "product_id": product_id(item),
                "issue": "invalid_rrf_components",
            })
            continue

        checked += 1
        calculated = sum(contributions)

        if abs(float(rrf_score) - calculated) > 1e-8:
            mismatches.append({
                "product_id": product_id(item),
                "stored": float(rrf_score),
                "calculated": calculated,
            })

        public_score = item.get("score")

        if (
            public_score is None
            or abs(float(public_score) - float(rrf_score)) > 1e-8
        ):
            score_mismatches.append({
                "product_id": product_id(item),
                "score": public_score,
                "rrf_score": rrf_score,
            })

    return {
        "checked_count": checked,
        "mismatch_count": len(mismatches),
        "all_checked_scores_match": checked > 0 and not mismatches,
        "mismatches": mismatches,
        "score_field_mismatch_count": len(score_mismatches),
        "score_field_mismatches": score_mismatches,
    }


def summarize_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "brand",
        "category",
        "category_paths",
        "color",
        "material",
        "fabric_type",
        "score",
        "rrf_score",
        "title_relevance_score",
        "product_type_score",
        "detail_relevance_score",
        "reranker_score",
        "identity_coverage",
        "catalog_coverage",
        "query_item_type",
        "query_item_type_terms",
        "query_attributes",
        "matched_query_attributes",
        "matched_title_attributes",
        "matched_catalog_attributes",
        "missing_query_attributes",
        "attribute_coverage",
        "title_attribute_coverage",
        "item_type_alignment",
        "item_type_reason",
        "primary_product_title",
        "final_relevance_score",
        "retrieval_method",
    )

    summarized = []

    for rank, item in enumerate(results[:TOP_K], start=1):
        entry = {
            "rank": rank,
            "product_id": product_id(item),
            "title": product_title(item),
        }

        for field in fields:
            if field in item:
                entry[field] = item[field]

        summarized.append(entry)

    return summarized


def print_results(results: list[dict[str, Any]]) -> None:
    for item in summarize_results(results):
        title = item["title"]
        primary_title = item.get("primary_product_title", "")

        print(
            f"  {item['rank']}. [{item['product_id']}] {title}"
        )

        print(
            f"     RRF={item.get('rrf_score', 'N/A')} | "
            f"Title model={item.get('title_relevance_score', 'N/A')} | "
            f"Taxonomy model={item.get('product_type_score', 'N/A')} | "
            f"Detail model={item.get('detail_relevance_score', 'N/A')} | "
            f"Final relevance={item.get('final_relevance_score', 'N/A')}"
        )

        print(
            f"     Inferred item type={item.get('query_item_type', '')!r} | "
            f"Type terms={item.get('query_item_type_terms', [])} | "
            f"Alignment={item.get('item_type_alignment', 'N/A')} | "
            f"Reason={item.get('item_type_reason', 'N/A')}"
        )

        print(
            f"     Query attributes={item.get('query_attributes', [])} | "
            f"Matched={item.get('matched_query_attributes', [])} | "
            f"Title matches={item.get('matched_title_attributes', [])} | "
            f"Metadata matches={item.get('matched_catalog_attributes', [])} | "
            f"Missing={item.get('missing_query_attributes', [])} | "
            f"Coverage={item.get('attribute_coverage', 0.0):.2f}"
        )

        print(
            f"     Category={item.get('category', '')!r} | "
            f"Color={item.get('color', '')!r} | "
            f"Material={item.get('material', '')!r} | "
            f"Fabric={item.get('fabric_type', '')!r}"
        )

        if item.get("category_paths"):
            path_text = repr(item["category_paths"])
            print(f"     Catalog paths={path_text[:220]}")

        if primary_title:
            print(f"     Primary title={primary_title!r}")

            normalized_primary = normalize_for_identity(primary_title)
            normalized_title = normalize_for_identity(title)

            if (
                normalized_primary
                and normalized_primary not in normalized_title
            ):
                print(
                    "     WARNING: Primary title does not match the "
                    "displayed product title. Inspect product identity "
                    "in the retrieval pipeline before trusting this row."
                )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reranking_enabled = env_enabled("QUERYLENCE_RERANKING")
    recovery_enabled = env_enabled("QUERYLENCE_RECOVERY")

    print("=" * 90)
    print("QUERYLENCE RETRIEVAL EVALUATION")
    print(
        f"Top K: {TOP_K} | Candidates per engine: {CANDIDATE_COUNT} "
        f"| RRF K: {RRF_K}"
    )
    print("=" * 90)
    print(f"Reranking enabled: {reranking_enabled}")
    print(f"Recovery enabled: {recovery_enabled}")

    bm25 = BM25SearchEngine()
    semantic = SemanticSearchEngine()

    hybrid = HybridSearchEngine(
        bm25_engine=bm25,
        semantic_engine=semantic,
    )

    print("\nWarming up retrieval models...")

    try:
        semantic.search(QUERIES[0], top_k=1)

        if reranking_enabled:
            hybrid.warmup()
            print("Cross-encoder warmup complete.")

    except Exception as error:
        raise RuntimeError(
            "Retrieval model warmup failed; evaluation was not run."
        ) from error

    report = []
    total_checked = 0
    total_rrf_mismatches = 0
    total_score_mismatches = 0

    for query_number, query in enumerate(QUERIES, start=1):
        print("\n" + "=" * 90)
        print(f"QUERY {query_number}/{len(QUERIES)}: {query}")
        print("=" * 90)

        bm25_results, bm25_ms = rank_results(
            bm25,
            query,
            TOP_K,
        )

        semantic_results, semantic_ms = rank_results(
            semantic,
            query,
            TOP_K,
        )

        started = time.perf_counter()
        hybrid_results = hybrid.search(query, top_k=TOP_K)
        hybrid_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        overlap = calculate_overlap(
            bm25_results,
            semantic_results,
        )

        verification = verify_rrf(hybrid_results)

        total_checked += verification["checked_count"]
        total_rrf_mismatches += verification["mismatch_count"]
        total_score_mismatches += verification["score_field_mismatch_count"]

        print(f"BM25 latency:     {bm25_ms:.2f} ms")
        print(f"Semantic latency: {semantic_ms:.2f} ms")
        print(f"Hybrid latency:   {hybrid_ms:.2f} ms")

        print(
            f"BM25/Semantic top-5 overlap: {overlap['overlap_count']} "
            f"(Jaccard={overlap['jaccard']:.4f})"
        )

        print("\nHybrid top results:")
        print_results(hybrid_results)

        print(
            "\nRRF verification: "
            + (
                "MATCH"
                if verification["all_checked_scores_match"]
                else "MISMATCH"
                if verification["mismatch_count"]
                else "NOT VERIFIABLE"
            )
        )

        report.append({
            "query": query,
            "top_k": TOP_K,
            "candidate_count": CANDIDATE_COUNT,
            "latency_ms": {
                "bm25": bm25_ms,
                "semantic": semantic_ms,
                "hybrid": hybrid_ms,
            },
            "baseline_overlap": overlap,
            "results": {
                "bm25": summarize_results(bm25_results),
                "semantic": summarize_results(semantic_results),
                "hybrid": summarize_results(hybrid_results),
            },
            "rrf_verification": verification,
            "settings": {
                "reranking_enabled": reranking_enabled,
                "recovery_enabled": recovery_enabled,
            },
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)
    print(f"Queries evaluated: {len(report)}")
    print(f"RRF entries verified: {total_checked}")
    print(f"RRF mismatches: {total_rrf_mismatches}")
    print(f"Public score mismatches: {total_score_mismatches}")
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()