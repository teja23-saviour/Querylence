
import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO

ARCHIVE_PATH = Path("data/raw/abo-listings.tar")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "products.jsonl"
REPORT_PATH = OUTPUT_DIR / "dataset_report.json"

# Prefer India and US English, then other English locales.
LANGUAGE_PRIORITY = {
    "en_IN": 0,
    "en_US": 1,
    "en_GB": 2,
    "en_CA": 3,
    "en_AU": 4,
}


def localized_value(
    entries: Any,
    preferred_language: str,
) -> str:
    """Select a localized text value, preferring English."""
    if not isinstance(entries, list):
        return ""

    valid = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("value"), str)
        and entry["value"].strip()
    ]

    if not valid:
        return ""

    matching = [
        entry for entry in valid
        if entry.get("language_tag") == preferred_language
    ]

    english = [
        entry for entry in valid
        if str(entry.get("language_tag", "")).startswith("en_")
    ]

    selected = matching or english or valid
    return selected[0]["value"].strip()


def localized_values(
    entries: Any,
    preferred_language: str,
) -> list[str]:
    """Extract multiple localized values."""
    if not isinstance(entries, list):
        return []

    matching = [
        entry for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("value"), str)
        and entry["value"].strip()
        and entry.get("language_tag") == preferred_language
    ]

    if not matching:
        matching = [
            entry for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("value"), str)
            and entry["value"].strip()
            and str(entry.get("language_tag", "")).startswith("en_")
        ]

    return list(dict.fromkeys(
        entry["value"].strip() for entry in matching
    ))


def category_paths(record: dict) -> list[str]:
    """Extract category hierarchy paths."""
    nodes = record.get("node", [])
    paths = []

    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                name = node.get("node_name")
                if isinstance(name, str) and name.strip():
                    paths.append(name.strip())

    return list(dict.fromkeys(paths))


def normalize_record(
    record: dict,
) -> Optional[dict]:
    """Convert a raw listing into Querylence's catalog schema."""
    product_id = record.get("item_id")

    if not isinstance(product_id, str) or not product_id.strip():
        return None

    title_entries = record.get("item_name", [])
    english_titles = [
        entry for entry in title_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("value"), str)
        and entry["value"].strip()
        and str(entry.get("language_tag", "")).startswith("en_")
    ] if isinstance(title_entries, list) else []

    # Keep only listings with an English title.
    if not english_titles:
        return None

    title_entry = min(
        english_titles,
        key=lambda entry: LANGUAGE_PRIORITY.get(
            entry.get("language_tag"), 99
        ),
    )

    language = title_entry.get("language_tag", "")
    title = title_entry["value"].strip()

    descriptions = localized_values(
        record.get("bullet_point", []), language
    )
    keywords = localized_values(
        record.get("item_keywords", []), language
    )

    brand = localized_value(record.get("brand", []), language)
    color = localized_value(record.get("color", []), language)
    material = localized_value(record.get("material", []), language)
    fabric = localized_value(record.get("fabric_type", []), language)

    product_types = record.get("product_type", [])
    product_type = ""

    if isinstance(product_types, list) and product_types:
        first = product_types[0]
        if isinstance(first, dict):
            product_type = str(first.get("value", "")).strip()

    image_id = record.get("main_image_id")
    other_images = record.get("other_image_id", [])

    if not isinstance(image_id, str):
        image_id = ""

    if not isinstance(other_images, list):
        other_images = []

    paths = category_paths(record)

    # This text becomes the document searched by BM25
    # and encoded by the semantic text model.
    searchable_text = " ".join(
        value for value in [
            title,
            brand,
            product_type,
            color,
            material,
            fabric,
            " ".join(keywords),
            " ".join(descriptions),
            " ".join(paths),
        ]
        if value
    )

    return {
        "product_id": product_id.strip(),
        "title": title,
        "description": " ".join(descriptions),
        "brand": brand,
        "category": product_type,
        "category_paths": paths,
        "color": color,
        "material": material,
        "fabric_type": fabric,
        "keywords": keywords,
        "main_image_id": image_id,
        "other_image_ids": [
            value for value in other_images
            if isinstance(value, str)
        ],
        "marketplace": record.get("marketplace", ""),
        "country": record.get("country", ""),
        "domain_name": record.get("domain_name", ""),
        "language": language,
        "searchable_text": searchable_text,
    }


def iter_records(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> Iterator[dict]:
    """Stream JSON-lines records from one compressed shard."""
    raw = archive.extractfile(member)

    if raw is None:
        return

    with raw:
        with gzip.GzipFile(fileobj=raw) as compressed:
            import io

            text_stream: TextIO = io.TextIOWrapper(
                compressed, encoding="utf-8"
            )

            for line in text_stream:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    yield record


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Archive not found: {ARCHIVE_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep one preferred English listing per product ID.
    products: dict[str, tuple[int, dict]] = {}
    records_seen = 0
    invalid_records = 0
    language_counts = Counter()
    marketplace_counts = Counter()

    with tarfile.open(ARCHIVE_PATH, mode="r") as archive:
        shards = sorted(
    (
        member
        for member in archive.getmembers()
        if member.name.startswith("listings/metadata/")
        and member.name.endswith(".json.gz")
    ),
    key=lambda member: member.name,
)

        if not shards:
            raise RuntimeError("No listing metadata shards found.")

        print(f"Metadata shards: {len(shards)}")

        for shard_index, shard in enumerate(shards, start=1):
            print(
                f"[{shard_index}/{len(shards)}] "
                f"Processing {shard.name}"
            )

            for raw_record in iter_records(archive, shard):
                records_seen += 1
                normalized = normalize_record(raw_record)

                if normalized is None:
                    invalid_records += 1
                    continue

                product_id = normalized["product_id"]
                language = normalized["language"]
                language_counts[language] += 1
                marketplace_counts[
                    normalized["domain_name"] or "unknown"
                ] += 1

                priority = LANGUAGE_PRIORITY.get(language, 99)
                existing = products.get(product_id)

                if existing is None or priority < existing[0]:
                    products[product_id] = (priority, normalized)

    # Write deterministic output order for reproducibility.
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for product_id in sorted(products):
            product = products[product_id][1]
            output.write(
                json.dumps(product, ensure_ascii=False) + "\n"
            )

    final_products = [
        products[product_id][1]
        for product_id in sorted(products)
    ]

    report = {
        "source": "Amazon Berkeley Objects",
        "source_archive": str(ARCHIVE_PATH),
        "license": "CC BY 4.0; attribution required",
        "metadata_shards": len(shards),
        "records_seen": records_seen,
        "records_without_english_title": invalid_records,
        "unique_english_products": len(final_products),
        "products_with_main_image_id": sum(
            bool(product["main_image_id"])
            for product in final_products
        ),
        "products_with_description": sum(
            bool(product["description"])
            for product in final_products
        ),
        "products_with_category": sum(
            bool(product["category"] or product["category_paths"])
            for product in final_products
        ),
        "products_with_color": sum(
            bool(product["color"])
            for product in final_products
        ),
        "selected_language_counts": dict(language_counts),
        "source_marketplace_counts": dict(marketplace_counts),
        "output_file": str(OUTPUT_PATH),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n----- DATASET REPORT -----")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nCatalog saved: {OUTPUT_PATH.resolve()}")
    print(f"Report saved: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
