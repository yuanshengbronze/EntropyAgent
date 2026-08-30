"""Build a structured, LLM-enriched catalog for conversational retrieval.

This follows the constrained-selection pattern from SemRank (paper section
3.1 / Figure 4): give the model a fixed attribute vocabulary and ask it to
extract only evidence grounded in the product record.  It is intentionally an
offline step, so the shopping agent does not have to call an LLM per catalog
item or rediscover product facts at query time.

Example:
    python starter/precompute.py --limit 5 --progress-every 1
    python starter/precompute.py --output data/clean_catalog.jsonl

Ollama must be running locally.  The default llama3.2:3b model works on CPU;
set --model or OLLAMA_MODEL to use another installed Ollama model.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from starter.semantic_index import concepts_for_item, file_sha256
except ModuleNotFoundError:  # Supports `python starter/precompute.py` as documented.
    from semantic_index import concepts_for_item, file_sha256


# ``null`` is valid in the Agent API as the absence of an attribute, but is not
# a useful JSON object key.  Unclassifiable evidence belongs under ``other``.
ATTRIBUTE_KEYS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
EXTRACTED_KEYS = tuple(key for key in ATTRIBUTE_KEYS if key != "category")


def _strings(value: Any) -> list[str]:
    """Flatten catalog values into compact, human-readable strings."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value != "" else []


def _compact_record(product: dict[str, Any], max_chars: int = 5_000) -> str:
    """Keep the prompt bounded while retaining the fields useful for attributes."""
    fields = (
        ("title", product.get("title")),
        ("categories", product.get("categories", product.get("category"))),
        ("brand/store", product.get("store")),
        ("price", product.get("price")),
        ("features", product.get("features")),
        ("details", product.get("details")),
        ("description", product.get("description")),
    )
    parts: list[str] = []
    remaining = max_chars
    for name, value in fields:
        values = _strings(value)
        if not values or remaining <= 0:
            continue
        text = f"{name}: " + " | ".join(values)
        clipped = text[:remaining]
        parts.append(clipped)
        remaining -= len(clipped) + 1
    return "\n".join(parts)


def _prompt(product: dict[str, Any]) -> str:
    return f"""You are building a trustworthy product-search index.
Extract short product attributes from the supplied catalog record.

Use only these attribute keys: {", ".join(EXTRACTED_KEYS)}.
Return JSON only: an object mapping each applicable key to a list of short
evidence phrases. Omit inapplicable keys. Do not include category; it is
preserved separately from the catalog. Use `other` only for useful evidence
that fits none of the named keys.

Rules:
- Select and extract only facts supported by the record; never guess.
- Preserve important wording (for example, material names, color names, fit,
  brand, price, or use case) instead of writing a marketing summary.
- Each phrase must be at most 80 characters. At most 6 phrases per key.
- Do not output product IDs, explanations, markdown, or a `null` key.

Catalog record:
{_compact_record(product)}
"""


def _ollama_extract(product: dict[str, Any], model: str, url: str, timeout: int) -> dict[str, Any]:
    """Run one constrained extraction call against the local Ollama server."""
    payload = json.dumps({
        "model": model,
        "prompt": _prompt(product),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        answer = json.loads(envelope["response"])
    except (HTTPError, URLError, OSError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama extraction failed for {product.get('parent_asin', '<unknown>')}: {exc}"
        ) from exc
    if not isinstance(answer, dict):
        raise RuntimeError("Ollama returned JSON that was not an object")
    return answer


def _clean_feature_values(value: Any) -> list[str]:
    candidates = value if isinstance(value, list) else [value]
    values: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = " ".join(candidate.split()).strip(" -,:;.")[:80]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            values.append(cleaned)
        if len(values) == 6:
            break
    return values


def clean_product(product: dict[str, Any], model: str, url: str, timeout: int) -> dict[str, Any]:
    """Produce the flat catalog_attributes schema for one original product."""
    extracted = _ollama_extract(product, model, url, timeout)
    category = product.get("categories", product.get("category"))
    # Do not let the LLM rewrite the catalog category. It remains only in the
    # top-level field, rather than being duplicated in the extracted features.
    features: dict[str, list[str]] = {}
    for key in EXTRACTED_KEYS:
        values = _clean_feature_values(extracted.get(key))
        if values:
            features[key] = values
    def first(key: str, default: Any = "unknown") -> Any:
        values = features.get(key)
        return ", ".join(values) if values else default

    return {
        "parent_asin": product["parent_asin"],
        "title": product.get("title"),
        "category": category,
        "materials": first("material"),
        "color": first("color"),
        "size": first("size"),
        "style": first("style"),
        "brand": first("brand"),
        "budget_price": first("budget"),
        "feature": first("feature"),
        "use_case": first("use_case"),
        "other": first("other"),
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
    }


def _existing_ids(output_path: Path) -> set[str]:
    """Support safely continuing a long run without duplicating products."""
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8") as handle:
        return {
            str(item["parent_asin"])
            for line in handle
            if line.strip()
            for item in [json.loads(line)]
            if "parent_asin" in item
        }


def build_embedding_index(
    input_path: Path,
    output_path: Path,
    model: str,
    url: str,
    timeout: int,
    batch_size: int,
) -> None:
    """Create or resume an exact concept-vector index for runtime reranking.

    The exact concept strings used by ``Agent`` are embedded once globally.
    The SQLite file remains valid only for this source catalog and model.
    """
    source_hash = file_sha256(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_path)
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS concepts (concept TEXT PRIMARY KEY, embedding BLOB, dimensions INTEGER)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS product_concepts ("
        "parent_asin TEXT NOT NULL, position INTEGER NOT NULL, concept TEXT NOT NULL, "
        "PRIMARY KEY (parent_asin, position))"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS product_concepts_asin ON product_concepts(parent_asin)")
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    expected = {"source_sha256": source_hash, "embedding_model": model}
    if metadata and any(metadata.get(key) != value for key, value in expected.items()):
        connection.close()
        raise RuntimeError(
            f"Embedding index {output_path} was built for a different catalog or model. "
            "Choose a new --embedding-output path."
        )
    if not metadata:
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())

    indexed_products = 0
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            parent_asin = str(item.get("parent_asin", ""))
            if not parent_asin:
                continue
            concepts = concepts_for_item(item)
            connection.executemany(
                "INSERT OR IGNORE INTO concepts(concept) VALUES (?)", ((concept,) for concept in concepts)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO product_concepts(parent_asin, position, concept) VALUES (?, ?, ?)",
                ((parent_asin, position, concept) for position, concept in enumerate(concepts)),
            )
            indexed_products += 1
            if indexed_products % 1000 == 0:
                connection.commit()
    connection.commit()

    completed = int(connection.execute("SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL").fetchone()[0])
    total = int(connection.execute("SELECT COUNT(*) FROM concepts").fetchone()[0])
    print(f"Catalog indexed: {indexed_products} products, {total} unique concepts ({completed} embedded).")
    while True:
        rows = connection.execute(
            "SELECT concept FROM concepts WHERE embedding IS NULL ORDER BY concept LIMIT ?", (batch_size,)
        ).fetchall()
        if not rows:
            break
        concepts = [str(row[0]) for row in rows]
        payload = json.dumps({"model": model, "input": concepts}).encode("utf-8")
        request = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            vectors = envelope.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(concepts):
                raise ValueError("unexpected embedding response")
            packed: list[tuple[bytes, int, str]] = []
            for concept, vector in zip(concepts, vectors):
                if not isinstance(vector, list) or not vector:
                    raise ValueError("invalid embedding vector")
                values = [float(value) for value in vector]
                # Keep Python's IEEE-754 doubles exactly as parsed from the
                # Ollama JSON response; this cache must not quantize rankings.
                packed.append((struct.pack(f"<{len(values)}d", *values), len(values), concept))
            connection.executemany(
                "UPDATE concepts SET embedding = ?, dimensions = ? WHERE concept = ?", packed
            )
            connection.commit()
            completed += len(concepts)
            print(f"Embedded {completed}/{total} concepts.", flush=True)
        except (HTTPError, URLError, OSError, TimeoutError, TypeError, ValueError, json.JSONDecodeError) as exc:
            connection.close()
            raise RuntimeError(f"Embedding failed after {completed}/{total} concepts: {exc}") from exc
    connection.close()
    print(f"Done. Wrote semantic index to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an Ollama-enriched clean product catalog.")
    parser.add_argument("--input", default="data/catalog.jsonl", help="Source catalog JSONL path.")
    parser.add_argument("--output", default="data/clean_catalog.jsonl", help="Clean JSONL path to create.")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--timeout", type=int, default=120, help="Seconds allowed for one Ollama call.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N new products (0 means all).")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N products (0 disables it).")
    parser.add_argument("--resume", action="store_true", help="Append missing products to an existing output file.")
    parser.add_argument("--dry-run", action="store_true", help="Print one clean record without writing a file.")
    parser.add_argument(
        "--build-embeddings", action="store_true", help="Build/resume a persistent semantic index, then exit."
    )
    parser.add_argument(
        "--embedding-input", default="data/catalog_attributes.jsonl", help="Structured catalog JSONL to embed."
    )
    parser.add_argument(
        "--embedding-output", default="data/semantic_index.sqlite", help="Persistent semantic SQLite index."
    )
    parser.add_argument(
        "--embedding-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    )
    parser.add_argument("--embedding-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.limit < 0 or args.progress_every < 0 or args.timeout <= 0 or args.embedding_batch_size <= 0:
        parser.error("limits must be non-negative and timeouts/batch sizes must be positive")

    if args.build_embeddings:
        embedding_input = Path(args.embedding_input)
        if not embedding_input.exists():
            parser.error(f"Embedding input does not exist: {embedding_input}")
        build_embedding_index(
            embedding_input,
            Path(args.embedding_output),
            args.embedding_model,
            args.ollama_url.rstrip("/").replace("/api/generate", "/api/embed"),
            args.timeout,
            args.embedding_batch_size,
        )
        return

    source_path = Path(args.input)
    output_path = Path(args.output)
    if not source_path.exists():
        parser.error(f"Input catalog does not exist: {source_path}")
    if output_path.exists() and not args.resume and not args.dry_run:
        parser.error(f"Output already exists: {output_path}. Use --resume or choose another --output path.")

    completed = _existing_ids(output_path) if args.resume else set()
    processed = 0
    with source_path.open(encoding="utf-8") as source:
        writer = None
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = output_path.open("a" if args.resume else "x", encoding="utf-8")
        try:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                if str(product.get("parent_asin")) in completed:
                    continue
                clean = clean_product(product, args.model, args.ollama_url, args.timeout)
                if args.dry_run:
                    print(json.dumps(clean, ensure_ascii=False, indent=2))
                    return
                assert writer is not None
                writer.write(json.dumps(clean, ensure_ascii=False) + "\n")
                writer.flush()
                processed += 1
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"Processed {processed} products (source line {line_number}).", flush=True)
                if args.limit and processed >= args.limit:
                    break
        finally:
            if writer is not None:
                writer.close()
    print(f"Done. Wrote {processed} products to {output_path}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume to continue.", file=sys.stderr)
        raise SystemExit(130)
