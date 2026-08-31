"""Build the optional persistent concept-vector index for runtime reranking.

Structured attributes are built by :mod:`extract_product_attributes`; this
module only embeds those already-normalized concepts.

Example:
    python -m submission.src.precompute
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .semantic_index import concepts_for_item, file_sha256
except ImportError:  # Supports direct script execution as a fallback.
    from semantic_index import concepts_for_item, file_sha256


def build_embedding_index(
    input_path: Path,
    output_path: Path,
    model: str,
    url: str,
    timeout: int,
    batch_size: int,
) -> int:
    """Create or resume an exact concept-vector index for runtime reranking.

    The exact concept strings used by ``Agent`` are embedded once globally.
    The SQLite file remains valid only for this source catalog and model.

    Returns:
        The total number of input tokens reported by Ollama for this run.
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
    embedding_tokens = 0
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
            try:
                embedding_tokens += max(0, int(envelope.get("prompt_eval_count", 0)))
            except (AttributeError, TypeError, ValueError):
                # Some Ollama-compatible servers omit usage metadata.
                pass
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
            raise RuntimeError(
                f"Embedding failed after {completed}/{total} concepts and "
                f"{embedding_tokens} embedding tokens: {exc}"
            ) from exc
    connection.close()
    print(f"Done. Wrote semantic index to {output_path}. Embedding tokens used: {embedding_tokens}.")
    return embedding_tokens


def _ollama_embed_url(url: str) -> str:
    """Accept either an Ollama base URL or a legacy generate/embed endpoint."""
    base = url.rstrip("/")
    if base.endswith("/api/embed"):
        return base
    if base.endswith("/api/generate"):
        base = base.removesuffix("/api/generate")
    return f"{base}/api/embed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds allowed for one Ollama call.")
    parser.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--embedding-input",
        default="submission/assets/catalog_attributes.jsonl",
        help="Structured catalog JSONL to embed.",
    )
    parser.add_argument(
        "--embedding-output",
        default="submission/assets/semantic_index.sqlite",
        help="Persistent semantic SQLite index.",
    )
    parser.add_argument(
        "--embedding-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    )
    parser.add_argument("--embedding-batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.timeout <= 0 or args.embedding_batch_size <= 0:
        parser.error("timeout and batch size must be positive")
    embedding_input = Path(args.embedding_input)
    if not embedding_input.exists():
        parser.error(f"Embedding input does not exist: {embedding_input}")
    build_embedding_index(
        embedding_input,
        Path(args.embedding_output),
        args.embedding_model,
        _ollama_embed_url(args.ollama_url),
        args.timeout,
        args.embedding_batch_size,
    )


if __name__ == "__main__":
    main()
