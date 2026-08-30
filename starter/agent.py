from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from starter.question_selection import (
    ASKABLE_ATTRIBUTES,
    choose_next_question,
    gain_ratio_multilabel_missing,
    ground_answer,
    normalize_attributes,
)
from starter.semantic_index import concepts_for_item, file_sha256


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
DEFAULT_BM25_WEIGHTS = (0.0, 4.5, 4.0, 2.5, 2.5, 1.5, 1.0)


def _bm25_expression() -> str:
    """Build the FTS5 BM25 expression, optionally from an environment override."""
    raw_weights = os.environ.get("BM25_WEIGHTS")
    if not raw_weights:
        weights = DEFAULT_BM25_WEIGHTS
    else:
        try:
            parsed = tuple(float(value.strip()) for value in raw_weights.split(","))
            if len(parsed) != len(DEFAULT_BM25_WEIGHTS) or any(value < 0 for value in parsed):
                raise ValueError
            weights = parsed
        except ValueError as exc:
            raise ValueError(
                "BM25_WEIGHTS must contain seven non-negative comma-separated numbers"
            ) from exc
    return "bm25(products, " + ", ".join(str(value) for value in weights) + ")"


BM25 = _bm25_expression()
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
MISSING_CONCEPT_VALUES = {
    "unknown", "n/a", "none", "not available", "not specified", "unspecified",
}
DEFAULT_SEMANTIC_CANDIDATES = 50
SEMANTIC_CANDIDATES = int(
    os.environ.get("SEMANTIC_CANDIDATES", str(DEFAULT_SEMANTIC_CANDIDATES))
)
DEFAULT_OVERRIDE_CANDIDATES = 150
OVERRIDE_CANDIDATES = int(
    os.environ.get("OVERRIDE_CANDIDATES", str(DEFAULT_OVERRIDE_CANDIDATES))
)
ENTROPY_POOL_SIZE = 20
MAX_TURNS = 10
MAX_CANDIDATE_CONCEPTS = 60
MAX_QUERY_CONCEPTS = 8
MAX_DOCUMENT_CONCEPTS = 16
EMBED_BATCH_SIZE = 96
# A semantic context score is only trusted when it separates the best candidate
# from the rest. This prevents a weak or generic context from reshuffling a
# useful lexical ranking.
SEMANTIC_CONFIDENCE_MARGIN = float(os.environ.get("SEMANTIC_CONFIDENCE_MARGIN", "0.015"))
SEMANTIC_BLEND_WEIGHT = float(os.environ.get("SEMANTIC_BLEND_WEIGHT", "0.85"))
# Catalog-constrained LLM concepts can expand the shopper's wording, but must
# not replace it.  The direct conversation remains the primary semantic signal.
DEFAULT_SEMANTIC_EXPANSION_WEIGHT = 0.20
SEMANTIC_EXPANSION_WEIGHT = float(
    os.environ.get("SEMANTIC_EXPANSION_WEIGHT", str(DEFAULT_SEMANTIC_EXPANSION_WEIGHT))
)
SEMANTIC_REQUIREMENT_WEIGHT = 3.0
SEMANTIC_CONFIRMED_WEIGHT = 2.0
HARD_CONSTRAINT_BONUS = 1.40
HARD_CONSTRAINT_PENALTY = 2.20
QUESTION_CONFIDENCE_TEMPERATURE = 0.85
# Product ratings are deliberately a light, post-retrieval signal.  They must
# never determine which concepts are embedded or which documents BM25 finds.
RATING_PRIOR = 4.0
RATING_FULL_CONFIDENCE_REVIEWS = 100
NO_PREFERENCE_RE = re.compile(
    r"\b(?:no preference|don't have (?:(?:an?|any) )?(?:additional )?preference|"
    r"do not have (?:(?:an?|any) )?(?:additional )?preference|don't care|doesn't matter|"
    r"does not matter|any (?:is fine|will do)|use your judgment)\b",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:please\s+)?ignore\s+(?:my\s+|the\s+)?(?:earlier|previous|prior)\s+"
    r"(?:preference|requirement|constraint)s?\b",
    re.IGNORECASE,
)
OVERRIDE_VALUE_RE = re.compile(
    r"\b(?:what\s+i\s+need\s+is|what\s+i\s+want\s+is|instead\s+i\s+"
    r"(?:need|want)|i\s+(?:need|want)\s+instead)\s*[:,-]?\s*(.+)$",
    re.IGNORECASE,
)
OPENING_CATEGORY_RE = re.compile(
    r"\bi['\u2019]?m\s+looking\s+for\s+(.+?)(?:[.!?]|,\s*(?:but|and)\b)",
    re.IGNORECASE,
)
KEY_REQUIREMENT_RE = re.compile(
    r"\b(?:a\s+)?key\s+requirement\s+is\s*:\s*(.+)$|\bmust\s+have\s*:\s*(.+)$",
    re.IGNORECASE,
)
DIRECT_REQUIREMENT_RE = re.compile(
    r"\b(?:what\s+matters\s+is|what\s+i\s+(?:need|want)\s+is|must\s+have)\s*[:,-]?\s*(.+)$",
    re.IGNORECASE,
)
EXPLICIT_ATTRIBUTE_RE = re.compile(
    r"^\s*(category|materials?|colou?r|sizes?|style|brand|budget|price|features?|"
    r"use[ _-]?cases?|other)\s*:\s*(.+)$",
    re.IGNORECASE,
)
ATTRIBUTE_ALIASES = {
    "materials": "material",
    "colour": "color",
    "sizes": "size",
    "price": "budget",
    "features": "feature",
    "use case": "use_case",
    "use-case": "use_case",
    "use cases": "use_case",
    "use-cases": "use_case",
}
ATTRIBUTE_HINTS = {
    "material": re.compile(
        r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|"
        r"synthetic|textile|rubber|alloy|steel|metal|linen|denim)\b",
        re.IGNORECASE,
    ),
    "color": re.compile(
        r"\b(?:colou?r|black|white|blue|red|pink|green|brown|gr[ae]y|purple|"
        r"yellow|orange|gold|silver)\b",
        re.IGNORECASE,
    ),
    "size": re.compile(r"\b(?:size|sizing|width|wide|narrow)\b", re.IGNORECASE),
    "style": re.compile(
        r"\b(?:department|style|fit|fitted|loose|sleeve|neck|casual|formal)\b",
        re.IGNORECASE,
    ),
    "use_case": re.compile(
        r"\b(?:hiking|running|gym|winter|outdoor|work|wedding|party|travel)\b",
        re.IGNORECASE,
    ),
    "budget": re.compile(r"(?:\bbudget\b|\bprice\b|\bunder\s+\$?\d|\$\s*\d)", re.IGNORECASE),
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _informative_concept(concept: str) -> bool:
    """Exclude missing-value placeholders from semantic retrieval signals."""
    _, separator, value = concept.partition(":")
    return not separator or value.strip().casefold() not in MISSING_CONCEPT_VALUES


def _attribute_value_matches(have: str, wanted: str) -> bool:
    """Match a normalized value while allowing qualified catalog wording."""
    have_key = have.casefold().strip()
    wanted_key = wanted.casefold().strip()
    if have_key == wanted_key:
        return True
    have_terms = set(_terms(have_key))
    wanted_terms = set(_terms(wanted_key))
    return bool(
        have_terms
        and wanted_terms
        and (have_terms <= wanted_terms or wanted_terms <= have_terms)
    )


def _z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation < 1e-9:
        return [0.0] * len(values)
    return [(value - mean) / deviation for value in values]


def _unit(vector: list[float]) -> list[float]:
    """Normalize an embedding once so semantic comparisons become cheap dots."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * len(vector)


def _finite_number(value: object) -> float | None:
    """Return a finite numeric catalog value, accepting numeric strings."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _jsonl(path: Path):
    """Yield non-empty JSONL records without duplicating file-loop plumbing."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class Agent:
    """Hybrid BM25 and local-Ollama semantic search agent.

    The semantic stage follows SemRank section 3.2: BM25 retrieves a small
    candidate set; concepts from those documents constrain an LLM's query
    analysis; local embeddings then semantically rerank just those candidates.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self._sessions: dict[str, dict] = {}
        self.concepts_by_asin: dict[str, list[str]] = {}
        self._semantic_loaded_asins: set[str] = set()
        self._semantic_connection: sqlite3.Connection | None = None
        self.embedding_cache: dict[str, list[float]] = {}
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.llm_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
        self.ollama_timeout = int(os.environ.get("OLLAMA_TIMEOUT", "30"))
        self._llm_available = os.environ.get("OLLAMA_ENABLED", "1").lower() not in {"0", "false", "no"}
        self._embedding_available = self._llm_available
        self.semantic_rerank_enabled = (
            os.environ.get("SEMANTIC_RERANK_ENABLED", "1").lower() not in {"0", "false", "no"}
        )
        self.attributes_by_asin: dict[str, dict] = {}
        self.attribute_values: dict[str, set[str]] = {
            attribute: set() for attribute in ASKABLE_ATTRIBUTES
        }
        self.ratings_by_asin: dict[str, tuple[float, float]] = {}
        self.connection = self._open_fts_cache()
        self._load_semantic_index()
        self._load_attribute_index()

    def _open_fts_cache(self) -> sqlite3.Connection:
        """Open a catalog-fingerprinted FTS cache, building it only once."""
        configured = os.environ.get("CATALOG_FTS_PATH")
        cache_path = Path(configured) if configured else self.catalog_path.with_name("catalog_fts.sqlite")
        fingerprint = file_sha256(self.catalog_path)
        if cache_path.exists():
            try:
                cached = sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True)
                metadata = dict(cached.execute("SELECT key, value FROM metadata"))
                if metadata.get("source_sha256") == fingerprint:
                    return cached 
                cached.close()
            except (OSError, sqlite3.DatabaseError):
                pass

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"{cache_path.stem}.", suffix=".building", dir=cache_path.parent
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            building = sqlite3.connect(temporary_path)
            try:
                self._build_fts_cache(building)
                building.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                building.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (("source_sha256", fingerprint),),
                )
                building.commit()
            finally:
                building.close()
            os.replace(temporary_path, cache_path)
            return sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True)
        except (OSError, sqlite3.DatabaseError):
            # Generated caches are an optimization only. The original in-memory
            # path preserves correctness when a deployment is read-only.
            connection = sqlite3.connect(":memory:")
            self._build_fts_cache(connection)
            return connection

    def _build_fts_cache(self, connection: sqlite3.Connection) -> None:
        """Build the weighted FTS document table in bounded batches."""
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE VIRTUAL TABLE products USING fts5(parent_asin UNINDEXED, "
            f"{', '.join(SEARCH_FIELDS)}, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for product in _jsonl(self.catalog_path):
            batch.append((str(product["parent_asin"]), *(_text(product.get(key)) for key in SEARCH_FIELDS)))
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        connection.commit()

    def _load_semantic_index(self) -> None:
        """Load offline LLM features, if precompute.py has generated them."""
        configured = os.environ.get("CLEAN_CATALOG_PATH")
        if configured:
            clean_path = Path(configured)
        else:
            clean_path = self.catalog_path.with_name("catalog_attributes.jsonl")
            if not clean_path.exists():
                # Lazily create the flat semantic catalog on first use.  This
                # keeps normal runs fast while allowing a raw catalog to work
                # without a separate manual preprocessing command.
                precompute = Path(__file__).with_name("precompute.py")
                if precompute.exists():
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                str(precompute),
                                "--input", str(self.catalog_path),
                                "--output", str(clean_path),
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except (OSError, subprocess.SubprocessError):
                        pass
            
        if not clean_path.exists():
            return

        semantic_path = Path(os.environ.get("SEMANTIC_INDEX_PATH", clean_path.with_name("semantic_index.sqlite")))
        if semantic_path.exists():
            try:
                connection = sqlite3.connect(f"file:{semantic_path.resolve()}?mode=ro", uri=True)
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                valid = (
                    metadata.get("source_sha256") == file_sha256(clean_path)
                    and metadata.get("embedding_model") == self.embedding_model
                )
                if valid:
                    self._semantic_connection = connection
                    return
                connection.close()
            except (OSError, sqlite3.DatabaseError):
                # A missing/incompatible generated cache must never change
                # retrieval correctness; use the original dynamic path instead.
                pass

        for item in _jsonl(clean_path):
            parent_asin = str(item.get("parent_asin", ""))
            if parent_asin:
                self.concepts_by_asin[parent_asin] = concepts_for_item(item)
                self._semantic_loaded_asins.add(parent_asin)

    def _ensure_candidate_concepts(self, rows: list[tuple[str, float]]) -> None:
        """Load only current-candidate concepts from the persisted index."""
        if self._semantic_connection is None:
            return
        missing = [parent_asin for parent_asin, _ in rows if parent_asin not in self._semantic_loaded_asins]
        if not missing:
            return
        placeholders = ", ".join("?" for _ in missing)
        found: dict[str, list[str]] = {parent_asin: [] for parent_asin in missing}
        cursor = self._semantic_connection.execute(
            "SELECT parent_asin, concept FROM product_concepts "
            f"WHERE parent_asin IN ({placeholders}) ORDER BY parent_asin, position",
            missing,
        )
        for parent_asin, concept in cursor:
            found[str(parent_asin)].append(str(concept))
        self.concepts_by_asin.update(found)
        self._semantic_loaded_asins.update(missing)

    def _load_persisted_embeddings(self, texts: list[str]) -> None:
        if self._semantic_connection is None:
            return
        missing = list(dict.fromkeys(text for text in texts if text not in self.embedding_cache))
        if not missing:
            return
        placeholders = ", ".join("?" for _ in missing)
        cursor = self._semantic_connection.execute(
            "SELECT concept, embedding, dimensions FROM concepts "
            f"WHERE concept IN ({placeholders}) AND embedding IS NOT NULL",
            missing,
        )
        for concept, blob, dimensions in cursor:
            count = int(dimensions)
            if isinstance(blob, bytes) and len(blob) == 8 * count:
                self.embedding_cache[str(concept)] = list(struct.unpack(f"<{count}d", blob))

    def _load_attribute_index(self) -> None:
        """Load question attributes and rating metadata from the flat catalog."""
        configured = os.environ.get("CATALOG_ATTRIBUTES_PATH")
        path = Path(configured) if configured else self.catalog_path.with_name("catalog_attributes.jsonl")
        if not path.exists():
            return
        for row in _jsonl(path):
            parent_asin = str(row.get("parent_asin", ""))
            if not parent_asin:
                continue
            attributes = normalize_attributes(row)
            self.attributes_by_asin[parent_asin] = attributes
            for attribute in ASKABLE_ATTRIBUTES:
                self.attribute_values[attribute].update(attributes.get(attribute) or [])
            rating = _finite_number(row.get("average_rating"))
            review_count = _finite_number(row.get("rating_number"))
            if rating is not None and 0.0 <= rating <= 5.0:
                self.ratings_by_asin[parent_asin] = (rating, max(0.0, review_count or 0.0))

    def _ollama_request(self, endpoint: str, payload: dict) -> dict:
        """POST JSON to Ollama and validate the response envelope."""
        request = Request(
            f"{self.ollama_url.rstrip('/')}/api/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.ollama_timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        if not isinstance(envelope, dict):
            raise ValueError("unexpected Ollama response")
        return envelope

    def _ollama_generate(self, prompt: str) -> tuple[dict | None, dict]:
        if not self._llm_available:
            return None, {"prompt_tokens": 0, "completion_tokens": 0}
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            envelope = self._ollama_request("generate", payload)
            answer = json.loads(envelope.get("response", "{}"))
            if not isinstance(answer, dict):
                return None, {"prompt_tokens": 0, "completion_tokens": 0}
            return answer, {
                "prompt_tokens": max(0, int(envelope.get("prompt_eval_count", 0))),
                "completion_tokens": max(0, int(envelope.get("eval_count", 0))),
            }
        except (HTTPError, URLError, OSError, TimeoutError, TypeError, ValueError, json.JSONDecodeError):
            self._llm_available = False
            return None, {"prompt_tokens": 0, "completion_tokens": 0}

    def _embed(self, texts: list[str]) -> dict[str, list[float]] | None:
        """Resolve persisted vectors first, embedding only genuinely new text."""
        self._load_persisted_embeddings(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in self.embedding_cache))
        if not missing:
            return {text: self.embedding_cache[text] for text in texts if text in self.embedding_cache}
        if not self._embedding_available:
            return {text: self.embedding_cache[text] for text in texts if text in self.embedding_cache}
        try:
            for start in range(0, len(missing), EMBED_BATCH_SIZE):
                batch = missing[start:start + EMBED_BATCH_SIZE]
                envelope = self._ollama_request(
                    "embed", {"model": self.embedding_model, "input": batch}
                )
                vectors = envelope.get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise ValueError("unexpected embedding response")
                for text, vector in zip(batch, vectors):
                    if not isinstance(vector, list) or not vector:
                        raise ValueError("invalid embedding vector")
                    self.embedding_cache[text] = [float(value) for value in vector]
        except (HTTPError, URLError, OSError, TimeoutError, TypeError, ValueError, json.JSONDecodeError):
            # The completion model may not support embeddings. BM25 still works.
            self._embedding_available = False
            return None
        return {text: self.embedding_cache[text] for text in texts if text in self.embedding_cache}

    def _candidate_concepts(self, rows: list[tuple[str, float]]) -> list[str]:
        """Pseudo-relevance feedback: frequent concepts from BM25's top hits."""
        counts: Counter[str] = Counter()
        for parent_asin, _ in rows:
            concepts = (
                concept
                for concept in self.concepts_by_asin.get(parent_asin, [])
                if _informative_concept(concept)
            )
            counts.update(dict.fromkeys(concepts, 1))
        return [concept for concept, _ in counts.most_common(MAX_CANDIDATE_CONCEPTS)]

    def _select_query_concepts(
        self,
        conversation: list[str],
        candidates: list[str],
    ) -> tuple[list[str], dict]:
        if not candidates:
            return [], {"prompt_tokens": 0, "completion_tokens": 0}
        prompt = f"""You are improving a clothing product search result.
Select the concepts that best express the shopper's current need.
Only select exact entries from Candidate concepts. Do not invent, paraphrase,
or repeat concepts. Select at most {MAX_QUERY_CONCEPTS}.

Return JSON only: {{"concepts": ["exact candidate concept"]}}

Conversation: {json.dumps(conversation[-6:], ensure_ascii=False)}
Candidate concepts: {json.dumps(candidates, ensure_ascii=False)}"""
        answer, usage = self._ollama_generate(prompt)
        if not answer:
            return [], usage
        canonical = {concept.casefold(): concept for concept in candidates}
        selected: list[str] = []
        for concept in answer.get("concepts", []):
            if isinstance(concept, str) and concept.casefold() in canonical:
                selected.append(canonical[concept.casefold()])
        selected = list(dict.fromkeys(selected))[:MAX_QUERY_CONCEPTS]
        return selected, usage

    @staticmethod
    def _direct_semantic_queries(session: dict) -> list[tuple[str, float]]:
        """Build weighted queries, prioritizing hard needs over stable category."""
        weighted: list[tuple[str, float]] = []
        for group in session["term_groups"]:
            text = group["text"]
            if not text:
                continue
            if group["kind"] == "category":
                weighted.append((f"category: {text}", 1.0))
            elif group["kind"] == "requirement":
                weighted.append((f"required shopper need: {text}", SEMANTIC_REQUIREMENT_WEIGHT))
            else:
                weighted.append((f"shopper request: {text}", 1.0))
        weighted = weighted[-6:]
        weighted.extend(
            (f"required {attribute}: {value}", SEMANTIC_CONFIRMED_WEIGHT)
            for attribute, values in session["confirmed"].items()
            for value in sorted(values)
        )
        deduplicated: dict[str, float] = {}
        for query, weight in weighted:
            deduplicated[query] = max(weight, deduplicated.get(query, 0.0))
        return list(deduplicated.items())

    def _semantic_scores(
        self,
        direct_queries: list[tuple[str, float]],
        query_concepts: list[str],
        rows: list[tuple[str, float]],
    ) -> list[float] | None:
        direct_queries = [(query, weight) for query, weight in direct_queries if weight > 0.0]
        if not direct_queries:
            return None
        direct_texts = [query for query, _ in direct_queries]
        document_concepts = list(dict.fromkeys(
            concept for parent_asin, _ in rows
            for concept in self.concepts_by_asin.get(parent_asin, [])
            if _informative_concept(concept)
        ))
        embeddings = self._embed([*direct_texts, *query_concepts, *document_concepts])
        if embeddings is None or any(query not in embeddings for query in direct_texts):
            return None
        # Normalizing once avoids recalculating two vector norms for every
        # query/document pair (thousands of pairs per rerank).
        vectors = {text: _unit(vector) for text, vector in embeddings.items()}
        scores: list[float] = []
        for parent_asin, _ in rows:
            concepts = [
                concept
                for concept in self.concepts_by_asin.get(parent_asin, [])
                if _informative_concept(concept) and concept in vectors
            ]
            if not concepts:
                scores.append(0.0)
                continue
            # The shopper's own words always determine most of the score. Each
            # direct query fragment matches its closest document concept.
            direct_maxima = [
                (
                    max(
                        sum(a * b for a, b in zip(vectors[query], vectors[document]))
                        for document in concepts
                    ),
                    weight,
                )
                for query, weight in direct_queries
            ]
            total_direct_weight = sum(weight for _, weight in direct_maxima)
            direct_score = (
                sum(similarity * weight for similarity, weight in direct_maxima)
                / total_direct_weight
            )

            available_expansions = [query for query in query_concepts if query in vectors]
            if not available_expansions:
                scores.append(direct_score)
                continue
            expansion_maxima = [
                max(sum(a * b for a, b in zip(vectors[query], vectors[document])) for document in concepts)
                for query in available_expansions
            ]
            expansion_score = sum(expansion_maxima) / len(expansion_maxima)
            scores.append(
                (1.0 - SEMANTIC_EXPANSION_WEIGHT) * direct_score
                + SEMANTIC_EXPANSION_WEIGHT * expansion_score
            )
        return scores

    @staticmethod
    def _rerank(rows: list[tuple[str, float]], scores: list[float]) -> list[tuple[str, float]]:
        """Sort candidates by replacement scores while keeping ASINs attached."""
        ranked = sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)
        return [(row[0], score) for score, row in ranked]

    @staticmethod
    def _semantic_is_confident(scores: list[float]) -> bool:
        """Whether retrieved context supplies a useful, non-generic ranking signal."""
        if len(scores) < 2:
            return bool(scores)
        ordered = sorted(scores, reverse=True)
        # Compare to the runner-up rather than an absolute score: embedding
        # models have different cosine scales, while a near-tie is consistently
        # unhelpful as a reranking signal.
        return ordered[0] - ordered[1] >= SEMANTIC_CONFIDENCE_MARGIN

    def _retrieve_candidates(
        self,
        expression: str,
        with_evidence: bool,
        candidate_limit: int = SEMANTIC_CANDIDATES,
        excluded_asins: set[str] | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """Run weighted BM25, omitting products already shown in this session."""
        selected_fields = f", {', '.join(SEARCH_FIELDS)}" if with_evidence else ""
        excluded = sorted(excluded_asins or ())
        exclusion_clause = ""
        parameters: list[object] = [expression]
        if excluded:
            exclusion_clause = f" AND parent_asin NOT IN ({', '.join('?' for _ in excluded)})"
            parameters.extend(excluded)
        parameters.append(candidate_limit)
        rows = self.connection.execute(
            f"SELECT parent_asin{selected_fields}, {BM25} FROM products "
            f"WHERE products MATCH ?{exclusion_clause} ORDER BY {BM25} LIMIT ?",
            parameters,
        ).fetchall()
        evidence: dict[str, str] = {}
        scored: list[tuple[str, float]] = []
        for parent_asin, *fields, score in rows:
            parent_asin = str(parent_asin)
            if with_evidence:
                evidence[parent_asin] = " ".join(_text(value)[:900] for value in fields if value)
            scored.append((parent_asin, -float(score)))  # SQLite BM25 is lower-is-better.
        return scored, evidence

    def _add_constraint(
        self,
        session: dict,
        *,
        text: str,
        attribute: str | None = None,
        values: set[str] | None = None,
    ) -> None:
        """Record a requirement separately from the broader FTS query terms."""
        terms = tuple(dict.fromkeys(_terms(text)))
        value_set = {value.casefold() for value in (values or set()) if value}
        if not terms and not value_set:
            return
        session["constraints"].append(
            {
                "terms": terms,
                "attribute": attribute,
                "values": value_set,
            }
        )

    def _apply_constraints(
        self,
        rows: list[tuple[str, float]],
        session: dict,
        evidence: dict[str, str],
        top_k: int,
    ) -> tuple[list[tuple[str, float]], dict[str, int]]:
        """Rerank by active requirements, then enforce grounded buying constraints."""
        constraints = session["constraints"]
        if not rows or not constraints:
            return rows, {"active_constraints": len(constraints), "typed_conflicts": 0}
        self._ensure_candidate_concepts(rows)
        base_scores = _z_scores([score for _, score in rows])
        rescored: list[tuple[str, float]] = []
        conflicts = 0
        for (parent_asin, _), base in zip(rows, base_scores):
            attrs = self.attributes_by_asin.get(parent_asin, {})
            searchable = " ".join(
                (
                    evidence.get(parent_asin, ""),
                    " ".join(self.concepts_by_asin.get(parent_asin, [])),
                    " ".join(
                        value
                        for values in attrs.values()
                        if isinstance(values, list)
                        for value in values
                    ),
                )
            )
            available = set(_terms(searchable))
            adjustment = 0.0
            for constraint in constraints:
                attribute = constraint["attribute"]
                wanted = constraint["values"]
                have = {str(value).casefold() for value in attrs.get(attribute, [])} if attribute else set()
                if have and wanted:
                    match = float(any(
                        _attribute_value_matches(have_value, wanted_value)
                        for have_value in have
                        for wanted_value in wanted
                    ))
                    contradicted = not match
                else:
                    terms = set(constraint["terms"])
                    match = len(terms & available) / len(terms) if terms else 0.0
                    contradicted = False
                adjustment += HARD_CONSTRAINT_BONUS * match
                if contradicted:
                    adjustment -= HARD_CONSTRAINT_PENALTY
                    conflicts += 1
            rescored.append((parent_asin, base + adjustment))
        rescored.sort(key=lambda row: row[1], reverse=True)

        # Missing metadata is allowed. Explicit conflicts sort last, or are
        # filtered for Buying only when enough matches remain to fill a page.
        confirmed = session["confirmed"]
        if confirmed:
            def conflicts_with_confirmed(row: tuple[str, float]) -> bool:
                attrs = self.attributes_by_asin.get(row[0], {})
                return any(
                    (have := set(attrs.get(attribute) or []))
                    and not any(
                        _attribute_value_matches(str(have_value), str(wanted_value))
                        for have_value in have
                        for wanted_value in wanted
                    )
                    for attribute, wanted in confirmed.items()
                )

            matched, rejected = [], []
            for row in rescored:
                (rejected if conflicts_with_confirmed(row) else matched).append(row)
            rescored = matched if session["route"] == "buying" and len(matched) >= top_k else matched + rejected
        return rescored, {"active_constraints": len(constraints), "typed_conflicts": conflicts}

    def _entropy_attribute(
        self,
        rows: list[tuple[str, float]],
        exhausted: set[str],
        turn: int,
    ) -> tuple[str | None, float]:
        """Choose the question with the greatest expected posterior rank reduction."""
        if turn >= MAX_TURNS or not self.attributes_by_asin:
            return None, 0.0
        candidate_rows = [row for row in rows[:ENTROPY_POOL_SIZE] if row[0] in self.attributes_by_asin]
        pool = [self.attributes_by_asin[parent_asin] for parent_asin, _ in candidate_rows]
        if not pool:
            return None, 0.0

        # The existing entropy score measures how cleanly an attribute splits
        # candidates. Weight it by a posterior over the current ranking to ask
        # questions that are likely to move a high-ranked recommendation.
        rank_scores = _z_scores([score for _, score in candidate_rows])
        weights = [math.exp(score / QUESTION_CONFIDENCE_TEMPERATURE) for score in rank_scores]
        total_weight = sum(weights) or 1.0
        ambiguity = 1.0 - max(weights, default=0.0) / total_weight
        utilities: dict[str, float] = {}
        for attribute in ASKABLE_ATTRIBUTES:
            if attribute in exhausted:
                continue
            value_mass: Counter[str] = Counter()
            known_weight = 0.0
            for item, weight in zip(pool, weights):
                values = item.get(attribute) or []
                if values:
                    known_weight += weight
                    value_mass.update({str(value): weight for value in set(values)})
            incidence = sum(value_mass.values())
            if not incidence:
                continue
            split = 1.0 - sum((mass / incidence) ** 2 for mass in value_mass.values())
            coverage = known_weight / total_weight
            gain_ratio = gain_ratio_multilabel_missing(pool, attribute)
            # The first term is expected mass removed by a useful answer; the
            # second carries the multi-label / cardinality / missing-value
            # safeguards from question_selection.py.
            utilities[attribute] = ambiguity * coverage * split + 0.35 * gain_ratio

        if not utilities:
            return choose_next_question(pool, exhausted), 0.0
        attribute, utility = max(utilities.items(), key=lambda item: item[1])
        # Preserve the wildcard escape hatch when structured attributes have no
        # plausible value-of-information, but do not ask it after it was spent.
        if utility < 0.005:
            return choose_next_question(pool, exhausted), utility
        return attribute, utility

    def _ground_answer(self, session: dict, answer: str, attribute: str) -> bool:
        """Map a free-text answer to curated catalog values and record the slot."""
        if not self.attributes_by_asin or not session["last_candidates"]:
            return False
        pool = [
            self.attributes_by_asin[parent_asin]
            for parent_asin in session["last_candidates"]
            if parent_asin in self.attributes_by_asin
        ]
        if not pool:
            return False
        embed_fn = self._embed if self._embedding_available or self._semantic_connection is not None else None
        grounded = ground_answer(answer, attribute, pool, embed_fn=embed_fn)
        changed = False
        for slot, values in grounded.items():
            prior = session["confirmed"].setdefault(slot, set())
            new_values = set(values) - prior
            if new_values:
                prior.update(new_values)
                self._add_constraint(
                    session,
                    text=" ".join(sorted(new_values)),
                    attribute=slot,
                    values=new_values,
                )
                changed = True
        return changed

    @staticmethod
    def _detect_route(first_message: str) -> str:
        """Buying vs Browsing routing from the opening customer message.

        The scenario type is never disclosed to the agent, so it is inferred
        from the simulator's opening phrasing (evaluator.local_evaluator
        .initial_message). Buying discloses "A key requirement is:"; Browsing
        says "still exploring". Anything ambiguous (including Intent Override,
        which may swap constraints) stays on the safer Browsing path.
        """
        lowered = first_message.casefold()
        return "buying" if "key requirement" in lowered or "must have" in lowered else "browsing"

    @staticmethod
    def _rating_weight(profile: dict) -> float:
        """Translate the aggregate rating habit into a deliberately small boost.

        A critical shopper's low prior ratings mean that they are discerning,
        not that they want low-rated products.  Positive shoppers retain a
        near-zero weight so relevance remains the deciding factor.
        """
        style = str(profile.get("rating_style", "")).casefold()
        prior_rating = _finite_number(profile.get("average_prior_rating"))
        if style == "critical":
            return 0.65 if prior_rating is not None and prior_rating <= 2.0 else 0.50
        if style == "mixed":
            return 0.12
        if style in {"usually positive", "positive"}:
            return 0.03
        return 0.0

    def _rating_quality(self, parent_asin: str) -> float:
        """Produce a bounded, review-count-smoothed quality score in [0, 1]."""
        rating_data = self.ratings_by_asin.get(parent_asin)
        if rating_data is None:
            return 0.5  # Missing metadata is neutral rather than a penalty.
        rating, review_count = rating_data
        confidence = min(
            1.0,
            math.log1p(review_count) / math.log1p(RATING_FULL_CONFIDENCE_REVIEWS),
        )
        smoothed = RATING_PRIOR + (rating - RATING_PRIOR) * confidence
        return max(0.0, min(1.0, (smoothed - 3.5) / 1.5))

    def _apply_profile_rating(self, rows: list[tuple[str, float]], profile: dict) -> list[tuple[str, float]]:
        """Rerank retrieved products using rating sensitivity, without retrieval IO.

        ``rows`` are already BM25/semantic candidates.  This method only
        changes their order, and receives no embedding inputs, so an ongoing
        offline embedding build remains valid and uninterrupted.
        """
        weight = self._rating_weight(profile)
        if weight == 0.0 or not rows:
            return rows
        relevance = _z_scores([score for _, score in rows])
        combined = [
            score + weight * self._rating_quality(parent_asin)
            for (parent_asin, _), score in zip(rows, relevance)
        ]
        return self._rerank(rows, combined)

    @staticmethod
    def _refresh_active_terms(session: dict) -> None:
        """Rebuild the deduplicated, bounded FTS query from retained turns."""
        terms: list[str] = []
        seen: set[str] = set()
        for group in session["term_groups"]:
            for term in group["terms"]:
                if term not in seen and len(terms) < 40:
                    seen.add(term)
                    terms.append(term)
        session["active_terms"] = terms

    def _infer_override_constraint(self, text: str) -> tuple[str | None, set[str]]:
        """Map an explicit replacement need to a canonical catalog attribute/value."""
        explicit = EXPLICIT_ATTRIBUTE_RE.match(text)
        if explicit:
            raw_attribute, value_text = explicit.groups()
            normalized_attribute = raw_attribute.casefold().replace("_", " ")
            attribute = ATTRIBUTE_ALIASES.get(
                normalized_attribute,
                normalized_attribute.rstrip("s").replace(" ", "_"),
            )
            candidate_attributes = (attribute,) if attribute in self.attribute_values else ()
        else:
            value_text = text
            hinted = [
                attribute
                for attribute, pattern in ATTRIBUTE_HINTS.items()
                if pattern.search(value_text)
            ]
            remaining = [
                attribute
                for attribute in ASKABLE_ATTRIBUTES
                if attribute not in hinted and attribute not in {"category", "brand", "budget"}
            ]
            candidate_attributes = (*hinted, *remaining)

        query_key = " ".join(value_text.casefold().split()).strip(" -,:;.")
        query_terms = set(_terms(value_text))
        if not query_key or not query_terms:
            return (candidate_attributes[0] if candidate_attributes else None), set()

        best_attribute: str | None = None
        best_score = 0.0
        best_values: set[str] = set()
        hinted_attributes = {
            attribute
            for attribute, pattern in ATTRIBUTE_HINTS.items()
            if pattern.search(value_text)
        }
        for attribute in candidate_attributes:
            attribute_best = 0.0
            attribute_values: set[str] = set()
            for value in self.attribute_values.get(attribute, set()):
                value_key = " ".join(value.casefold().split()).strip(" -,:;.")
                value_terms = set(_terms(value))
                if not value_key or not value_terms:
                    continue
                if value_key == query_key:
                    score = 1.0
                elif value_terms <= query_terms or query_terms <= value_terms:
                    if all(term.isdigit() for term in value_terms):
                        continue
                    score = 0.90
                else:
                    overlap = len(query_terms & value_terms)
                    if not overlap:
                        continue
                    query_coverage = overlap / len(query_terms)
                    value_coverage = overlap / len(value_terms)
                    if min(query_coverage, value_coverage) < 0.75:
                        continue
                    score = 0.55 + 0.35 * min(query_coverage, value_coverage)
                if attribute in hinted_attributes:
                    score += 0.15
                if score > attribute_best + 1e-9:
                    attribute_best = score
                    attribute_values = {value.casefold()}
                elif abs(score - attribute_best) <= 1e-9:
                    attribute_values.add(value.casefold())
            if attribute_best > best_score + 1e-9:
                best_attribute = attribute
                best_score = attribute_best
                best_values = attribute_values

        if best_score < 0.70:
            fallback = candidate_attributes[0] if explicit and candidate_attributes else None
            return fallback, set()
        return best_attribute, set(sorted(best_values)[:6])

    def _record_message_terms(self, session: dict, message: str) -> bool:
        """Add a turn to retrieval state, retiring superseded preferences."""
        is_override = bool(OVERRIDE_RE.search(message))
        if is_override:
            # Keep the category disclosed in the opening turn, but retire every
            # prior positive preference before rebuilding retrieval state. A
            # prior "no preference" remains valid unless this override supplies
            # a value for that same attribute.
            session["term_groups"] = [
                group for group in session["term_groups"] if group["kind"] == "category"
            ]
            session["constraints"].clear()
            session["confirmed"].clear()
            session["asked_attributes"].clear()
            session["pending_attribute"] = None
            value = OVERRIDE_VALUE_RE.search(message)
            text = value.group(1) if value else message
            terms = _terms(text)
            session["term_groups"].append({"text": text, "terms": terms, "kind": "requirement"})
            attribute, values = self._infer_override_constraint(text)
            if attribute is not None:
                session["no_preference_attributes"].discard(attribute)
            if attribute is not None and values:
                session["confirmed"][attribute] = set(values)
                self._add_constraint(
                    session,
                    text=text,
                    attribute=attribute,
                    values=values,
                )
            else:
                self._add_constraint(session, text=text)
        elif not session["term_groups"]:
            # The opening message carries both the stable category and an
            # optional preference. Preserve only the category on an override.
            category = OPENING_CATEGORY_RE.search(message)
            category_terms = _terms(category.group(1)) if category else []
            all_terms = _terms(message)
            if category_terms:
                session["term_groups"].append(
                    {"text": category.group(1), "terms": category_terms, "kind": "category"}
                )
                category_set = set(category_terms)
                preference_terms = [term for term in all_terms if term not in category_set]
            else:
                preference_terms = all_terms
            if preference_terms:
                session["term_groups"].append(
                    {"text": message, "terms": preference_terms, "kind": "preference"}
                )
            direct = KEY_REQUIREMENT_RE.search(message)
            if direct:
                self._add_constraint(session, text=next(value for value in direct.groups() if value))
        else:
            session["term_groups"].append(
                {"text": message, "terms": _terms(message), "kind": "preference"}
            )
            direct = DIRECT_REQUIREMENT_RE.search(message)
            if direct:
                self._add_constraint(session, text=direct.group(1))
        self._refresh_active_terms(session)
        return is_override

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile,
            "active_terms": [],
            "term_groups": [],
            "asked_attributes": set(),
            "no_preference_attributes": set(),
            "pending_attribute": None,
            "confirmed": {},
            "constraints": [],
            "route": None,
            "shown_parent_asins": set(),
            "last_candidates": [],
            "last_rows": [],
            "last_evidence": {},
            "diagnostics": [],
        }

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]
        shown_parent_asins: set[str] = session["shown_parent_asins"]
        if session["route"] is None:
            session["route"] = self._detect_route(user_message)
        pending_attribute = session["pending_attribute"]
        is_override_message = bool(OVERRIDE_RE.search(user_message))
        if is_override_message:
            # The evaluator does not score recommendations shown before an
            # intent override. Start a new exclusion window so those products,
            # including the eventual target, may be recommended again under
            # the replacement intent.
            shown_parent_asins.clear()
        no_preference_reply = bool(
            not is_override_message and pending_attribute and NO_PREFERENCE_RE.search(user_message)
        )
        grounded_changed = False
        if no_preference_reply:
            session["no_preference_attributes"].add(pending_attribute)
        elif pending_attribute and not is_override_message:
            # An override answers the shopper's old question only accidentally;
            # it must reset state before any old slot can be confirmed.
            grounded_changed = self._ground_answer(session, user_message, pending_attribute)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        evidence: dict[str, str]
        diagnostics = {
            "turn": turn,
            "lexical_candidates": 0,
            "retrieval_limit": 0,
            "semantic_requested": False,
            "semantic_used": False,
            "semantic_confident": False,
            "grounded_changed": grounded_changed,
            "override": is_override_message,
        }
        if no_preference_reply and session["last_rows"]:
            # A declined attribute adds no retrieval evidence. Keep the prior
            # ranking and do not pollute the FTS query with the refusal text.
            # Products previously displayed are treated as rejected: the
            # evaluator would already have ended the conversation otherwise.
            rows = [row for row in session["last_rows"] if row[0] not in shown_parent_asins]
            evidence = {
                parent_asin: value
                for parent_asin, value in session["last_evidence"].items()
                if parent_asin not in shown_parent_asins
            }
            if len(rows) < top_k and session["active_terms"]:
                expression = " OR ".join(f'"{term}"' for term in session["active_terms"])
                rows, evidence = self._retrieve_candidates(
                    expression,
                    bool(session["constraints"]),
                    SEMANTIC_CANDIDATES,
                    shown_parent_asins,
                )
                rows = self._apply_profile_rating(rows, session["profile"])
                rows, constraint_diagnostics = self._apply_constraints(rows, session, evidence, top_k)
                diagnostics["lexical_candidates"] = len(rows)
                diagnostics.update(constraint_diagnostics)
        else:
            is_override_message = self._record_message_terms(session, user_message)
            diagnostics["override"] = is_override_message
            expression = " OR ".join(f'"{term}"' for term in session["active_terms"])
            if not expression:
                rows: list[tuple[str, float]] = []
                evidence = {}
            else:
                candidate_limit = (
                    OVERRIDE_CANDIDATES if is_override_message else SEMANTIC_CANDIDATES
                )
                diagnostics["retrieval_limit"] = candidate_limit
                rows, evidence = self._retrieve_candidates(
                    expression,
                    bool(session["constraints"]),
                    candidate_limit,
                    shown_parent_asins,
                )
            diagnostics["lexical_candidates"] = len(rows)

            # Retrieval context is used for the initial view and whenever the
            # conversation adds evidence. This includes Buying turns and the
            # first response after an Intent Override, not just early Browsing.
            should_semantic_rerank = (
                self.semantic_rerank_enabled
                and bool(rows)
                and (turn <= 2 or grounded_changed or is_override_message or bool(_terms(user_message)))
            )
            diagnostics["semantic_requested"] = should_semantic_rerank
            if should_semantic_rerank:
                self._ensure_candidate_concepts(rows)
                candidate_concepts = self._candidate_concepts(rows)
                conversation = [group["text"] for group in session["term_groups"]]
                direct_queries = self._direct_semantic_queries(session)
                query_concepts: list[str] = []
                if candidate_concepts and SEMANTIC_EXPANSION_WEIGHT > 0.0:
                    query_concepts, usage = self._select_query_concepts(conversation, candidate_concepts)
                diagnostics["semantic_direct_queries"] = len(direct_queries)
                diagnostics["semantic_expansions"] = len(query_concepts)
                semantic_scores = self._semantic_scores(direct_queries, query_concepts, rows)
                if semantic_scores is not None:
                    diagnostics["semantic_confident"] = self._semantic_is_confident(semantic_scores)
                    if diagnostics["semantic_confident"]:
                        base_scores = _z_scores([score for _, score in rows])
                        combined = [
                            base + SEMANTIC_BLEND_WEIGHT * semantic
                            for base, semantic in zip(base_scores, _z_scores(semantic_scores))
                        ]
                        rows = self._rerank(rows, combined)
                        diagnostics["semantic_used"] = True
            rows = self._apply_profile_rating(rows, session["profile"])
            rows, constraint_diagnostics = self._apply_constraints(rows, session, evidence, top_k)
            diagnostics.update(constraint_diagnostics)
        session["last_candidates"] = [parent_asin for parent_asin, _ in rows]
        session["last_rows"] = list(rows)
        session["last_evidence"] = dict(evidence)

        exhausted = (
            session["asked_attributes"]
            | session["no_preference_attributes"]
            | set(session["confirmed"])
        )
        ask_attribute, question_utility = self._entropy_attribute(rows, exhausted, turn)
        if ask_attribute is not None:
            # Every attribute - "other" included - is asked at most once.
            session["asked_attributes"].add(ask_attribute)
        session["pending_attribute"] = ask_attribute
        diagnostics["question"] = ask_attribute
        diagnostics["question_utility"] = round(question_utility, 6)
        diagnostics["top_score_gap"] = round(rows[0][1] - rows[1][1], 6) if len(rows) > 1 else None
        session["diagnostics"].append(diagnostics)
        recommendations = [{"parent_asin": parent_asin} for parent_asin, _ in rows[:top_k]]
        shown_parent_asins.update(item["parent_asin"] for item in recommendations)
        if ask_attribute == "other":
            message = "Is there any other detail that matters for what you need?"
        elif ask_attribute:
            message = f"I found some close matches. Any preference on {ask_attribute.replace('_', ' ')}?"
        else:
            message = "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": usage,
        }

    def get_diagnostics(self, session_id: str) -> list[dict]:
        """Return read-only per-turn retrieval diagnostics for local analysis."""
        if session_id not in self._sessions:
            raise RuntimeError("unknown session")
        return [dict(item) for item in self._sessions[session_id]["diagnostics"]]
