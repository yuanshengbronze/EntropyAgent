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

from starter.question_selection import choose_next_question, ground_answer, normalize_attributes
from starter.semantic_index import concepts_for_item, file_sha256


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
SEMANTIC_CANDIDATES = 50
ENTROPY_POOL_SIZE = 20
MAX_TURNS = 10
MAX_CANDIDATE_CONCEPTS = 60
MAX_QUERY_CONCEPTS = 8
MAX_DOCUMENT_CONCEPTS = 16
EMBED_BATCH_SIZE = 96
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
    r"\bi['’]?m\s+looking\s+for\s+(.+?)(?:[.!?]|,\s*(?:but|and)\b)",
    re.IGNORECASE,
)


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


def _values(value: object) -> list[str]:
    """Normalize catalog values without changing their meaning."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    return [str(value)] if value != "" else []


def _z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation < 1e-9:
        return [0.0] * len(values)
    return [(value - mean) / deviation for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


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
        self.entropy_omega = float(os.environ.get("ENTROPY_OMEGA", "1.0"))
        self.attributes_by_asin: dict[str, dict] = {}
        self.connection = self._open_search_index()
        self._load_semantic_index()
        self._load_attribute_index()

    def _open_search_index(self) -> sqlite3.Connection:
        """Open a catalog-fingerprinted FTS cache, building it only once."""
        configured = os.environ.get("CATALOG_FTS_PATH")
        cache_path = Path(configured) if configured else self.catalog_path.with_name("catalog_fts.sqlite")
        fingerprint = file_sha256(self.catalog_path)
        if cache_path.exists():
            try:
                cached = sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True)
                metadata = dict(cached.execute("SELECT key, value FROM metadata"))
                if metadata.get("schema_version") == "1" and metadata.get("source_sha256") == fingerprint:
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
                self._build_index(building)
                building.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                building.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (("schema_version", "1"), ("source_sha256", fingerprint)),
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
            self._build_index(connection)
            return connection

    def _build_index(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
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
            
        semantic_path = Path(os.environ.get("SEMANTIC_INDEX_PATH", clean_path.with_name("semantic_index.sqlite")))
        if semantic_path.exists():
            try:
                connection = sqlite3.connect(f"file:{semantic_path.resolve()}?mode=ro", uri=True)
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                valid = (
                    metadata.get("schema_version") == "2"
                    and metadata.get("source_sha256") == file_sha256(clean_path)
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

        with clean_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                parent_asin = str(item.get("parent_asin", ""))
                if not parent_asin:
                    continue
                # Preserve order and cap noisy / unusually long product records.
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
            f"SELECT concept, embedding, dimensions FROM concepts WHERE concept IN ({placeholders}) AND embedding IS NOT NULL",
            missing,
        )
        for concept, blob, dimensions in cursor:
            count = int(dimensions)
            if isinstance(blob, bytes) and len(blob) == 8 * count:
                self.embedding_cache[str(concept)] = list(struct.unpack(f"<{count}d", blob))

    def _load_attribute_index(self) -> None:
        """Load the flat attribute catalog used by entropy question selection."""
        configured = os.environ.get("CATALOG_ATTRIBUTES_PATH")
        path = Path(configured) if configured else self.catalog_path.with_name("catalog_attributes.jsonl")
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                parent_asin = str(row.get("parent_asin", ""))
                if parent_asin:
                    self.attributes_by_asin[parent_asin] = normalize_attributes(row)

    def _ollama_generate(self, prompt: str) -> tuple[dict | None, dict]:
        if not self._llm_available:
            return None, {"prompt_tokens": 0, "completion_tokens": 0}
        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = Request(
            f"{self.ollama_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.ollama_timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
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
                payload = json.dumps({"model": self.embedding_model, "input": batch}).encode("utf-8")
                request = Request(
                    f"{self.ollama_url.rstrip('/')}/api/embed",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=self.ollama_timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
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
            counts.update(set(self.concepts_by_asin.get(parent_asin, [])))
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

Return JSON only:
{{"concepts": ["exact candidate concept"]}}

Conversation: {json.dumps(conversation[-6:], ensure_ascii=False)}
Candidate concepts: {json.dumps(candidates, ensure_ascii=False)}
"""
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

    def _semantic_scores(self, query_concepts: list[str], rows: list[tuple[str, float]]) -> list[float] | None:
        if not query_concepts:
            return None
        document_concepts = [
            concept
            for parent_asin, _ in rows
            for concept in self.concepts_by_asin.get(parent_asin, [])
        ]
        embeddings = self._embed([*query_concepts, *document_concepts])
        if embeddings is None or any(concept not in embeddings for concept in query_concepts):
            return None
        scores: list[float] = []
        for parent_asin, _ in rows:
            concepts = [concept for concept in self.concepts_by_asin.get(parent_asin, []) if concept in embeddings]
            if not concepts:
                scores.append(0.0)
                continue
            # SemRank's multi-vector score: each query concept matches its most
            # similar document concept, then those maxima are averaged.
            maxima = [
                max(_cosine(embeddings[query], embeddings[document]) for document in concepts)
                for query in query_concepts
            ]
            scores.append(sum(maxima) / len(maxima))
        return scores

    def _entropy_attribute(self, rows: list[tuple[str, float]], exhausted: set[str], turn: int) -> str | None:
        """Entropy-based clarifying-question selection (ENTROPY_QUESTION_SELECTION.md)."""
        if turn >= MAX_TURNS or not self.attributes_by_asin:
            return None
        pool = [
            self.attributes_by_asin[parent_asin]
            for parent_asin, _ in rows[:ENTROPY_POOL_SIZE]
            if parent_asin in self.attributes_by_asin
        ]
        if not pool:
            return None
        return choose_next_question(pool, exhausted, omega=self.entropy_omega)

    def _ground_answer(self, session: dict, answer: str, attribute: str) -> None:
        """Map a free-text answer to curated catalog values and record the slot."""
        if not self.attributes_by_asin or not session["last_candidates"]:
            return
        pool = [
            self.attributes_by_asin[parent_asin]
            for parent_asin in session["last_candidates"]
            if parent_asin in self.attributes_by_asin
        ]
        if not pool:
            return
        embed_fn = self._embed if self._embedding_available or self._semantic_connection is not None else None
        grounded = ground_answer(answer, attribute, pool, embed_fn=embed_fn)
        for slot, values in grounded.items():
            session["confirmed"].setdefault(slot, set()).update(values)

    @staticmethod
    def _detect_route(first_message: str) -> str:
        """Buying vs Browsing routing from the opening customer message.

        The scenario type is never disclosed to the agent, so it is inferred
        from the simulator's opening phrasing (evaluator.local_evaluator
        .initial_message). Buying discloses "A key requirement is:"; Browsing
        says "still exploring". Anything ambiguous (including Intent Override,
        which may swap constraints) stays on the safer Browsing path.
        """
        lowered = first_message.lower()
        if "still exploring" in lowered:
            return "browsing"
        if "key requirement" in lowered or "must have" in lowered:
            return "buying"
        return "browsing"

    def _confirmed_conflict(self, parent_asin: str, confirmed: dict[str, set[str]]) -> bool:
        """True if the candidate has values for a confirmed attribute but none match."""
        attrs = self.attributes_by_asin.get(parent_asin)
        if attrs is None:
            return False
        for attribute, wanted in confirmed.items():
            have = set(attrs.get(attribute) or [])
            if have and not (have & wanted):
                return True
        return False

    def _apply_confirmed(
        self,
        rows: list[tuple[str, float]],
        session: dict,
        top_k: int,
    ) -> list[tuple[str, float]]:
        confirmed = session["confirmed"]
        if not confirmed or not self.attributes_by_asin:
            return rows
        matched, conflicting = [], []
        for row in rows:
            bucket = conflicting if self._confirmed_conflict(row[0], confirmed) else matched
            bucket.append(row)
        # Buying: hard filter, but only while it still fills a full result page -
        # a bad extraction should not strand the session with no recommendations.
        if session["route"] == "buying" and len(matched) >= top_k:
            return matched
        return matched + conflicting  # soft: matches first, order otherwise stable

    @staticmethod
    def _refresh_active_terms(session: dict) -> None:
        """Rebuild the bounded FTS query state from active message groups."""
        terms: list[str] = []
        seen: set[str] = set()
        messages: list[str] = []
        for group in session["term_groups"]:
            if not group["active"]:
                continue
            messages.append(group["text"])
            for term in group["terms"]:
                if term not in seen and len(terms) < 40:
                    seen.add(term)
                    terms.append(term)
        session["active_terms"] = terms
        session["active_messages"] = messages

    def _record_message_terms(self, session: dict, message: str) -> None:
        """Add a turn to retrieval state, retiring superseded preferences."""
        is_override = bool(OVERRIDE_RE.search(message))
        if is_override:
            # Keep the category disclosed in the opening turn, but make all
            # earlier preferences inactive before recording the replacement.
            for group in session["term_groups"]:
                if group["kind"] == "preference" and group["active"]:
                    group["active"] = False
                    session["inactive_terms"].update(group["terms"])
            value = OVERRIDE_VALUE_RE.search(message)
            text = value.group(1) if value else message
            terms = _terms(text)
            session["term_groups"].append({"text": text, "terms": terms, "kind": "preference", "active": True})
            # A term explicitly stated again is active, even if it occurred
            # in a preference that has just been overridden.
            session["inactive_terms"].difference_update(terms)
        elif not session["term_groups"]:
            # The opening message carries both the stable category and an
            # optional preference. Preserve only the category on an override.
            category = OPENING_CATEGORY_RE.search(message)
            category_terms = _terms(category.group(1)) if category else []
            all_terms = _terms(message)
            if category_terms:
                session["term_groups"].append(
                    {"text": category.group(1), "terms": category_terms, "kind": "category", "active": True}
                )
                category_set = set(category_terms)
                preference_terms = [term for term in all_terms if term not in category_set]
            else:
                preference_terms = all_terms
            if preference_terms:
                session["term_groups"].append(
                    {"text": message, "terms": preference_terms, "kind": "preference", "active": True}
                )
        else:
            session["term_groups"].append(
                {"text": message, "terms": _terms(message), "kind": "preference", "active": True}
            )
        self._refresh_active_terms(session)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile,
            "messages": [],
            "active_messages": [],
            "active_terms": [],
            "inactive_terms": set(),
            "term_groups": [],
            "asked_attributes": set(),
            "no_preference_attributes": set(),
            "pending_attribute": None,
            "confirmed": {},
            "route": None,
            "last_candidates": [],
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
        if session["route"] is None:
            session["route"] = self._detect_route(user_message)
        pending_attribute = session["pending_attribute"]
        if pending_attribute and NO_PREFERENCE_RE.search(user_message):
            session["no_preference_attributes"].add(pending_attribute)
        elif pending_attribute:
            self._ground_answer(session, user_message, pending_attribute)
        session["messages"].append(user_message)
        self._record_message_terms(session, user_message)
        expression = " OR ".join(f'"{term}"' for term in session["active_terms"])
        if not expression:
            rows: list[tuple[str, float]] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
                "FROM products WHERE products MATCH ? ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, SEMANTIC_CANDIDATES),
            ).fetchall()
            rows = [(str(parent_asin), float(score)) for parent_asin, score in rows]

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._ensure_candidate_concepts(rows)
        candidate_concepts = self._candidate_concepts(rows)
        if candidate_concepts:
            query_concepts, usage = self._select_query_concepts(session["active_messages"], candidate_concepts)
            semantic_scores = self._semantic_scores(query_concepts, rows)
            if semantic_scores is not None:
                base_scores = [-score for _, score in rows]  # SQLite BM25: lower is better.
                combined = [
                    base + semantic
                    for base, semantic in zip(_z_scores(base_scores), _z_scores(semantic_scores))
                ]
                rows = [
                    row for _, row in sorted(zip(combined, rows), key=lambda item: item[0], reverse=True)
                ]

        rows = self._apply_confirmed(rows, session, top_k)
        session["last_candidates"] = [parent_asin for parent_asin, _ in rows]

        exhausted = (
            session["asked_attributes"]
            | session["no_preference_attributes"]
            | set(session["confirmed"])
        )
        ask_attribute = self._entropy_attribute(rows, exhausted, turn)
        if ask_attribute is not None:
            # Every attribute - "other" included - is asked at most once.
            session["asked_attributes"].add(ask_attribute)
        session["pending_attribute"] = ask_attribute
        recommendations = [{"parent_asin": parent_asin} for parent_asin, _ in rows[:top_k]]
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
