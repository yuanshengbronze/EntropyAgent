"""Entropy-based clarifying-question selection.

Standalone implementation of ``ENTROPY_QUESTION_SELECTION.md``: given the
current ranked candidate pool, decide which attribute the agent should ask
about next. Kept free of any agent/evaluator imports so it can be unit tested
in isolation.

Pipeline (see the spec for citations):

1. ``multi_label_entropy``  - Clare & King (PKDD 2001). An item may carry
   several values for one attribute ("cotton" *and* "spandex"), so we sum
   per-value binary entropies instead of assuming mutual exclusivity.
2. ``split_information``     - Quinlan C4.5 (1993). Divides raw gain back down
   for attributes fragmented into many near-unique values.
3. coverage discount         - C4.5-style handling of missing values: scale the
   score by the fraction of the pool that can even answer the question.
4. ``choose_next_question``  - Nunez EG2 (1991) Information Cost Function folds
   in how answerable each attribute actually is for the simulator.

``ground_answer`` closes the loop: a free-text simulator answer ("sporty")
rarely matches a curated catalog value ("athletic") verbatim, so it is mapped
onto the known values for the asked attribute - by embedding cosine similarity
when an embed function is supplied, else by significant-token overlap.
"""

from __future__ import annotations

import bisect
import math
import re
from typing import Callable, Iterable, Sequence

# ``category`` and ``brand`` are excluded entirely - the deterministic
# simulator never reveals them (spec section 1). ``other`` is handled
# separately as the safe fallback, not scored here.
ASKABLE_ATTRIBUTES: tuple[str, ...] = (
    "material",
    "color",
    "size",
    "style",
    "feature",
    "use_case",
    "budget",
)

# How often the deterministic simulator can actually answer a question about
# each attribute, used as the EG2 cost term ``cost = 1 / prior``. Estimated
# from the structure of ``evaluator.local_evaluator``:
#   - ``intent_card`` front-loads a material and a color constraint whenever the
#     product text mentions one, so ``material``/``color`` are frequently
#     answerable; ``"budget around $X"`` is appended last and almost never
#     survives into the disclosed window.
#   - ``classify_constraint`` routes any unrecognised phrase to ``feature``,
#     making it a near-catch-all; ``style`` needs department/fit/sleeve/neck
#     wording, ``size`` needs width/size wording, ``use_case`` needs an activity
#     word - all comparatively rare.
#   - ``other`` matches any undisclosed constraint (wildcard).
# Rough values - tune against a dev sample, never the official public set.
ANSWERABILITY_PRIOR: dict[str, float] = {
    "feature": 0.90,
    "material": 0.70,
    "color": 0.25,
    "style": 0.10,
    "size": 0.05,
    "use_case": 0.02,
    "budget": 0.01,   # appended last in intent_card; floored so cost stays finite
    "other": 0.95,
}

EPSILON = 1e-9

_MISSING_TOKENS = {"", "unknown", "n/a", "na", "none", "null", "-"}
_PRICE_KEY = "_price"
_BUDGET_BUCKETS = 4


def _split_values(raw: object) -> list[str]:
    """Flatten a raw ``catalog_attributes.jsonl`` cell into clean tokens."""
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [str(raw)]
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        for item in raw:
            parts.extend(_split_values(item))
        return list(dict.fromkeys(parts))
    tokens: list[str] = []
    for part in str(raw).replace(";", ",").split(","):
        cleaned = " ".join(part.split()).strip(" -").lower()
        if cleaned and cleaned not in _MISSING_TOKENS:
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))


def _price(raw: object) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    if raw is None:
        return None
    text = str(raw).replace("$", "").replace(",", "").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def normalize_attributes(row: dict) -> dict:
    """One ``catalog_attributes.jsonl`` row -> ``{attribute: [values], _price}``.

    ``brand`` and ``category`` are dropped (never askable). The raw price is
    kept under ``_price`` for on-the-fly budget bucketing.
    """
    return {
        "material": _split_values(row.get("materials", row.get("material"))),
        "color": _split_values(row.get("color")),
        "size": _split_values(row.get("size")),
        "style": _split_values(row.get("style")),
        "feature": _split_values(row.get("feature")),
        "use_case": _split_values(row.get("use_case")),
        "other": _split_values(row.get("other")),
        _PRICE_KEY: _price(row.get("budget_price", row.get("budget"))),
    }


def build_pool(rows: Iterable[dict]) -> list[dict]:
    """Convenience: normalize an iterable of raw catalog rows."""
    return [normalize_attributes(row) for row in rows]


def _values(item: dict, attribute: str) -> Sequence[str]:
    value = item.get(attribute) or ()
    return value if isinstance(value, (list, tuple)) else (value,)


def multi_label_entropy(pool: Sequence[dict], attribute: str) -> float:
    """Clare & King multi-label entropy over the pool for one attribute.

    ``-Sigma [ p(v)*log2 p(v) + (1 - p(v))*log2(1 - p(v)) ]`` where ``p(v)`` is
    the fraction of pool items carrying value ``v``.
    """
    total = len(pool)
    if total == 0:
        return 0.0
    distinct = {value for item in pool for value in _values(item, attribute)}
    entropy = 0.0
    for value in distinct:
        present = sum(1 for item in pool if value in _values(item, attribute))
        p = present / total
        if 0.0 < p < 1.0:
            entropy += -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    return entropy


def split_information(pool: Sequence[dict], attribute: str) -> float:
    """C4.5 SplitInfo, adapted for multi-label values.

    Values overlap, so ``Sigma |S_v|`` can exceed ``|S|`` and the textbook
    ``|S_v| / |S|`` ratio stops being a probability. We normalize each value
    count by the total incidence ``T = Sigma |S_v|`` instead, giving a proper
    entropy over the value-occurrence distribution. An attribute split into
    many small even pieces scores high here and divides its gain back down.
    """
    counts: list[int] = []
    distinct = {value for item in pool for value in _values(item, attribute)}
    for value in distinct:
        counts.append(sum(1 for item in pool if value in _values(item, attribute)))
    incidence = sum(counts)
    if incidence == 0:
        return 0.0
    info = 0.0
    for count in counts:
        p = count / incidence
        if p > 0.0:
            info += -p * math.log2(p)
    return info


def gain_ratio_multilabel_missing(pool: Sequence[dict], attribute: str) -> float:
    """Combined score for one attribute (spec section 4).

    ``gain_ratio(known) * coverage`` where ``known`` is the subset of the pool
    that actually has a value for ``attribute``.
    """
    total = len(pool)
    if total == 0:
        return 0.0
    known = [item for item in pool if _values(item, attribute) and any(_values(item, attribute))]
    if not known:
        return 0.0
    coverage = len(known) / total
    gain = multi_label_entropy(known, attribute)
    info = split_information(known, attribute)
    gain_ratio = gain / info if info > EPSILON else 0.0
    return gain_ratio * coverage


def _log_quantile_edges(prices: Sequence[float], buckets: int) -> list[float]:
    logs = sorted(math.log(price) for price in prices)
    edges: list[float] = []
    for index in range(1, buckets):
        position = round(index * (len(logs) - 1) / buckets)
        edges.append(logs[position])
    # Keep edges strictly increasing so bisect assigns stable buckets.
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = math.nextafter(edges[index - 1], math.inf)
    return edges


def _with_budget_buckets(pool: Sequence[dict], buckets: int = _BUDGET_BUCKETS) -> list[dict]:
    """Return a shallow copy of the pool with a bucketed ``budget`` value list.

    Log-quantile buckets (spec section 5): "$20 vs $40" and "$400 vs $420" are
    very different practical gaps despite an identical raw difference.
    """
    prices = [item[_PRICE_KEY] for item in pool if item.get(_PRICE_KEY)]
    view = [dict(item) for item in pool]
    if len(prices) < buckets:
        for item in view:
            item["budget"] = []
        return view
    edges = _log_quantile_edges(prices, buckets)
    for item in view:
        price = item.get(_PRICE_KEY)
        if price:
            index = bisect.bisect_right(edges, math.log(price))
            item["budget"] = [f"budget_q{index + 1}"]
        else:
            item["budget"] = []
    return view


def information_cost_score(gain_ratio: float, cost: float, omega: float) -> float:
    """Nunez EG2 Information Cost Function: ``(2**gr - 1) / (cost + 1)**omega``."""
    return (2.0 ** gain_ratio - 1.0) / (cost + 1.0) ** omega


def choose_next_question(
    pool: Sequence[dict],
    exhausted: Iterable[str] = (),
    *,
    omega: float = 1.0,
    priors: dict[str, float] | None = None,
    askable: Sequence[str] = ASKABLE_ATTRIBUTES,
) -> str:
    """Pick the attribute worth asking about next.

    ``pool`` is the current ranked candidate set as normalized attribute maps
    (see ``normalize_attributes``). ``exhausted`` are attributes already asked
    or declined. Returns one of ``askable`` or ``"other"`` - the safe fallback
    that bypasses classification and matches any undisclosed constraint.
    """
    priors = priors or ANSWERABILITY_PRIOR
    blocked = set(exhausted)
    candidates = [a for a in askable if a not in blocked]
    if not pool or not candidates:
        return "other"

    scored_pool: Sequence[dict] = pool
    if "budget" in candidates:
        scored_pool = _with_budget_buckets(pool)

    scores: dict[str, float] = {}
    for attribute in candidates:
        gain_ratio = gain_ratio_multilabel_missing(scored_pool, attribute)
        if gain_ratio <= 0.0:
            continue
        prior = priors.get(attribute, 0.5)
        cost = 1.0 / prior if prior > 0.0 else math.inf
        scores[attribute] = information_cost_score(gain_ratio, cost, omega)

    if not scores or max(scores.values()) < EPSILON:
        return "other"
    return max(scores, key=scores.get)


# --------------------------------------------------------------------------- #
# Answer grounding: free-text answer -> curated catalog values
# --------------------------------------------------------------------------- #

ANSWER_SIMILARITY_THRESHOLD = 0.60

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GROUND_STOPWORDS = {
    "for", "that", "what", "matters", "is", "the", "and", "with", "you", "your",
    "i", "me", "my", "a", "an", "of", "to", "in", "on", "it", "prefer", "want",
    "need", "looking", "like", "would", "please", "dont", "have", "additional",
    "preference", "no", "not", "any", "something", "around", "about", "some",
}

# One raw catalog answer signalling "no value here"; callers usually screen
# these earlier but grounding must not invent a match from them.
_NO_VALUE_RE = re.compile(
    r"\b(?:no preference|don't have|do not have|doesn't matter|does not matter|"
    r"use your judgment|not quite right)\b",
    re.IGNORECASE,
)

EmbedFn = Callable[[list[str]], "dict[str, list[float]] | None"]


def _significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _GROUND_STOPWORDS
    }


def _token_overlap_matches(answer: str, values: Sequence[str]) -> list[str]:
    answer_tokens = _significant_tokens(answer)
    answer_lower = answer.lower()
    matched: list[str] = []
    for value in values:
        value_tokens = _significant_tokens(value)
        if not value_tokens:
            continue
        if value.lower() in answer_lower or value_tokens & answer_tokens:
            matched.append(value)
    return matched


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _embedding_matches(
    answer: str,
    values: Sequence[str],
    embed_fn: EmbedFn,
    threshold: float,
) -> list[str]:
    try:
        vectors = embed_fn([answer, *values])
    except Exception:
        return []
    if not vectors or answer not in vectors:
        return []
    query = vectors[answer]
    return [
        value
        for value in values
        if value in vectors and _cosine(query, vectors[value]) >= threshold
    ]


def _attribute_vocabulary(pool: Sequence[dict], attribute: str) -> list[str]:
    return sorted({value for item in pool for value in (item.get(attribute) or [])})


def ground_answer(
    answer_text: str,
    attribute: str,
    pool: Sequence[dict],
    *,
    embed_fn: EmbedFn | None = None,
    similarity_threshold: float = ANSWER_SIMILARITY_THRESHOLD,
) -> dict[str, list[str]]:
    """Map a free-text answer onto known catalog values.

    ``attribute`` is the attribute that was asked. For ``"other"`` the answer is
    matched against every askable attribute's vocabulary and recorded wherever
    values land. Returns ``{attribute: [matched values]}`` (empty when nothing
    matches or the answer expresses no preference).
    """
    if not answer_text or _NO_VALUE_RE.search(answer_text):
        return {}
    targets = ASKABLE_ATTRIBUTES if attribute == "other" else (attribute,)
    result: dict[str, list[str]] = {}
    for target in targets:
        if target not in ASKABLE_ATTRIBUTES:
            continue
        values = _attribute_vocabulary(pool, target)
        if not values:
            continue
        matched: list[str] = []
        if embed_fn is not None:
            matched = _embedding_matches(answer_text, values, embed_fn, similarity_threshold)
        if not matched:
            matched = _token_overlap_matches(answer_text, values)
        if matched:
            result[target] = matched
    return result
