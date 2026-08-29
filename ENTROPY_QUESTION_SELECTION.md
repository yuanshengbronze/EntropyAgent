# Entropy-Based Clarifying Question Selection

Standalone spec for one component of the TechJam conversational search
agent: **how the agent decides which attribute to ask about next.** Split
out from the main architecture README so it can be implemented and tested
in isolation. See the main README for how this plugs into the rest of the
system (Signal 1 ranking, state tracking, scenario handling).

---

## 1. Where this fits

Each turn, the agent may attach an `ask_attribute` to its response, chosen
from: `category, material, color, size, style, brand, budget, feature,
use_case, other, null`. `category` and `brand` are excluded entirely (see
main README §2 — structurally never answerable by the simulator). This
document covers exactly one function:

```
ask_attribute = choose_next_question(top_k_candidates, confirmed_slots)
```

`top_k_candidates` is Signal 1's current ranked output (e.g. top 20
products by search score) — this function never looks at the whole
catalog, only at what's currently plausible, so its output adapts every
turn as the ranking changes.

---

## 2. The base idea: this is one greedy split of a decision tree

Each conversational turn is structurally identical to choosing the best
splitting attribute at one node of a decision tree — where "the dataset at
this node" is the current `top_k_candidates`, not the full training set,
and there's no recursion (we only ever take one question per turn, then
re-rank and repeat).

**Core formula — Shannon entropy / information gain** (Quinlan, *Induction
of Decision Trees*, *Machine Learning*, 1986 —
https://link.springer.com/article/10.1007/BF00116251):

```
Entropy(S)   = Σ -Pᵢ · log₂(Pᵢ)
Gain(S, A)   = Entropy(S) - Σᵥ (|Sᵥ|/|S|) · Entropy(Sᵥ)
```

Pick `argmax_A Gain(S, A)` — the attribute whose current value
distribution is most "mixed" (highest entropy) is the one worth asking
about, because knowing the answer would do the most to narrow the pool.

**Honest scoping note:** what we compute in practice is closer to raw
entropy of the current pool per attribute than full ID3 `Gain`
(entropy-before minus expected-entropy-after-split) — we don't get to
simulate every possible answer's resulting sub-pool before asking, we only
get one categorical answer back. This is a standard, defensible
simplification when full post-split simulation isn't available, but it
should be stated precisely rather than claimed as exact ID3 equivalence.

**Worked example** (from main README, repeated here for self-containment):
turn 2, `material = cotton` already confirmed, `top_k` = 20 cotton
candidates.

| attribute | value distribution across top_k | entropy | picked? |
|---|---|---|---|
| color | 14 blue, 3 black, 3 red | 1.16 | ✅ highest |
| style | 15 crew neck, 5 v-neck | 0.81 | |
| use_case | 19 casual, 1 athletic | 0.29 | |

`color` wins — not because it's inherently important, but because it's
where *this specific candidate pool* currently disagrees the most. If the
pool later converges to all-blue items, color's entropy drops toward 0 and
a different attribute wins instead, automatically.

---

## 3. Why raw entropy breaks on the real catalog

A real catalog row (`data/public_set.jsonl`-adjacent product data) looks
like this:

```json
{"parent_asin": "B07K34RX5J",
 "title": "Kandinsky Statement Earrings for Women by Spirit Hoops, Fabric...",
 "features": ["Spandex", "Made in USA and Imported",
              "Fashion jewelry: Beautiful fabric earrings...", ...],
 "price": null,
 "details": {"Department": "Womens", "Manufacturer": "Spirit Hoops", ...}}
```

No dedicated `color`/`material`/`style` fields — everything must be
extracted from free text first (see main README §2's AVE/dictionary
extraction pipeline). Three concrete problems fall out of this, each with
an established classical fix (no training required for any of them).

### Problem 1 — one item, multiple values for one attribute

A shirt's features might mention both "cotton" and "spandex." Plain
single-label entropy assumes one value per item per attribute and breaks
under multi-valued extraction results.

**Fix — multi-label entropy** (Clare & King, *Knowledge Discovery in
Multi-label Phenotype Data*, PKDD 2001 —
https://www.semanticscholar.org/paper/Knowledge-Discovery-in-Multi-label-Phenotype-Data-Clare-King/e64ef24d0f6a9cefd7bf7a6b1d5f34f90ec37939).
Originally built for genes with multiple simultaneous functions — same
shape of problem. Sums per-value binary entropies instead of assuming
mutual exclusivity:

```
entropy(E) = -Σᵢ [ p(cᵢ)·log(p(cᵢ)) + q(cᵢ)·log(q(cᵢ)) ],   q(cᵢ) = 1 - p(cᵢ)
```

```python
def multi_label_entropy(top_k, attribute):
    all_values = set(v for item in top_k for v in item.attrs.get(attribute, []))
    total = 0
    for v in all_values:
        p = sum(1 for item in top_k if v in item.attrs.get(attribute, [])) / len(top_k)
        if 0 < p < 1:
            total += -(p * log2(p) + (1 - p) * log2(1 - p))
    return total
```

### Problem 2 — missing/unknown values

`price: null` in the example above; many products will have no detected
`color` or `style` at all after extraction.

**Fix — coverage discount**, in the spirit of C4.5's handling of
incomplete data (Quinlan, 1993 — see §4 below for full citation). Don't
drop items with a missing value, and don't force a fake bucket for them.
Instead, discount the signal by how much of the pool can even answer:

```python
def coverage_discount(top_k, attribute):
    known = [item for item in top_k if attribute in item.attrs]
    if not known:
        return 0.0   # nobody in the pool can answer this — worthless to ask
    return len(known) / len(top_k)
```

(This is a simplified stand-in for C4.5's fractional-instance
branch-splitting, adapted because we only get one categorical answer per
turn rather than a full branch simulation — see the scoping note in §2.)

### Problem 3 — too many distinct categories (worst in Browsing mode)

Raw information gain is well known to favor attributes with many
outcomes — an attribute fragmented into 40 near-unique values will look
maximally "informative" even though asking about it barely narrows
anything (each value only matches ~1 item). This is exactly why C4.5
exists as a correction to ID3.

**Fix — Gain Ratio** (Quinlan, *C4.5: Programs for Machine Learning*, 1993
— formulas summarized at https://arxiv.org/pdf/2603.11117, Appendix B,
Eq. B.6–B.7):

```
SplitInfo(A)  = -Σᵥ (|Sᵥ|/|S|) · log₂(|Sᵥ|/|S|)
GainRatio(A)  = Gain(A) / SplitInfo(A)
```

An attribute that splits the pool into many small, even partitions gets a
high `SplitInfo`, dividing its raw gain back down.

**Why this matters more for Browsing than Buying:** Buying sessions start
with a real disclosed constraint and narrow fast, so cardinality and
missingness stay contained. Browsing sessions start from a wide,
category-only pool (see main README §1) — exactly where both problems are
worst, and exactly where the system has the least other signal to fall
back on. Gain Ratio and the coverage discount are load-bearing for
Browsing; for Buying they're a smaller correctness improvement.

---

## 4. Combining all three: the actual function to implement

```python
def gain_ratio_multilabel_missing(top_k, attribute):
    known = [item for item in top_k if attribute in item.attrs]
    coverage = len(known) / len(top_k) if top_k else 0.0
    if not known:
        return 0.0

    gain = multi_label_entropy(known, attribute)          # Clare & King, §3.1
    split_info = split_information(known, attribute)      # Quinlan C4.5, §3.3
    gain_ratio = gain / split_info if split_info > 0 else 0.0

    return gain_ratio * coverage                          # missingness discount, §3.2
```

### Cost-weighting on top — why raw gain ratio still isn't the final score

Even a high-gain-ratio attribute might be nearly unanswerable by the
simulator (e.g. `budget` is usually buried outside the disclosed
constraint window — see main README §2). Fold in the reverse-engineered
answerability priors using EG2's Information Cost Function (Núñez,
*Machine Learning*, 1991 — via survey
https://www.researchgate.net/publication/261853221):

```
ICF(A) = (2^GainRatio(A) - 1) / (Cost(A) + 1)^ω
```

```python
def choose_next_question(top_k, confirmed_slots, omega=1.0):
    candidates = {"material", "color", "style", "feature", "use_case", "size", "budget"}
    candidates -= confirmed_slots.exhausted   # drop attributes already answered/exhausted

    scores = {}
    for a in candidates:
        gr = gain_ratio_multilabel_missing(top_k, a)
        cost = 1.0 / answerability_prior(a)   # from reverse-engineered simulator rules
        scores[a] = (2 ** gr - 1) / (cost + 1) ** omega

    if not scores or max(scores.values()) < EPSILON:
        return "other"   # safe fallback — bypasses classification, matches
                          # any undisclosed constraint
    return max(scores, key=scores.get)
```

`omega` is a free parameter controlling how strongly cost dominates gain —
tune it against the 200 public sessions (grid search, see main README §4).

---

## 5. Continuous attributes (budget)

`budget` must be bucketed before any of the above applies (it isn't
categorical). Per earlier team discussion, use **logarithmic** or
percentage-relative buckets rather than linear dollar bands — "$20 vs $40"
and "$400 vs $420" represent very different practical differences despite
an identical raw gap:

```python
def bucket_budget(top_k, n_buckets=4):
    prices = sorted(log(item.price) for item in top_k if item.price is not None)
    # quantile-bucket on log(price), not raw price
    ...
```

---

## 6. Open items to validate empirically

- Confirm `answerability_prior` values against the actual 200 public
  sessions (run every attribute type across all sessions, log which ever
  return non-empty responses) rather than trusting the reverse-engineered
  code read alone.
- Check whether joint (multi-attribute) entropy is worth the added
  complexity over independent per-attribute scoring — likely unnecessary
  since only one attribute is asked per turn, but worth a quick check once
  the real 50k-item catalog is downloaded.
- Tune `omega` and the exhaustion/coverage thresholds via grid search once
  Milestone 0 (baseline) is working — see main README §5 build order.

---

## 7. References

| Paper | Link | Contribution |
|---|---|---|
| Quinlan, *Induction of Decision Trees*, 1986 | https://link.springer.com/article/10.1007/BF00116251 | Base entropy / information gain formula |
| Quinlan, *C4.5: Programs for Machine Learning*, 1993 | https://arxiv.org/pdf/2603.11117 (formulas summarized, Appendix B) | Gain Ratio (cardinality correction), missing-value handling |
| Clare & King, *Knowledge Discovery in Multi-label Phenotype Data*, PKDD 2001 | https://www.semanticscholar.org/paper/Knowledge-Discovery-in-Multi-label-Phenotype-Data-Clare-King/e64ef24d0f6a9cefd7bf7a6b1d5f34f90ec37939 | Multi-label entropy (item has multiple attribute values) |
| Núñez, *EG2*, *Machine Learning*, 1991 | https://www.researchgate.net/publication/261853221 | Cost-weighted attribute selection (Information Cost Function) |
