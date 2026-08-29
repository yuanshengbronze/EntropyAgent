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
use_case, other, null`. All of these except `null` are candidates:
`category … use_case` are scored by the function below, and `other` is the
wildcard fallback. This document covers exactly one function:

```
ask_attribute = choose_next_question(top_k_candidates, exhausted_slots)
```

> **`category` and `brand` are suppressed, not excluded.**
> `evaluator/local_evaluator.py`'s `classify_constraint` has no branch that
> returns `category` or `brand`, so against the *local* evaluator those two
> questions always draw "no preference" (and the category is already stated
> in the opening message). They stay in the candidate set but with a
> near-zero answerability prior (§4), which the cost term drives to the
> bottom — so they are only ever asked when nothing else has any signal.
> The component should not hard-code an assumption about a simulator the
> private evaluation set may not share.

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

| attribute | value distribution across top_k | multi-label entropy (§3.1) | picked? |
|---|---|---|---|
| color | 14 blue, 3 black, 3 red | 2.10 | ✅ highest |
| style | 15 crew neck, 5 v-neck | 1.62 | |
| use_case | 19 casual, 1 athletic | 0.57 | |

`color` wins — not because it's inherently important, but because it's
where *this specific candidate pool* currently disagrees the most. If the
pool later converges to all-blue items, color's entropy drops toward 0 and
a different attribute wins instead, automatically.

---

## 3. Two corrections we apply, and one we don't

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
extraction pipeline). Three concrete problems fall out of this. We apply
the classical fix for the first two; for the third we take the known cost
knowingly (see below).

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
high `SplitInfo`, dividing its raw gain back down. This matters directly
for `brand` and `feature`, which in a 20-item pool are often close to 20
distinct values.

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
constraint window; `category` / `brand` are never classified at all —
see §1). Fold in the reverse-engineered answerability priors using EG2's
Information Cost Function (Núñez, *Machine Learning*, 1991 — via survey
https://www.researchgate.net/publication/261853221):

```
ICF(A) = (2^GainRatio(A) - 1) / (Cost(A) + 1)^ω
```

```python
def choose_next_question(top_k, exhausted, omega=1.0):
    candidates = {"category", "material", "color", "size", "style",
                  "brand", "budget", "feature", "use_case"} - exhausted

    scores = {}
    for a in candidates:
        gr = gain_ratio_multilabel_missing(top_k, a)
        cost = 1.0 / answerability_prior(a)   # from reverse-engineered simulator rules
        scores[a] = (2 ** gr - 1) / (cost + 1) ** omega

    if not scores or max(scores.values()) < EPSILON:
        return None if "other" in exhausted else "other"   # wildcard, then silence
    return max(scores, key=scores.get)
```

`exhausted` = attributes already asked, declined ("no preference"), or
confirmed. `"other"` is the safe fallback — it bypasses classification and
matches any undisclosed constraint; once it too is exhausted the function
returns `None` and the agent asks nothing that turn.

`category` and `brand` are in `candidates` but carry a near-zero
answerability prior (`~0.005`), so `ICF` pushes them to the bottom unless
nothing else has any signal. The local simulator never answers them; the
private evaluator might, so they are suppressed rather than removed.

`omega` is a free parameter controlling how strongly cost dominates gain —
tune it against a dev sample (grid search, see main README §4).

### The final formula, consolidated

For the current pool `S` (the top-k ranked candidates this turn) and a
candidate attribute `A`, write `A(i)` for the set of values item `i` carries
for `A`, and `V_A = ⋃_{i∈S} A(i)` for the values seen in the pool.

**1. Known subset and coverage** (missing-value discount, §3.2)

```
S_A    = { i ∈ S : A(i) ≠ ∅ }
cov(A) = |S_A| / |S|
```

**2. Multi-label entropy** over `S_A` (Clare & King, §3.1), with
`p(v) = |{ i ∈ S_A : v ∈ A(i) }| / |S_A|`

```
H(A) = − Σ_{v ∈ V_A}  [ p(v)·log₂ p(v) + (1−p(v))·log₂(1−p(v)) ]
       ( terms with p(v) ∈ {0, 1} contribute 0 )
```

**3. Split information** (C4.5, §3.3), multi-label adaptation: normalise each
value count `c(v) = |{ i ∈ S_A : v ∈ A(i) }|` by the **total incidence**
`T = Σ_{v ∈ V_A} c(v)` (not `|S_A|`, which is not a probability when values
overlap)

```
SI(A) = − Σ_{v ∈ V_A}  (c(v)/T)·log₂(c(v)/T)
```

**4. Coverage-discounted gain ratio**

```
GR(A) = H(A) / SI(A)          ( 0 if SI(A) = 0 )
G(A)  = GR(A) · cov(A)
```

**5. Cost-weighted score** (Núñez EG2), with `π(A)` the answerability prior
and `cost(A) = 1 / π(A)`

```
ICF(A) = ( 2^{G(A)} − 1 ) / ( cost(A) + 1 )^ω
```

**6. Selection**

```
𝒜        = { category, material, color, size, style, brand, budget, feature, use_case }
exhausted = asked ∪ declined(no-preference) ∪ confirmed
A*        = argmax_{ A ∈ 𝒜 \ exhausted, G(A) > 0 }  ICF(A)

ask_attribute = A*        if  ICF(A*) > ε
              = "other"   else if  "other" ∉ exhausted   ( wildcard fallback )
              = None      otherwise                      ( ask nothing )
```

Notes:

- `budget` is not categorical — before step 2 its values are replaced by
  log-quantile bucket labels computed across `S` (§5); with fewer than 4
  priced items it is skipped that turn (`G(budget) = 0`).
- `category` and `brand` stay in `𝒜` but `π(category) = π(brand) ≈ 0.005`,
  so `ICF` all but eliminates them (§1).
- `ω` = `ENTROPY_OMEGA` (default `1.0`); `ε` = `1e-9`.
- Reference implementation: `starter/question_selection.py`
  (`gain_ratio_multilabel_missing`, `information_cost_score`,
  `choose_next_question`).

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

- Confirm the `answerability_prior` values on a dev sample — never
  `data/public_set.jsonl`, which is the held-out test set. Current values
  are estimated from the structure of `intent_card` / `classify_constraint`
  in `evaluator/local_evaluator.py`; `category` / `brand` are pinned near
  zero because that simulator has no branch for them.
- Check whether joint (multi-attribute) entropy is worth the added
  complexity over independent per-attribute scoring — likely unnecessary
  since only one attribute is asked per turn.
- Tune `omega` and the exhaustion/coverage thresholds via grid search on a
  dev sample once Milestone 0 (baseline) is working.
- Revisit the pool size (currently top 20) and whether `category` / `brand`
  should be scored at all once the private-set behaviour is known.

---

## 7. References

| Paper | Link | Contribution |
|---|---|---|
| Quinlan, *Induction of Decision Trees*, 1986 | https://link.springer.com/article/10.1007/BF00116251 | Base entropy / information gain formula |
| Quinlan, *C4.5: Programs for Machine Learning*, 1993 | https://arxiv.org/pdf/2603.11117 (formulas summarized, Appendix B) | Gain Ratio (cardinality correction), missing-value handling |
| Clare & King, *Knowledge Discovery in Multi-label Phenotype Data*, PKDD 2001 | https://www.semanticscholar.org/paper/Knowledge-Discovery-in-Multi-label-Phenotype-Data-Clare-King/e64ef24d0f6a9cefd7bf7a6b1d5f34f90ec37939 | Multi-label entropy (item has multiple attribute values) |
| Núñez, *EG2*, *Machine Learning*, 1991 | https://www.researchgate.net/publication/261853221 | Cost-weighted attribute selection (Information Cost Function) |
