# TechJam Submission Guide and Report

This document describes the participant bundle in `submission/`. The agent
entry point is `submission/agent.py`, which exports `Agent` with the required
`reset(...)` and `respond(...)` methods.

## Submission bundle

Upload the `submission/` directory with these participant-owned runtime files:

```text
submission/__init__.py
submission/agent.py
submission/demo.py
submission/README.md
submission/requirements.txt
submission/assets/catalog_attributes.jsonl
submission/src/__init__.py
submission/src/extract_product_attributes.py
submission/src/question_selection.py
submission/src/semantic_index.py
submission/src/precompute.py
```

The submission-specific `submission/README.md` is self-contained and includes
setup, harness integration, method/model choice, cost, limitations, a
multi-turn demonstration, and team contributions. The repository-level
`README.md` and `SUBMISSION.md` provide additional project detail but are not
required by the runtime bundle.

The repository does not include `data/`, `evaluator/`, `tests/`, or `docs/`.
The organizer-provided frozen catalog must be mounted or copied to a path that
is passed to `Agent(...)`. Do not submit public labels, the evaluator, tests,
tuning scripts, development results, generated SQLite caches, or copied
organizer documentation. In particular, exclude:

```text
data/public_set.jsonl
submission/assets/catalog_fts.sqlite
submission/assets/semantic_index.sqlite
evaluator/
tests/
docs/
tools/
results*.json
```

The bundle includes `submission/assets/catalog_attributes.jsonl`, generated
deterministically from the public frozen catalog. It contains no development
session labels, private evaluation data, user data, or credentials. Reproduce
or refresh it with:

```bash
python -m submission.src.extract_product_attributes \
  --input data/catalog.jsonl \
  --output submission/assets/catalog_attributes.jsonl
```

This asset powers question selection, constraints, ratings, and dynamic
semantic features. The published rules allow lightweight local assets required
by the agent and do not state a bundle-size limit; confirm the approximately
20.9 MB file is accepted by the actual submission portal.

## Runtime and setup

- CPython 3.10 or newer is required; the submission was tested with CPython
  3.14.6.
- The runtime uses only the Python standard library. The included
  `submission/requirements.txt` records that there are no third-party
  dependencies; running `pip install -r` therefore installs nothing.
- SQLite must include FTS5 support, as it does in standard CPython builds.
- Write access to `submission/assets` is optional. If unavailable, the agent builds
  its FTS table in memory instead of writing `catalog_fts.sqlite`.

For a deterministic run that makes no model or network request, set
`OLLAMA_ENABLED=0` before starting the organizer-provided harness:

```bash
export OLLAMA_ENABLED=0
```

With the separately supplied organizer development package, the one-command
public reproduction run is:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

The official harness should import `Agent` from `submission.agent`, pass the
organizer catalog path to the constructor, call `reset` once per session, and
then call `respond` for each turn. For local development only, teams may use
the evaluator and public sessions from the organizer's separate development
package.

## Method and model choice

The agent uses weighted SQLite FTS5 BM25 retrieval over title, category,
features, details, store, and description. It retains conversational query
terms, recognizes explicit intent overrides, grounds clarification answers to
catalog attributes, applies hard-constraint reranking, and uses a small
review-count-smoothed rating prior. Follow-up attributes are chosen using a
rank-weighted information-gain calculation over the current candidates.

Semantic reranking is optional. The default local configuration uses
`llama3.2:3b` to select catalog-constrained query concepts and
`nomic-embed-text-v2-moe` for embeddings through a local Ollama HTTP endpoint.
BM25 first creates a bounded candidate set; semantic scoring never searches the
full catalog. A prebuilt semantic index can be generated during setup with:

```bash
python -m submission.src.precompute
```

The generated `submission/assets/semantic_index.sqlite` is about 447 MB in the
current development environment and is deliberately excluded from the normal
submission bundle unless the organizer explicitly permits an artifact of that
size.

## Network and offline behavior

The agent does not use an internet API and requires no API key or live
credential. When enabled, it sends requests only to the configurable Ollama
endpoint, which defaults to `http://localhost:11434`. Model downloads, if
needed, are a setup-time action and should not occur during official scoring.

If Ollama is disabled, unavailable, malformed, or times out, the agent falls
back to BM25, catalog constraints, rating reranking, and token-overlap answer
grounding. Set `OLLAMA_ENABLED=0` to select this fallback immediately and avoid
waiting for a failed local-model request in a network-restricted environment.

## Latency, token usage, and cost disclosure

Measurements below were taken on the current Windows development machine on
2026-08-30 with warm generated retrieval caches:

- Offline demonstration (`python -m submission.demo`, `OLLAMA_ENABLED=0`): agent
  initialization plus a full simulated clarification dialogue completed in
  roughly 3 seconds wall-clock.
- The recorded 200-session development run in `results.dev.json` reported
  18,292 prompt tokens and 2,313 completion tokens, or 20,605 total generation
  tokens (103.025 per session on average).
- Offline fallback reports zero model tokens. `precompute.py` reports embedding
  token usage when the local Ollama response includes `prompt_eval_count`.
- Estimated external model/API cost is USD 0.00 for the intended local Ollama
  configuration and the offline fallback. This estimate excludes local
  hardware, electricity, model-download bandwidth, and setup time. If
  `OLLAMA_URL` is redirected to a billable hosted service, its provider pricing
  must be disclosed separately before submission.

Local-model latency is hardware- and cache-dependent and was not captured in
the saved development result. The offline measurement is a warm-cache
reference, not a claim about cold-start or Ollama latency on organizer
hardware; a bundle that excludes generated caches will be slower on its first
run.

## Team contributions

Two members owned implementation; three owned the research and methodology
behind the solution. Members collaborated across areas; the split below reflects
primary ownership, and the research contributions are not reflected in Git
commit history.

- **Elbert Tristan Lie — Retrieval & pipeline implementation.** Built the
  dual-track intent routing, the weighted FTS5 BM25 index and multi-route
  retrieval, the deterministic/Ollama attribute extractor, and the
  BM25 → semantic-rerank in-memory pipeline.
- **Kelven Nathanael — Dialogue & question-selection implementation.** Built the
  conversational state tracker (incremental slot accumulation and
  intent-override rewriting) and the entropy / gain-ratio question selector with
  its `ENTROPY_QUESTION_SELECTION.md` specification.
- **Alexandra Martina Setiawan — Retrieval research.** Surveyed hybrid
  lexical-plus-dense retrieval, field-weighted BM25, pseudo-relevance feedback,
  and candidate-constrained LLM reranking; benchmarked local embedding models.
- **Frederico Samuel Halim — Conversational-strategy research.** Reviewed
  mixed-initiative conversational search, dialogue state tracking and
  slot-filling, information-gain clarification, and intent-override recovery;
  sourced the ID3 / C4.5 gain-ratio and multi-label-entropy literature and the
  personalized context-distillation framing the implementation follows.
- **Aufan Ahmad Mumtaza — Evaluation methodology.** Studied the organizer's
  scoring criteria (Hit Rate@K, MRR, MTTC, and the Efficiency reward) to
  translate them into concrete design targets, analyzed the Amazon Reviews 2023
  catalog for coverage and attribute sparsity, and built the offline evaluation
  loop and per-turn diagnostics the team used to track those metrics during
  development.

## Environment variables

All variables are optional; the defaults reproduce the checked-in configuration.

<!-- Keep this table in sync with README.md § Configuration. -->

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_ENABLED` | `0` | Set to `1` to opt into local Ollama; `0`, `false`, `no`, or unset selects the deterministic offline path. |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama base URL. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Catalog-constrained concept-selection model. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Local embedding model and semantic-index identity. |
| `OLLAMA_TIMEOUT` | `30` | Per-request timeout in seconds for the agent. The `extract_product_attributes` build script uses `120` when the variable is unset. |
| `SEMANTIC_CANDIDATES` | `50` | Normal BM25 candidate-pool size. |
| `OVERRIDE_CANDIDATES` | `150` | Candidate-pool size after an intent override. |
| `SEMANTIC_CONFIDENCE_MARGIN` | `0.015` | Minimum top-vs-runner-up semantic-score separation before a rerank is trusted. |
| `SEMANTIC_BLEND_WEIGHT` | `0.85` | Semantic contribution to a confident rerank. |
| `SEMANTIC_EXPANSION_WEIGHT` | `0.20` | Contribution from model-selected concept expansions; `0` disables the LLM expansion call. |
| `BM25_WEIGHTS` | `0,4.5,4,2.5,2.5,1.5,1` | Seven non-negative FTS5 column weights: `parent_asin` (unindexed), title, categories, features, details, store, description. |
| `CATALOG_FTS_PATH` | `submission/assets/catalog_fts.sqlite` | Optional generated FTS cache path. |
| `CATALOG_ATTRIBUTES_PATH` | `submission/assets/catalog_attributes.jsonl` | Structured attributes used for questions, constraints, and ratings. |
| `CLEAN_CATALOG_PATH` | `submission/assets/catalog_attributes.jsonl` | Concept source for semantic retrieval (the same bundled file by default). |
| `SEMANTIC_INDEX_PATH` | `submission/assets/semantic_index.sqlite` | Optional persisted concept-vector index (resolved beside `CLEAN_CATALOG_PATH`). |

## Known limitations

- The offline fallback loses semantic synonym matching and model-selected query
  expansion, so retrieval quality can be lower than the local-model path.
- Without a persisted semantic index, the local-model path embeds only current
  candidate concepts at runtime, which increases cold-start latency.
- Intent and attribute parsing is primarily English and uses curated patterns.
- Question quality depends on coverage and cleanliness of the derived catalog
  attributes; missing metadata is treated as unknown rather than conflicting.
- The generated FTS cache is an optimization. Read-only environments remain
  correct but may have higher initialization time and memory use.
