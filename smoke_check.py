"""Manual smoke test for entropy question selection - catalog data only.

Feeds the agent a hand-written scenario and prints, per turn: the detected
route, the attribute the entropy picker chose, the per-attribute scores behind
that choice, the grounded/confirmed slots, and the top recommendations.

    python smoke_check.py
    OLLAMA_ENABLED=0 python smoke_check.py      # force the token-overlap path

Does NOT touch data/public_set.jsonl.
"""

from __future__ import annotations

from submission.agent import Agent, ENTROPY_POOL_SIZE
from submission.src.question_selection import ASKABLE_ATTRIBUTES, gain_ratio_multilabel_missing

# (message, turn). Edit freely - these are not real sessions, just probes.
SCENARIO = [
    "I need a men's rain jacket. A key requirement is: waterproof.",
    "For that, what matters is: sealed seams; packable.",
    "I'd like it in navy.",
    "I don't have a preference for that; use your judgment.",
]


def main() -> None:
    agent = Agent("data/catalog.jsonl")
    print(f"attribute index: {len(agent.attributes_by_asin)} products\n")
    agent.reset("smoke", {"summary": "test"})
    session = agent._sessions["smoke"]

    for turn, message in enumerate(SCENARIO, start=1):
        response = agent.respond("smoke", message, turn, 10)
        pool = [
            agent.attributes_by_asin[a]
            for a, _ in [(x, 0) for x in session["last_candidates"][:ENTROPY_POOL_SIZE]]
            if a in agent.attributes_by_asin
        ]
        scores = {a: round(gain_ratio_multilabel_missing(pool, a), 3) for a in ASKABLE_ATTRIBUTES}
        print(f"turn {turn}: {message!r}")
        print(f"  route          : {session['route']}")
        print(f"  ask_attribute  : {response['ask_attribute']}   msg: {response['message']!r}")
        print(f"  gain ratios    : {scores}")
        print(f"  confirmed      : { {k: sorted(v) for k, v in session['confirmed'].items()} }")
        print(f"  no-preference  : {sorted(session['no_preference_attributes'])}")
        print(f"  top recs       : {[r['parent_asin'] for r in response['recommendations'][:5]]}")
        print()


if __name__ == "__main__":
    main()
