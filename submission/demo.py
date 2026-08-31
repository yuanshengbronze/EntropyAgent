"""Offline multi-turn demonstration of the submission agent.

Rather than scripting the shopper's exact words, this replays a hidden intent
card (a target product, a category, a disclosed requirement, and private
per-attribute preferences) through a small simulated customer, the way an
evaluation harness would. The agent never sees the card; it only sees each
generated message and recovers the intent through its own clarifying questions.
Each turn also reports where the hidden target sits in the agent's ranking.

The deterministic path is forced on (``OLLAMA_ENABLED=0``). It needs the
organizer catalog at ``data/catalog.jsonl`` (or ``--catalog PATH``); the bundled
``assets/catalog_attributes.jsonl`` supplies the structured attributes. It never
reads ``data/public_set.jsonl``.

    python -m submission.demo
    python -m submission.demo --catalog /path/to/catalog.jsonl
"""

from __future__ import annotations

import argparse
import os

if __package__:
    from .agent import Agent
else:  # Supports direct execution: python submission/demo.py
    from agent import Agent


# The hidden shopper intent. ``target_parent_asin`` is the product the shopper
# actually wants; the agent must surface it without ever seeing the card. A
# preference is revealed only when the agent asks about that attribute; every
# other attribute draws "no preference". Set ``route`` to "browsing" to drop the
# disclosed requirement. Edit freely.
INTENT_CARD = {
    "target_parent_asin": "B07TGH64XN",  # a real parent_asin from the catalog
    "category": "men's rain jacket",
    "route": "buying",
    "requirement": "waterproof",
    "preferences": {"feature": ["breathable"]},
}

MAX_TURNS = 10


class SimulatedCustomer:
    """Generates shopper turns from a fixed intent card.

    The phrasing matches what the agent is built to parse: an opening message
    that states the category (plus a key requirement when buying), a direct
    requirement phrase when asked about a known preference, and an explicit
    "no preference" otherwise.
    """

    def __init__(self, card: dict) -> None:
        self._card = card

    def opening(self) -> str:
        category = self._card["category"]
        if self._card.get("route") == "buying" and self._card.get("requirement"):
            return (
                f"I'm looking for a {category}. "
                f"A key requirement is: {self._card['requirement']}."
            )
        return f"I'm looking for a {category}, but I'm still exploring."

    def reply(self, ask_attribute: str | None) -> str | None:
        if not ask_attribute:
            return None
        values = self._card.get("preferences", {}).get(ask_attribute)
        if values:
            return f"For that, what matters is: {'; '.join(values)}."
        return "I don't have a preference for that; use your judgment."


def _target_rank(recommendations: list[dict], target: str | None) -> int | None:
    """1-based position of the hidden target in this turn's list, or None."""
    if not target:
        return None
    for position, item in enumerate(recommendations, start=1):
        if item["parent_asin"] == target:
            return position
    return None


def run(catalog_path: str) -> None:
    os.environ.setdefault("OLLAMA_ENABLED", "0")
    agent = Agent(catalog_path)
    agent.reset("demo", {"summary": "offline demonstration session"})
    customer = SimulatedCustomer(INTENT_CARD)
    target = INTENT_CARD.get("target_parent_asin")

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    first_hit: tuple[int, int] | None = None  # (turn, rank)
    message = customer.opening()

    for turn in range(1, MAX_TURNS + 1):
        if message is None:
            break
        response = agent.respond("demo", message, turn, 10)
        usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        usage["completion_tokens"] += response["usage"]["completion_tokens"]
        recommendations = response["recommendations"]
        top = recommendations[0]["parent_asin"] if recommendations else "(none)"
        rank = _target_rank(recommendations, target)
        if rank is not None and first_hit is None:
            first_hit = (turn, rank)

        print(f"User:  {message}")
        print(f"Agent: {response['message']}")
        print(f"       ask_attribute: {response['ask_attribute']}")
        print(f"       Top result: {top}")
        if target:
            print(f"       Target {target}: {'rank ' + str(rank) if rank else 'not in Top 10'}")
        print()

        message = customer.reply(response["ask_attribute"])

    print(
        "Cumulative usage: "
        f"{usage['prompt_tokens']} prompt / {usage['completion_tokens']} completion tokens"
    )
    if target:
        if first_hit:
            print(f"Target entered the Top 10 on turn {first_hit[0]} at rank {first_hit[1]}.")
        else:
            print("Target never entered the Top 10.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="data/catalog.jsonl",
        help="Path to the organizer-provided frozen catalog JSONL.",
    )
    run(parser.parse_args().catalog)


if __name__ == "__main__":
    main()
