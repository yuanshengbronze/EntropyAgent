"""Offline multi-turn demonstration of the submission agent.

Runs a short, hand-written shopping conversation through ``Agent`` with the
deterministic path forced on (``OLLAMA_ENABLED=0``) and prints each turn: the
shopper message, the agent reply, its ``ask_attribute``, the top
recommendation, and the cumulative token usage.

This is a usage example, not a test: there are no assertions and it touches
no private agent state. It needs the organizer-provided catalog at
``data/catalog.jsonl`` (or ``--catalog PATH``); the bundled
``assets/catalog_attributes.jsonl`` supplies the structured attributes.

    python -m submission.demo
    python -m submission.demo --catalog /path/to/catalog.jsonl

It never reads ``data/public_set.jsonl``.
"""

from __future__ import annotations

import argparse
import os

if __package__:
    from .agent import Agent
else:  # Supports direct execution: python submission/demo.py
    from agent import Agent


# (message per turn). Edit freely - this is an illustrative probe, not a
# recorded session.
SCENARIO = [
    "I need a men's rain jacket. A key requirement is: waterproof.",
    "For that, what matters is: sealed seams; packable.",
    "I'd like it in navy.",
    "I don't have a preference for that; use your judgment.",
]


def run(catalog_path: str) -> None:
    os.environ.setdefault("OLLAMA_ENABLED", "0")
    agent = Agent(catalog_path)
    agent.reset("demo", {"summary": "offline demonstration session"})

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for turn, message in enumerate(SCENARIO, start=1):
        response = agent.respond("demo", message, turn, 10)
        usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        usage["completion_tokens"] += response["usage"]["completion_tokens"]
        recommendations = response["recommendations"]
        top = recommendations[0]["parent_asin"] if recommendations else "(none)"
        print(f"User:  {message}")
        print(f"Agent: {response['message']}")
        print(f"       ask_attribute: {response['ask_attribute']}")
        print(f"       Top result: {top}")
        print()

    print(
        "Cumulative usage: "
        f"{usage['prompt_tokens']} prompt / {usage['completion_tokens']} completion tokens"
    )


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
