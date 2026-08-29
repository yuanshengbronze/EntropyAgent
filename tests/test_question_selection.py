from __future__ import annotations

import unittest

from starter.question_selection import (
    choose_next_question,
    gain_ratio_multilabel_missing,
    ground_answer,
    information_cost_score,
    multi_label_entropy,
    normalize_attributes,
)


def _pool(spec: dict[str, list[list[str]]]) -> list[dict]:
    """Build a pool from ``{attribute: [values_per_item, ...]}``."""
    size = max(len(values) for values in spec.values())
    pool: list[dict] = []
    for index in range(size):
        item: dict = {}
        for attribute, per_item in spec.items():
            item[attribute] = list(per_item[index]) if index < len(per_item) else []
        pool.append(item)
    return pool


# Spec section 2 worked example: turn 2, material=cotton confirmed, 20 candidates.
WORKED_EXAMPLE = _pool({
    "color": [["blue"]] * 14 + [["black"]] * 3 + [["red"]] * 3,
    "style": [["crew neck"]] * 15 + [["v-neck"]] * 5,
    "use_case": [["casual"]] * 19 + [["athletic"]],
})

ALL_ASKABLE = {
    "category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case",
}


class MultiLabelEntropyTest(unittest.TestCase):
    def test_matches_hand_computed_value(self) -> None:
        pool = _pool({"x": [["a"]] * 6 + [["b"]] * 4})
        # Two disjoint values at p=0.6 and p=0.4; each binary entropy ~0.97095.
        self.assertAlmostEqual(multi_label_entropy(pool, "x"), 1.9419011889, places=8)

    def test_value_shared_by_every_item_carries_no_information(self) -> None:
        pool = _pool({"x": [["a"]] * 5})
        self.assertEqual(multi_label_entropy(pool, "x"), 0.0)


class GainRatioTest(unittest.TestCase):
    def test_zero_coverage_attribute_scores_zero(self) -> None:
        pool = _pool({"color": [["blue"], ["red"], []], "size": [[], [], []]})
        self.assertEqual(gain_ratio_multilabel_missing(pool, "size"), 0.0)

    def test_gain_ratio_penalises_high_cardinality_split(self) -> None:
        pool = _pool({
            "clean": [["a"]] * 10 + [["b"]] * 10,
            "frag": [[f"v{i}"] for i in range(20)],
        })
        clean = gain_ratio_multilabel_missing(pool, "clean")
        frag = gain_ratio_multilabel_missing(pool, "frag")
        # "frag" has far higher raw entropy but splits into 20 tiny pieces.
        self.assertGreater(multi_label_entropy(pool, "frag"), multi_label_entropy(pool, "clean"))
        self.assertGreater(clean, frag)

    def test_coverage_discount_scales_the_score(self) -> None:
        full = _pool({"color": [["blue"]] * 10 + [["red"]] * 10})
        half = _pool({"color": [["blue"]] * 5 + [["red"]] * 5 + [[]] * 10})
        self.assertAlmostEqual(
            gain_ratio_multilabel_missing(half, "color"),
            0.5 * gain_ratio_multilabel_missing(full, "color"),
            places=8,
        )


class InformationCostScoreTest(unittest.TestCase):
    def test_omega_flips_preference_toward_cheaper_attribute(self) -> None:
        # a: higher gain, higher cost. b: lower gain, free.
        self.assertGreater(
            information_cost_score(1.0, cost=1.0, omega=0.0),
            information_cost_score(0.5, cost=0.0, omega=0.0),
        )
        self.assertLess(
            information_cost_score(1.0, cost=1.0, omega=3.0),
            information_cost_score(0.5, cost=0.0, omega=3.0),
        )


class ChooseNextQuestionTest(unittest.TestCase):
    def test_worked_example_picks_color(self) -> None:
        self.assertEqual(choose_next_question(WORKED_EXAMPLE), "color")

    def test_exhausted_attribute_is_skipped(self) -> None:
        self.assertEqual(choose_next_question(WORKED_EXAMPLE, exhausted={"color"}), "style")

    def test_empty_pool_falls_back_to_other(self) -> None:
        self.assertEqual(choose_next_question([]), "other")

    def test_all_attributes_exhausted_falls_back_to_other(self) -> None:
        self.assertEqual(choose_next_question(WORKED_EXAMPLE, exhausted=ALL_ASKABLE), "other")

    def test_other_also_exhausted_returns_none(self) -> None:
        self.assertIsNone(choose_next_question(WORKED_EXAMPLE, exhausted=ALL_ASKABLE | {"other"}))

    def test_category_and_brand_are_suppressed_not_excluded(self) -> None:
        # brand splits the pool 20 ways (real signal) but its answerability prior
        # is ~0, so a modestly-informative color question still wins.
        pool = _pool({
            "brand": [[f"brand{i}"] for i in range(20)],
            "color": [["blue"]] * 10 + [["red"]] * 10,
        })
        self.assertGreater(gain_ratio_multilabel_missing(pool, "brand"), 0.0)
        self.assertEqual(choose_next_question(pool, askable=("brand", "color")), "color")

    def test_equal_priors_prefer_the_cleaner_split(self) -> None:
        pool = _pool({
            "clean": [["a"]] * 10 + [["b"]] * 10,
            "frag": [[f"v{i}"] for i in range(20)],
        })
        chosen = choose_next_question(
            pool,
            priors={"clean": 0.5, "frag": 0.5},
            askable=("clean", "frag"),
        )
        self.assertEqual(chosen, "clean")


class NormalizeAttributesTest(unittest.TestCase):
    def test_flat_schema_row_is_split_and_cleaned(self) -> None:
        row = {
            "parent_asin": "B07KCFS4VC",
            "category": "clothing",
            "materials": "cotton, polyester, jersey",
            "color": "unknown",
            "size": "unknown",
            "style": "athletic, casual",
            "brand": "Columbia",
            "budget_price": 27.99,
            "feature": "UV protection, moisture-wicking",
            "use_case": "hiking/outdoor, everyday wear",
            "other": "department: men",
        }
        normalized = normalize_attributes(row)
        self.assertEqual(normalized["category"], ["clothing"])
        self.assertEqual(normalized["material"], ["cotton", "polyester", "jersey"])
        self.assertEqual(normalized["color"], [])  # "unknown" dropped
        self.assertEqual(normalized["style"], ["athletic", "casual"])
        self.assertEqual(normalized["brand"], ["columbia"])
        self.assertEqual(normalized["_price"], 27.99)
        self.assertIn("department: men", normalized["other"])


class GroundAnswerTest(unittest.TestCase):
    POOL = _pool({
        "style": [["athletic"], ["casual"], ["retro"], ["formal"]],
        "color": [["blue"], ["black"], ["red"], ["blue"]],
    })

    def test_token_overlap_matches_exact_value(self) -> None:
        grounded = ground_answer("For that, what matters is: casual.", "style", self.POOL)
        self.assertEqual(grounded, {"style": ["casual"]})

    def test_no_preference_answer_grounds_nothing(self) -> None:
        grounded = ground_answer("I don't have a preference for style.", "style", self.POOL)
        self.assertEqual(grounded, {})

    def test_unmatched_answer_returns_empty(self) -> None:
        self.assertEqual(ground_answer("something sparkly", "style", self.POOL), {})

    def test_embed_fn_resolves_a_synonym(self) -> None:
        # "sporty" shares no token with "athletic"; a stub embedder aligns them.
        vectors = {
            "sporty vibe": [1.0, 0.0],
            "athletic": [0.98, 0.2],
            "casual": [0.0, 1.0],
            "retro": [-1.0, 0.0],
            "formal": [0.0, -1.0],
        }
        grounded = ground_answer(
            "sporty vibe", "style", self.POOL,
            embed_fn=lambda texts: {t: vectors[t] for t in texts if t in vectors},
        )
        self.assertEqual(grounded, {"style": ["athletic"]})

    def test_other_matches_across_attributes(self) -> None:
        grounded = ground_answer("blue and casual", "other", self.POOL)
        self.assertEqual(grounded, {"color": ["blue"], "style": ["casual"]})


if __name__ == "__main__":
    unittest.main()
