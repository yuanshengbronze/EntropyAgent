from __future__ import annotations

import unittest

from starter.question_selection import (
    ASKABLE_ATTRIBUTES,
    choose_next_question,
    gain_ratio_multilabel_missing,
    ground_answer,
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

ALL_ASKABLE = set(ASKABLE_ATTRIBUTES)


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


class ChooseNextQuestionTest(unittest.TestCase):
    def test_worked_example_prefers_a_clean_binary_split(self) -> None:
        # Without the answerability cost term the pick is pure gain ratio. Any
        # clean 2-value split normalises to gain ratio 2.0 (H = 2*Hb, SI = Hb),
        # so style (15/5) and use_case (19/1) both beat color's 3-way 14/3/3
        # spread; style wins on candidate order.
        self.assertEqual(choose_next_question(WORKED_EXAMPLE), "style")

    def test_exhausted_attribute_is_skipped(self) -> None:
        self.assertEqual(
            choose_next_question(WORKED_EXAMPLE, exhausted={"style"}), "use_case"
        )

    def test_empty_pool_falls_back_to_other(self) -> None:
        self.assertEqual(choose_next_question([]), "other")

    def test_all_attributes_exhausted_falls_back_to_other(self) -> None:
        self.assertEqual(choose_next_question(WORKED_EXAMPLE, exhausted=ALL_ASKABLE), "other")

    def test_other_also_exhausted_returns_none(self) -> None:
        self.assertIsNone(choose_next_question(WORKED_EXAMPLE, exhausted=ALL_ASKABLE | {"other"}))

    def test_category_and_brand_are_not_in_the_default_candidate_set(self) -> None:
        self.assertNotIn("category", ASKABLE_ATTRIBUTES)
        self.assertNotIn("brand", ASKABLE_ATTRIBUTES)
        # A pool that only disagrees on category/brand has nothing askable left.
        pool = _pool({
            "category": [["shirts"]] * 12 + [["pants"]] * 8,
            "brand": [[f"brand{i}"] for i in range(20)],
        })
        self.assertEqual(choose_next_question(pool), "other")

    def test_the_cleaner_split_wins_on_gain_ratio_alone(self) -> None:
        pool = _pool({
            "clean": [["a"]] * 10 + [["b"]] * 10,
            "frag": [[f"v{i}"] for i in range(20)],
        })
        self.assertEqual(
            choose_next_question(pool, askable=("clean", "frag")), "clean"
        )


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
