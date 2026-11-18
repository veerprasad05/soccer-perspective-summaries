import unittest

from src.activations import summary_positions, validate_train_pairs
from src.steering import (add_last_token, arm_label, experiment_grid, generate_with_direction,
                          parse_seeds, parse_strengths, validate_baselines,
                          validate_vector_metadata)


def pair_row(condition):
    return {"match_id": "match-1", "split": "train", "condition": condition,
            "status": "complete", "coverage_mode": "concise",
            "prompt_version": "record-concise-v1", "revision": "revision-1",
            "split_manifest_sha256": "manifest-1", "mock": False,
            "output_text": "A factual soccer summary.", "input_hash": "same-facts",
            "affinity": "casual", "settings": {"do_sample": False}}


class FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)

    def to(self, **_kwargs):
        return self

    def __add__(self, other):
        return FakeVector([a + b for a, b in zip(self.values, other.values)])


class FakeHidden:
    ndim = 3
    shape = (1, 2, 2)
    device = "cpu"
    dtype = "float"

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def clone(self):
        return FakeHidden(self.rows)

    def __getitem__(self, key):
        if key != (slice(None), -1, slice(None)):
            raise AssertionError(key)
        return FakeVector(self.rows[-1])

    def __setitem__(self, key, value):
        if key != (slice(None), -1, slice(None)):
            raise AssertionError(key)
        self.rows[-1] = value.values


class FakeHandle:
    def __init__(self, module):
        self.module = module

    def remove(self):
        self.module.hook = None


class FakeModule:
    def __init__(self):
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return FakeHandle(self)


class FakeGenerator:
    def __init__(self, fail=False):
        self.module = FakeModule()
        self.model = type("Model", (), {"model": type("Core", (), {"layers": [self.module]})()})()
        self.fail = fail
        self.hook_seen = False

    def generate(self, _messages, _settings):
        self.hook_seen = self.module.hook is not None
        if self.fail:
            raise RuntimeError("generation failed")
        return "text"


class SteeringTests(unittest.TestCase):
    def test_train_pairs_reject_wrong_prompt_duplicates_and_mismatch(self):
        rows = [pair_row("fan_A"), pair_row("fan_B")]
        self.assertEqual(set(validate_train_pairs(rows, {"match-1"}, "revision-1", "manifest-1")),
                         {"match-1"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_train_pairs(rows + [pair_row("fan_A")], {"match-1"}, "revision-1", "manifest-1")
        bad = [dict(rows[0]), dict(rows[1])]
        bad[1]["input_hash"] = "other-facts"
        with self.assertRaisesRegex(ValueError, "Mismatched factual input"):
            validate_train_pairs(bad, {"match-1"}, "revision-1", "manifest-1")
        bad[1]["input_hash"] = "same-facts"
        bad[1]["prompt_version"] = "baseline-v1"
        with self.assertRaisesRegex(ValueError, "record-concise-v1"):
            validate_train_pairs(bad, {"match-1"}, "revision-1", "manifest-1")

    def test_summary_span_excludes_chat_special_tokens(self):
        offsets = [(0, 0), (0, 7), (8, 11), (12, 16), (0, 0)]
        self.assertEqual(summary_positions(offsets, 8, 16), [2, 3])
        with self.assertRaises(ValueError):
            summary_positions(offsets, 20, 25)

    def test_grid_has_zero_learned_signs_and_matched_random_controls(self):
        grid = experiment_grid(parse_strengths("0.5,1"), parse_seeds("17,29"))
        self.assertEqual(grid[0], ("none", 0.0, None))
        self.assertEqual(len(grid), 1 + 2 * 2 * (1 + 2))
        self.assertIn(("learned", -1.0, None), grid)
        self.assertIn(("random", 1.0, 29), grid)
        self.assertEqual(arm_label("none", 0, None), "none_alpha_0")
        self.assertEqual(arm_label("learned", -1, None), "learned_alpha_-1")
        self.assertEqual(arm_label("random", 1, 29), "random_alpha_+1_seed_29")
        with self.assertRaises(ValueError):
            parse_strengths("0,1")

    def test_dev_requires_full_concise_train_vector(self):
        config = {"model_id": "Qwen", "revision": "revision-1"}
        metadata = {"model_id": "Qwen", "revision": "revision-1",
                    "method": "paired_mean_last_summary_token_A_minus_B_v1",
                    "split_manifest_sha256": "manifest-1", "pair_prompt_version": "record-concise-v1",
                    "direction_sign": "fan_A minus fan_B", "layers": [12],
                    "n_pairs": 2, "smoke_test": True}
        with self.assertRaisesRegex(ValueError, "all 200"):
            validate_vector_metadata(metadata, config, "manifest-1", 12, "dev")
        validate_vector_metadata(metadata, config, "manifest-1", 12, "train")

    def test_mock_neutral_baseline_is_rejected(self):
        row = {"match_id": "match-1", "condition": "neutral", "status": "complete",
               "coverage_mode": "concise", "mock": True,
               "split_manifest_sha256": "manifest-1"}
        with self.assertRaisesRegex(ValueError, "completed concise neutral"):
            validate_baselines([row], {"match-1"}, "manifest-1")

    def test_addition_changes_only_last_token_without_mutating_source(self):
        original = FakeHidden([[1, 2], [3, 4]])
        result = add_last_token((original, "cache"), FakeVector([10, -10]))
        self.assertEqual(original.rows, [[1, 2], [3, 4]])
        self.assertEqual(result[0].rows, [[1, 2], [13, -6]])
        self.assertEqual(result[1], "cache")

    def test_zero_has_no_hook_and_hook_cleans_up_after_failure(self):
        generator = FakeGenerator()
        self.assertEqual(generate_with_direction(generator, [], {}, 0), "text")
        self.assertFalse(generator.hook_seen)
        self.assertEqual(generate_with_direction(generator, [], {}, 0, FakeVector([1, 2])), "text")
        self.assertTrue(generator.hook_seen)
        self.assertIsNone(generator.module.hook)
        failing = FakeGenerator(fail=True)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            generate_with_direction(failing, [], {}, 0, FakeVector([1, 2]))
        self.assertIsNone(failing.module.hook)


if __name__ == "__main__":
    unittest.main()
