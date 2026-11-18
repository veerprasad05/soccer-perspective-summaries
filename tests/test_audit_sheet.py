import unittest

from src.audit_sheet import build_rows


class AuditSheetTests(unittest.TestCase):
    def test_dev_only_blinded_balanced_sheet_is_deterministic(self):
        records = []
        generations = []
        for i, score in enumerate(((1, 0), (2, 1), (0, 1), (1, 3), (0, 0), (2, 2))):
            mid = f"dev-{i}"
            records.append({"match_id": mid, "team_a": f"Home{i}", "team_b": f"Away{i}",
                            "score": {"A": score[0], "B": score[1]}, "events": [], "statistics": {}})
            for condition in ("neutral", "fan_A", "fan_B"):
                generations.append({"match_id": mid, "condition": condition, "status": "complete",
                                    "output_text": f"Summary {i}", "run_id": f"{mid}-{condition}"})
        dev_ids = {r["match_id"] for r in records}
        expected = build_rows(records, generations, dev_ids)
        self.assertEqual(expected, build_rows(records, generations, dev_ids))
        blinded, key = expected
        self.assertEqual(len(blinded), 18)
        self.assertEqual(len(key), 18)
        self.assertNotIn("condition", blinded[0])
        self.assertEqual({k["match_id"] for k in key}, dev_ids)
        self.assertTrue(all(len([k for k in key if k["match_id"] == mid]) == 3 for mid in dev_ids))
        with self.assertRaises(ValueError):
            build_rows(records, generations[:-1], dev_ids)


if __name__ == "__main__":
    unittest.main()
