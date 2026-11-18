import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.sportssum import aggregate, evaluate, main, reference_sets, score_sets


def fixture():
    return {
        "team_a": "North City", "team_b": "South United", "goal_events_complete": True,
        "events": [
            {"player": "Alice Smith", "type": "goal", "goal_type": "regular"},
            {"player": "Bob Jones", "type": "yellow_card"},
            {"player": "Cara Doe", "type": "red_card"},
            {"player": "Dave Lee", "replaced_player": "Erin Kay", "type": "substitution"},
            {"player": "Frank Foo", "type": "goal", "goal_type": "own_goal"},
        ],
    }


class SportsSumTests(unittest.TestCase):
    def test_paper_style_set_f1_deduplicates_and_defines_empty_case(self):
        result = score_sets(["alice", "alice", "bob"], ["alice", "carol"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["f1"], 0.5)
        self.assertIsNone(score_sets([], [])["f1"])

    def test_reference_uses_subject_event_pairs(self):
        names, events = reference_sets(fixture())
        self.assertEqual(len(names), 6)
        self.assertIn(("frank foo", "own_goal"), events)
        self.assertIn(("dave lee", "sub_on"), events)
        self.assertIn(("erin kay", "sub_off"), events)

    def test_synonyms_surnames_unknown_name_and_team_filter(self):
        text = ("North City won. Smith netted, Jones received a yellow card, "
                "Doe was sent off. Lee replaced Kay. An own goal by Foo. Fake Name scored.")
        result = evaluate(fixture(), text)
        self.assertEqual((result["name"]["tp"], result["name"]["fp"], result["name"]["fn"]), (6, 1, 0))
        self.assertEqual((result["event"]["tp"], result["event"]["fp"], result["event"]["fn"]), (6, 1, 0))
        self.assertEqual((result["salient_event"]["tp"], result["salient_event"]["fp"]), (3, 1))
        self.assertEqual(result["unresolved_action_names"], ["unresolved:fake name"])
        self.assertNotIn("north city", result["predicted_names"])

    def test_accent_fold_possessive_and_goal_list(self):
        record = {"team_a": "AS Roma", "team_b": "Empoli", "goal_events_complete": True,
                  "events": [{"player": "James Rodríguez", "type": "goal"},
                             {"player": "Rade Krunić", "type": "yellow_card"}]}
        result = evaluate(record, "Goals from Rodriguez put Roma ahead. Empoli's Rade Krunic received a yellow card.")
        self.assertEqual(result["name"]["tp"], 2)
        self.assertEqual(result["name"]["fp"], 0)
        self.assertIn(["rade krunic", "yellow_card"], result["predicted_events"])

    def test_complete_goal_subset_uses_only_flagged_matches(self):
        record = fixture()
        first = {"match_id": "one", "condition": "neutral", **evaluate(record, "Smith scored.")}
        record["goal_events_complete"] = False
        second = {"match_id": "two", "condition": "neutral", **evaluate(record, "Smith scored.")}
        table = aggregate([first, second])
        self.assertEqual([r["n_matches"] for r in table], [2, 1])

    def test_event_lists_stop_at_other_clauses_and_own_goal_stays_distinct(self):
        record = fixture()
        result = evaluate(record, "Goals from Smith (subbed by Lee) won it. Yellow cards for Jones, and a red for Doe. Foo scored an own goal.")
        predicted = {tuple(x) for x in result["predicted_events"]}
        self.assertIn(("alice smith", "goal"), predicted)
        self.assertIn(("bob jones", "yellow_card"), predicted)
        self.assertIn(("frank foo", "own_goal"), predicted)
        self.assertNotIn(("dave lee", "goal"), predicted)
        self.assertNotIn(("cara doe", "yellow_card"), predicted)
        self.assertNotIn(("frank foo", "goal"), predicted)

    def test_cli_rejects_partial_condition_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = root / "ids.json"
            matches = root / "matches.jsonl"
            generations = root / "generations.jsonl"
            ids.write_text(json.dumps(["one", "two"]))
            matches.write_text("\n".join(json.dumps({"match_id": mid, **fixture()}) for mid in ("one", "two")) + "\n")
            generations.write_text(json.dumps({"match_id": "one", "condition": "neutral", "status": "complete",
                                               "run_id": "example", "output_text": "Smith scored."}) + "\n")
            argv = ["sportssum", "--matches", str(matches), "--generations", str(generations),
                    "--split-ids", str(ids), "--output", str(root / "out.csv"),
                    "--details", str(root / "out.jsonl")]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "Missing 1 match/condition"):
                main()
            self.assertFalse((root / "out.csv").exists())


if __name__ == "__main__":
    unittest.main()
