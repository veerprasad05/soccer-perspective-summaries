import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import stance_judge


def generation(match_id, arm, text):
    return {"run_id": match_id + ":" + arm, "match_id": match_id,
            "split": "dev", "condition": "neutral", "coverage_mode": "concise",
            "status": "complete", "evaluation_condition": arm,
            "output_text": text, "split_manifest_sha256": "manifest"}


def response(stance=0):
    team = "City" if stance < 0 else "United" if stance > 0 else "neither"
    return {"id": "resp-1", "model": "gpt-5.6-luna", "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "output": [{"type": "message", "content": [{"type": "output_text",
                        "text": json.dumps({"favored_team": team, "strength": abs(stance), "emotion": 1,
                                            "uncertain": False, "rationale": "Plain wording."})}]}]}


class FakeClient:
    def __init__(self):
        self.calls = []

    def create(self, body):
        self.calls.append(body)
        return response()


class StanceJudgeTests(unittest.TestCase):
    def test_targeted_shift_uses_proposals_negative_a_positive_b_scale(self):
        self.assertEqual(stance_judge.targeted_shift(1, -2), 2)
        self.assertEqual(stance_judge.targeted_shift(-1, 2), 2)
        self.assertEqual(stance_judge.targeted_shift(1, 2), -2)

    def test_request_is_blinded_and_schema_is_validated(self):
        record = {"team_a": "City", "team_b": "United", "score": {"A": 1, "B": 0}}
        row = generation("one", "learned_alpha_+1", "City won 1-0.")
        body = stance_judge.request_body(record, row, "gpt-5.6-luna", "low")
        self.assertIn("City won 1-0.", body["input"])
        self.assertNotIn("learned_alpha", body["input"])
        self.assertNotIn("alpha", body["instructions"])
        self.assertIn("Team A = City", body["input"])
        self.assertEqual(body["text"]["format"]["schema"]["properties"]["favored_team"]["enum"],
                         ["City", "United", "neither"])
        self.assertFalse(body["store"])
        self.assertEqual(stance_judge.parse_response(response(-2), record)["stance"], -2)
        madrid = {"team_a": "Real Madrid", "team_b": "Granada CF"}
        self.assertEqual(stance_judge.normalize_rating({
            "favored_team": "Real Madrid", "strength": 1, "emotion": 1,
            "uncertain": False, "rationale": "The summary favors Real Madrid."}, madrid)["stance"], -1)
        with self.assertRaisesRegex(ValueError, "violates"):
            stance_judge.parse_response(response(4), record)
        inconsistent = response()
        inconsistent["output"][0]["content"][0]["text"] = json.dumps({
            "favored_team": "City", "strength": 0, "emotion": 1,
            "uncertain": False, "rationale": "Wrong strength."})
        with self.assertRaisesRegex(ValueError, "violates"):
            stance_judge.parse_response(inconsistent, record)
        refused = response()
        refused["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
        with self.assertRaisesRegex(ValueError, "refused"):
            stance_judge.parse_response(refused, record)

    def test_selection_balances_arms_and_requires_no_hook(self):
        rows = [generation(mid, arm, "text") for mid in ("one", "two", "three")
                for arm in ("none_alpha_0", "learned_alpha_+1")]
        selected = stance_judge.select_rows(rows, pilot_matches=2, seed=1)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({r["match_id"] for r in selected}), 2)
        with self.assertRaisesRegex(ValueError, "none_alpha_0"):
            stance_judge.select_rows(rows, arms=["learned_alpha_+1"])

    def test_run_reuses_identical_blinded_requests_and_resumes(self):
        records = {mid: {"team_a": "City", "team_b": "United", "score": {"A": 1, "B": 0}}
                   for mid in ("one", "two")}
        arms = ("none_alpha_0", "learned_alpha_+1")
        rows = [generation(mid, arm, "same text" if mid == "one" else arm)
                for mid in records for arm in arms]
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "judgments.jsonl")
            args = SimpleNamespace(generations="unused", matches="unused", dev_ids="unused",
                                   arms=None, pilot_matches=None, seed=42,
                                   model="gpt-5.6-luna", effort="low",
                                   max_output_tokens=600,
                                   output=output, dry_run=False)
            client = FakeClient()
            with patch.object(stance_judge, "load_dev", return_value=(records, rows)):
                stance_judge.run_judge(args, client)
                stance_judge.run_judge(args, client)
                report_args = SimpleNamespace(generations="unused", matches="unused",
                                              dev_ids="unused", judgments=output,
                                              output=str(Path(directory) / "summary.csv"))
                stance_judge.report(report_args)
            saved = list(stance_judge.read_jsonl(output))
            self.assertEqual(len(saved), 4)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(sum(row["api_call_made"] for row in saved), 3)
            with open(report_args.output, encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            sheet, key = stance_judge.audit_rows(records, rows, saved, arms, count=1, seed=1)
            self.assertEqual(len(sheet), 2)
            self.assertNotIn("arm", sheet[0])
            self.assertIn("arm", key[0])
            for row in sheet:
                row["human_stance_minus3_to_plus3"] = "0"
            sheet_path, key_path = Path(directory) / "sheet.csv", Path(directory) / "key.csv"
            stance_judge.write_csv(sheet_path, sheet)
            stance_judge.write_csv(key_path, key)
            with redirect_stdout(io.StringIO()) as output_stream:
                stance_judge.compare_human(SimpleNamespace(sheet=sheet_path, key=key_path))
            self.assertEqual(json.loads(output_stream.getvalue())["exact_agreement"], 1.0)

    def test_loader_rejects_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f"match-{i}" for i in range(50)]
            (root / "dev.json").write_text(json.dumps(ids))
            (root / "matches.jsonl").write_text("\n".join(
                json.dumps({"match_id": mid, "team_a": "A", "team_b": "B"}) for mid in ids) + "\n")
            rows = [generation(mid, "none_alpha_0", "text") for mid in ids]
            rows[0]["split"] = "test"
            (root / "generations.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            with self.assertRaisesRegex(ValueError, "dev steering rows"):
                stance_judge.load_dev(root / "generations.jsonl", root / "matches.jsonl", root / "dev.json")


if __name__ == "__main__":
    unittest.main()
