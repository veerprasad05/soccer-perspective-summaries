import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from src.data.split import create_split, smoke_ids
from src.data.normalize import normalize_row, parse_time
from src.data.validate import reasons
from src.prompting.render import render_match_facts, input_hash
from src.prompting.templates import build_messages
from src.generate import run, MockGenerator
from src.evaluate import evaluate
from src.data.common import atomic_jsonl
from src.data.soccernet_bbc import parse_match

def fixture(i):
    return {"match_id":f"synthetic-{i:04d}","team_a":f"Home{i}","team_b":f"Away{i}","score":{"A":1,"B":0},
            "events":[{"event_id":f"e{i}","minute":45,"stoppage_minute":2,"type":"goal","team":"A","player":"Alice"}],
            "statistics":{},"competition":None,"date":None,"source_files":["SYNTHETIC"]}

class PipelineTests(unittest.TestCase):
    def test_exact_split_determinism_and_shortage(self):
        rows=[fixture(i) for i in range(500)]
        with tempfile.TemporaryDirectory() as d:
            m=create_split(rows,d,300,100,100)
            self.assertEqual(m["counts"],{"train":300,"dev":100,"test":100})
            before=(Path(d)/"manifest.json").read_bytes()
            self.assertEqual(m,create_split(list(reversed(rows)),d,300,100,100))
            self.assertEqual(len(smoke_ids(d,2)),2)
            self.assertEqual(before,(Path(d)/"manifest.json").read_bytes())
            ids=[set(json.loads((Path(d)/f"{x}.json").read_text())) for x in ("train","dev","test")]
            self.assertFalse(ids[0]&ids[1] or ids[1]&ids[2] or ids[0]&ids[2])
            with self.assertRaisesRegex(ValueError,"shortage 1"):create_split(rows[:-1],Path(d)/"short",300,100,100)
    def test_approved_299_split(self):
        rows=[fixture(i) for i in range(299)]
        with tempfile.TemporaryDirectory() as d:
            m=create_split(rows,d)
            self.assertEqual(m["counts"],{"train":200,"dev":50,"test":49})
            with self.assertRaisesRegex(ValueError,"shortage 1"):
                create_split(rows[:-1],Path(d)/"short")
    def test_normalize_render_and_unknown(self):
        raw={"match_id":"x","teams":{"A":"<A>","B":"B"},"score":{"A":2,"B":1},"events":[{"minute":"90+3","type":"goal","team":"A","player":None},{"minute":"12","type":"card","team":"B"}]}
        r=normalize_row(raw,"synthetic.jsonl",1)
        self.assertEqual(parse_time("90+3"),(90,3))
        self.assertEqual([e["minute"] for e in r["events"]],[12,90])
        self.assertIsNone(r["events"][1]["player"])
        facts=render_match_facts(r)
        self.assertIn("90+3'",facts);self.assertIn("&lt;A&gt;",facts)
        self.assertEqual(input_hash(r),input_hash(r))
        for c in ("neutral","fan_A","fan_B"):
            self.assertEqual(build_messages(r,c)[1]["content"].split("Verified match facts:\n")[1],facts)
    def test_record_coverage_prompts_share_authoritative_input(self):
        record = fixture(1)
        record["statistics"] = {"shots": {"A": 10, "B": 8}}
        concise = build_messages(record, "fan_A", coverage_mode="concise")
        complete = build_messages(record, "fan_A", coverage_mode="complete")
        self.assertEqual(concise[1], complete[1])
        self.assertIn("final truth", concise[0]["content"])
        self.assertIn("Never invent, infer, or contradict", concise[0]["content"])
        self.assertIn("150 words", concise[0]["content"])
        self.assertIn("every listed event", complete[0]["content"])
        self.assertIn("every listed statistic", complete[0]["content"])
        self.assertNotIn("every listed event", concise[0]["content"])
    def test_validate(self):
        r=fixture(1);self.assertEqual(reasons(r),[])
        r["score"]["A"]=0;self.assertIn("goal_count_exceeds_score",reasons(r))
    def test_archive_adapter_deduplicates_and_checks_source(self):
        folder="SoccerNet-BBC/7_epl 2014-2015 2015-02-21 - 18-00 Home 1 - 1 Away BBC.com"
        commentary="""Post update
Match ends, Home 1, Away 1.
50'
goal
Goal!
Goal! Home 1, Away 1. Alice (Home) scores.
50'
goal
Goal!
Goal! Home 1, Away 1. Alice (Home) scores.
10'
goal
Goal!
Goal! Home 0, Away 1. Bob (Away) scores.
"""
        stats="Home Team\nHome\nAway Team\nAway\nShots\nHome\n4\nAway\n3\n"
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"source.zip"
            with zipfile.ZipFile(path,"w") as z:
                for name,content in {"commentary.txt":commentary,"match_stats.txt":stats,"info.txt":'{}',"lineup.txt":""}.items():
                    z.writestr(folder+"/"+name,content)
            with zipfile.ZipFile(path) as z:r=parse_match(z,folder,"synthetic-sha")
            goals=[e for e in r["events"] if e["type"]=="goal"]
            self.assertEqual(len(goals),2)
            self.assertEqual([e["team"] for e in goals],["B","A"])
            self.assertTrue(r["goal_events_complete"])
            self.assertEqual(len(goals[1]["source_references"]),2)
            self.assertEqual(reasons(r),[])
    def test_mock_two_matches_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); records=[fixture(i) for i in range(2)]
            atomic_jsonl(d/"matches.jsonl",records)
            create_split(records,d/"splits",2,0,0)
            cfg={"splits_dir":str(d/"splits"),"matches":str(d/"matches.jsonl"),"model_id":"mock","revision":"synthetic","affinity":"casual","max_new_tokens":10,"do_sample":False,"seed":42}
            out=d/"out.jsonl"
            run(cfg,"train",["neutral","fan_A","fan_B"],out,MockGenerator())
            rows=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertEqual(len({r["input_hash"] for r in rows}),2)
            self.assertTrue(all(r["status"]=="complete" for r in rows))
            self.assertEqual(run(cfg,"train",["neutral","fan_A","fan_B"],out,MockGenerator()),[])
            self.assertEqual(len(out.read_text().splitlines()),6)
    def test_record_coverage_modes_have_distinct_runs_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_jsonl(root/"matches.jsonl", [fixture(1), fixture(2)])
            create_split([fixture(1), fixture(2)], root/"splits", 2, 0, 0)
            config = {"splits_dir": str(root/"splits"), "matches": str(root/"matches.jsonl"),
                      "model_id": "mock", "revision": "synthetic", "affinity": "casual",
                      "max_new_tokens": 250, "do_sample": False, "seed": 42}
            concise_path, complete_path = root/"concise.jsonl", root/"complete.jsonl"
            run(config, "train", ["fan_A"], concise_path, MockGenerator(), coverage_mode="concise")
            run(config, "train", ["fan_A"], complete_path, MockGenerator(), coverage_mode="complete")
            concise = [json.loads(line) for line in concise_path.read_text().splitlines()]
            complete = [json.loads(line) for line in complete_path.read_text().splitlines()]
            self.assertEqual(len(concise), len(complete))
            self.assertEqual([r["input_hash"] for r in concise], [r["input_hash"] for r in complete])
            self.assertNotEqual([r["run_id"] for r in concise], [r["run_id"] for r in complete])
            self.assertEqual({r["prompt_version"] for r in complete}, {"record-complete-v1"})
            self.assertEqual({r["settings"]["max_new_tokens"] for r in complete}, {250})
            self.assertEqual({r["output_word_count"] for r in complete}, {6})
            with self.assertRaisesRegex(ValueError, "another coverage mode"):
                run(config, "train", ["fan_A"], concise_path, MockGenerator(), coverage_mode="complete")
    def test_generation_records_token_cap_diagnostic(self):
        class TokenGenerator:
            def generate(self, messages, settings):
                self.last_generated_tokens = settings["max_new_tokens"]
                self.last_hit_max_new_tokens = True
                return "One sentence."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = fixture(1)
            atomic_jsonl(root/"matches.jsonl", [record])
            create_split([record], root/"splits", 1, 0, 0)
            config = {"splits_dir": str(root/"splits"), "matches": str(root/"matches.jsonl"),
                      "model_id": "test", "revision": "synthetic", "affinity": "casual",
                      "max_new_tokens": 250, "do_sample": False, "seed": 42}
            output = root/"out.jsonl"
            run(config, "train", ["neutral"], output, TokenGenerator(), coverage_mode="complete")
            row = json.loads(output.read_text().splitlines()[0])
            self.assertEqual((row["output_word_count"], row["output_token_count"],
                              row["hit_max_new_tokens"]), (2, 250, True))
    def test_evaluator(self):
        r=fixture(1)
        good=evaluate(r,"Home1 1-0 Away1. Alice scored.")
        bad=evaluate(r,"Home1 0-1 Away1. Bob scored.")
        self.assertEqual(good["contradicted"],0)
        self.assertGreater(bad["contradicted"],0)
        self.assertIsNone(evaluate(r,"A lively contest.")["fact_precision"])
        self.assertEqual(evaluate(r,"Bob scored.")["claims"][0]["status"],"unverifiable")
        self.assertEqual(evaluate(r,"Alice scored in the 44th minute.")["claims"][-1]["status"],"contradicted")
        self.assertEqual(evaluate(r,"Alice scored in the 45+2 minute.")["claims"][-1]["status"],"supported")
        self.assertEqual(evaluate(r,"Alice scored. Alice scored.")["event_denominator"],1)
        self.assertEqual(evaluate(r,"Away1 beat Home1.")["claims"][0]["status"],"contradicted")
        r["score"]={"A":None,"B":None}
        self.assertEqual(evaluate(r,"Home1 1-0 Away1.")["claims"][0]["status"],"unverifiable")
    def test_evaluator_does_not_reverse_opponent_score_or_count_shots(self):
        r=fixture(1)
        r["score"]={"A":0,"B":1}
        report=evaluate(r,"Away1 defeated Home1 1-0. Home1 had 14-12 shots on target.")
        scores=[claim for claim in report["claims"] if claim["kind"]=="score"]
        self.assertEqual(len(scores),1)
        self.assertEqual(scores[0]["status"],"supported")
    def test_evaluator_distinguishes_team_from_scorer_and_accepts_unique_surname(self):
        r=fixture(1)
        r["events"][0]["player"]="Alice Smith"
        r["goal_events_complete"]=True
        report=evaluate(r,"Home1 scored after Smith scored in the 45+2 minute.")
        scorer=[claim for claim in report["claims"] if claim["kind"]=="scorer"]
        self.assertEqual(len(scorer),1)
        self.assertEqual(scorer[0]["status"],"supported")
if __name__=="__main__":unittest.main()
