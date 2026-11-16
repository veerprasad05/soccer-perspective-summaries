"""Explicit source adapter. Input mapping must be verified against inspected files."""
import argparse
import json
import re
from pathlib import Path
from .common import PARSER_VERSION, atomic_jsonl, read_jsonl, sha256_file

def parse_time(value):
    if value is None: return None, None
    if isinstance(value, int) and value >= 0: return value, None
    m = re.fullmatch(r"(\d{1,3})(?:\+(\d{1,2}))?'?", str(value).strip())
    if not m: return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None

def normalize_row(row, source_file, line_no):
    """Adapter for inspected, already structured JSONL records with explicit A/B mapping.

    Never derives a final score from goals or parses BBC article prose.
    """
    mid = row.get("match_id")
    if not isinstance(mid, str) or not mid.strip(): raise ValueError(f"{source_file}:{line_no}: missing stable match_id")
    teams = row.get("teams") or {}
    score = row.get("score") or {}
    events = []
    for idx, e in enumerate(row.get("events") or []):
        minute, stoppage = parse_time(e.get("minute"))
        events.append({"event_id": f"{mid}:event:{idx:04d}", "minute": minute, "stoppage_minute": stoppage,
                       "type": e.get("type"), "team": e.get("team") if e.get("team") in ("A", "B") else None,
                       "player": e.get("player"), "source_reference": f"{source_file}:{line_no}:events[{idx}]"})
    events.sort(key=lambda e: (e["minute"] is None, e["minute"] if e["minute"] is not None else 9999, e["stoppage_minute"] or 0, e["event_id"]))
    return {"match_id": mid.strip(), "source": row.get("source") or "soccernet-bbc", "competition": row.get("competition"),
            "date": row.get("date"), "team_a": teams.get("A"), "team_b": teams.get("B"),
            "home_away_mapping": row.get("home_away_mapping") or {"A": "home", "B": "away"},
            "score": {"A": score.get("A"), "B": score.get("B")}, "events": events,
            "statistics": row.get("statistics") or {}, "source_files": [source_file], "parse_warnings": [],
            "parser_version": PARSER_VERSION}

def normalize(source, output):
    source = Path(source)
    if source.suffix.lower() == ".zip":
        from .soccernet_bbc import load_archive
        rows=load_archive(source)
    elif source.suffix.lower() == ".jsonl":
        rows = [normalize_row(row, str(source), i) for i, row in enumerate(read_jsonl(source), 1)]
    else:
        raise ValueError("Expected inspected SoccerNet-BBC ZIP or explicitly mapped JSONL")
    atomic_jsonl(output, rows)
    return {"rows": len(rows), "source_sha256": sha256_file(source), "output_sha256": sha256_file(output), "parser_version": PARSER_VERSION}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--output", default="data/processed/matches.jsonl")
    a = p.parse_args(); print(json.dumps(normalize(a.source, a.output), indent=2))
if __name__ == "__main__": main()
