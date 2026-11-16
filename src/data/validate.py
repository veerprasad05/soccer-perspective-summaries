"""Eligibility: two distinct names, numeric result, ordered timed event, no goal-count contradiction."""
import argparse
import collections
import json
from .common import read_jsonl, atomic_json, outcome

def reasons(r):
    why = []
    a, b = r.get("team_a"), r.get("team_b")
    if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip() or a.strip().casefold() == b.strip().casefold(): why.append("missing_or_invalid_teams")
    s = r.get("score") or {}
    if any(type(s.get(t)) is not int or s[t] < 0 for t in ("A", "B")): why.append("missing_or_invalid_score")
    events = r.get("events") or []
    timed = [e for e in events if isinstance(e.get("minute"), int) and e["minute"] >= 0 and e.get("type")]
    if not timed: why.append("no_timed_events")
    if len({e.get("event_id") for e in events}) != len(events): why.append("duplicate_event_ids")
    if any(e.get("team") not in ("A", "B", None) for e in events): why.append("invalid_event_team")
    warnings=set(r.get("parse_warnings") or [])
    if "commentary_final_score_conflict" in warnings or "multiple_final_scores_conflict" in warnings: why.append("score_contradiction")
    if "commentary_team_mismatch" in warnings: why.append("commentary_team_mismatch")
    if "statistics_team_mismatch" in warnings: why.append("statistics_team_mismatch")
    if "missing_or_invalid_score" not in why:
        for team in ("A", "B"):
            goals = sum(e.get("type") == "goal" and e.get("team") == team for e in events)
            if goals > s[team]: why.append("goal_count_exceeds_score"); break
    return why

def audit(rows):
    rows = list(rows); ids = collections.Counter(r.get("match_id") for r in rows)
    excluded = {}; missing = collections.Counter(); eligible = []
    for r in rows:
        why = reasons(r)
        if not r.get("match_id") or ids[r.get("match_id")] != 1: why.append("missing_or_duplicate_match_id")
        for key in ("competition", "date", "team_a", "team_b"):
            if r.get(key) is None: missing[key] += 1
        for team in ("A", "B"):
            if (r.get("score") or {}).get(team) is None: missing[f"score_{team}"] += 1
        if not (r.get("events") or []): missing["events"] += 1
        if why: excluded[str(r.get("match_id"))] = sorted(set(why))
        else: eligible.append(r)
    return eligible, {"raw_records": len(rows), "unique_match_ids": len(ids), "normalized": len(rows), "eligible": len(eligible),
                      "excluded": len(rows)-len(eligible), "exclusion_reasons": dict(collections.Counter(x for v in excluded.values() for x in v)),
                      "excluded_match_ids": excluded, "missing_fields": dict(missing), "outcomes": dict(collections.Counter(outcome(r) for r in eligible)),
                      "source_counts":dict(collections.Counter(r.get("source") for r in rows)),
                      "event_types":dict(collections.Counter(e.get("type") for r in eligible for e in (r.get("events") or []))),
                      "parse_warnings":dict(collections.Counter(w for r in rows for w in (r.get("parse_warnings") or []))),
                      "eligible_goal_event_complete_count":sum(bool(r.get("goal_events_complete")) for r in eligible),
                      "sample_eligible_ids": [r["match_id"] for r in eligible[:5]]}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", default="data/processed/audit.json")
    a=p.parse_args(); _, report = audit(read_jsonl(a.input)); atomic_json(a.output, report); print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
