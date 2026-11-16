"""Deterministic factual input, with instruction delimiter escaping."""
from ..data.common import canonical_json, sha256_bytes

def safe(value):
    return str(value).replace("<", "&lt;").replace(">", "&gt;").replace("[", "&#91;").replace("]", "&#93;")

def render_match_facts(r):
    lines = [f"Match: {safe(r['team_a'])} (Team A) vs {safe(r['team_b'])} (Team B)"]
    if r.get("competition"): lines.append(f"Competition: {safe(r['competition'])}")
    if r.get("date"): lines.append(f"Date: {safe(r['date'])}")
    score = r.get("score") or {}
    if score.get("A") is not None and score.get("B") is not None:
        lines.append(f"Final score: {safe(r['team_a'])} {score['A']}, {safe(r['team_b'])} {score['B']}")
    else: lines.append("Final score: unknown")
    lines.append("Recorded events (the source event list may be incomplete; do not infer unlisted events):")
    events = sorted(r.get("events") or [], key=lambda e:(e.get("minute") is None,e.get("minute") if e.get("minute") is not None else 9999,e.get("stoppage_minute") or 0,e.get("event_id") or ""))
    if not events: lines.append("- No event information recorded")
    for e in events:
        m=e.get("minute"); time = "unknown time" if m is None else f"{m}{'+'+str(e['stoppage_minute']) if e.get('stoppage_minute') else ''}'"
        team = r.get("team_a") if e.get("team")=="A" else r.get("team_b") if e.get("team")=="B" else None
        label=(e.get("type") or "event").replace("_"," ")
        detail=f"- {time}: {safe(label)}" + (f"; team {safe(team)}" if team else "") + (f"; player {safe(e['player'])}" if e.get("player") else "")
        if e.get("goal_type")=="own_goal":detail+="; own goal by listed player, credited to listed team"
        if e.get("replaced_player"):detail+=f"; replaced {safe(e['replaced_player'])}"
        if e.get("score_after"):
            score_after=e["score_after"]
            detail+=f"; score after event {score_after['A']}-{score_after['B']} (Team A-Team B)"
        lines.append(detail)
    if r.get("statistics"): lines.append("Statistics: " + canonical_json(r["statistics"]))
    return "\n".join(lines)

def input_hash(r): return sha256_bytes(render_match_facts(r).encode())
