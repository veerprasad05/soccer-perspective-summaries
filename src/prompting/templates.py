from .render import render_match_facts
CONDITIONS = ("neutral", "fan_A", "fan_B")
AFFINITIES = ("casual", "passionate")
VERSION = "baseline-v1"
def build_messages(record, condition, affinity="casual"):
    if condition not in CONDITIONS: raise ValueError(f"Invalid condition {condition}")
    if affinity not in AFFINITIES: raise ValueError(f"Invalid affinity {affinity}")
    audience = {"neutral":"Write a balanced summary for a general reader.",
                "fan_A":f"Write for supporters of {record['team_a']} (Team A).",
                "fan_B":f"Write for supporters of {record['team_b']} (Team B)."}[condition]
    energy = "Use restrained language." if affinity == "casual" else "Use lively language while keeping every claim factual."
    system = "You write short soccer match summaries from supplied match facts only. Do not invent scores, events, scorers, minutes, or statistics. Use 80 to 110 words. If a detail is unknown, omit it."
    user = audience + " " + energy + "\n\nVerified match facts:\n" + render_match_facts(record)
    return [{"role":"system","content":system},{"role":"user","content":user}]
