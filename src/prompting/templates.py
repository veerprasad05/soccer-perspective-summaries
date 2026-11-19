from .render import render_match_facts
CONDITIONS = ("neutral", "fan_A", "fan_B")
AFFINITIES = ("casual", "passionate")
VERSION = "baseline-v1"
COVERAGE_MODES = ("baseline", "concise", "complete", "concise_open")
VERSIONS = {"baseline": VERSION, "concise": "record-concise-v1",
            "complete": "record-complete-v1",
            "concise_open": "record-concise-open-v1"}


def build_messages(record, condition, affinity="casual", coverage_mode="baseline"):
    if condition not in CONDITIONS: raise ValueError(f"Invalid condition {condition}")
    if affinity not in AFFINITIES: raise ValueError(f"Invalid affinity {affinity}")
    if coverage_mode not in COVERAGE_MODES: raise ValueError(f"Invalid coverage mode {coverage_mode}")
    if coverage_mode == "concise_open" and condition != "neutral":
        raise ValueError("concise_open is reserved for neutral generation")
    audience = {"neutral":"Write a balanced summary for a general reader.",
                "fan_A":f"Write for supporters of {record['team_a']} (Team A).",
                "fan_B":f"Write for supporters of {record['team_b']} (Team B)."}[condition]
    energy = "Use restrained language." if affinity == "casual" else "Use lively language while keeping every claim factual."
    if coverage_mode == "baseline":
        system = "You write short soccer match summaries from supplied match facts only. Do not invent scores, events, scorers, minutes, or statistics. Use 80 to 110 words. If a detail is unknown, omit it."
    else:
        system = ("You write soccer match summaries from the supplied BBC match record only. "
                  "Treat that record as the final truth for this task. Never invent, infer, or contradict "
                  "any match fact, including scores, events, players, minutes, and statistics. "
                  "If a detail is absent or unknown in the record, omit it. Aim to use no more than 150 words.")
        if coverage_mode == "complete":
            system += (" Include every available fact shown in the verified match facts: both teams, "
                       "competition and date when present, final score, every listed event with its "
                       "recorded minute, type, team, player, replacement, own-goal status, and "
                       "score-after when present, and every listed statistic for both teams. "
                       "Use compact clauses if needed; full coverage takes priority over the word target.")
    user_prefix = ("Write a summary for the match." if coverage_mode == "concise_open"
                   else audience + " " + energy)
    user = user_prefix + "\n\nVerified match facts:\n" + render_match_facts(record)
    return [{"role":"system","content":system},{"role":"user","content":user}]
