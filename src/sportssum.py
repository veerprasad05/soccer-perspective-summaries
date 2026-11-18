"""SportsSum-style name/event F-scores against recorded BBC match facts.

Huang et al. (2020) compare generated text with reference news. Here the
reference is the structured event list supplied to Qwen, so these scores are
an adaptation and cannot be compared numerically with the paper's results.
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .data.common import atomic_jsonl, read_jsonl

CAP = r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’-]*"
PARTICLE = r"(?:de|del|da|dos|van|von|di|le|la)"
PERSON = rf"{CAP}(?:\s+(?:{CAP}|{PARTICLE})){{0,3}}"
GOAL_VERBS = r"scored|scoring|netted|equalized|equalised"
METRIC_VERSION = "sportssum-bbc-v2"


def fold(value):
    value = unicodedata.normalize("NFKD", value.casefold().replace("’", "'"))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip(" .,:;!?'\"")


def score_sets(reference, predicted):
    """Paper-style set precision, recall, F1; empty/empty is undefined."""
    reference, predicted = set(reference), set(predicted)
    tp = len(reference & predicted)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    if not reference and not predicted:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def reference_sets(record):
    names, events = set(), set()
    for item in record.get("events", []):
        player = item.get("player")
        if player:
            player = fold(player)
            names.add(player)
            kind = item.get("type")
            if kind == "goal":
                events.add((player, "own_goal" if item.get("goal_type") == "own_goal" else "goal"))
            elif kind in ("yellow_card", "red_card"):
                events.add((player, kind))
            elif kind == "substitution":
                events.add((player, "sub_on"))
        if item.get("type") == "substitution" and item.get("replaced_player"):
            outgoing = fold(item["replaced_player"])
            names.add(outgoing)
            events.add((outgoing, "sub_off"))
    return names, events


class NameResolver:
    def __init__(self, record, reference_names):
        self.teams = set()
        for team in (record["team_a"], record["team_b"]):
            parts = fold(team).split()
            self.teams.update((" ".join(parts), parts[0], parts[-1]))
        self.aliases = {}
        candidates = defaultdict(set)
        for name in reference_names:
            words = name.split()
            candidates[name].add(name)
            for count in (1, 2):
                if len(words) > count:
                    alias = " ".join(words[-count:])
                    if len(alias) >= 3:
                        candidates[alias].add(name)
        for alias, players in candidates.items():
            if len(players) == 1 and alias not in self.teams:
                self.aliases[alias] = next(iter(players))

    def resolve(self, candidate):
        name = fold(candidate)
        # A noun phrase may begin with the possessing team, e.g. "Roma's
        # Dzeko". The event subject is the player after that possessive.
        if "'s " in name:
            name = name.rsplit("'s ", 1)[-1]
        if name in self.teams or name.startswith("the ") or name in {"team a", "team b", "both", "they", "the", "we", "he", "she"}:
            return None
        if name in self.aliases:
            return self.aliases[name]
        for alias in sorted(self.aliases, key=len, reverse=True):
            if name.endswith(" " + alias):
                return self.aliases[alias]
        return f"unresolved:{name}" if len(name) >= 3 else None

    def known_mentions(self, text):
        folded = fold(text)
        found = set()
        for alias, canonical in self.aliases.items():
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", folded):
                found.add(canonical)
        return found


def extract_predictions(record, text, reference_names):
    """Extract English player names and a fixed synonym inventory of actions.

    Unknown names are detected only in explicit action phrases. This does not
    claim general named-entity or semantic-relation extraction.
    """
    resolver = NameResolver(record, reference_names)
    names = resolver.known_mentions(text)
    events = set()
    unresolved = set()

    def add(candidate, kind):
        person = resolver.resolve(candidate)
        if person is None:
            return
        names.add(person)
        events.add((person, kind))
        if person.startswith("unresolved:"):
            unresolved.add(person)

    direct = (
        (rf"\b(?P<person>{PERSON})\s+(?i:{GOAL_VERBS})\b", "goal"),
        (rf"\b(?P<person>{PERSON})\s+(?i:opened the scoring|added (?:a |another |two )?goals?|bagged (?:a |two )?goals?)\b", "goal"),
        (rf"\b(?P<person>{PERSON})['’]s\s+(?i:goal|brace|hat-trick)\b", "goal"),
        (rf"\b(?P<person>{PERSON})\s+(?i:was\s+)?(?i:booked)\b", "yellow_card"),
        (rf"\b(?P<person>{PERSON})\s+(?i:received|picked up|was shown)\s+(?:a\s+)?(?i:yellow card)\b", "yellow_card"),
        (rf"\b(?P<person>{PERSON})\s+(?i:was\s+)?(?i:sent off|shown red)\b", "red_card"),
        (rf"\b(?P<person>{PERSON})\s+(?i:received|picked up|was shown)\s+(?:a\s+)?(?i:red card)\b", "red_card"),
        (rf"\b(?P<person>{PERSON})\s+(?i:came on|was substituted on)\b", "sub_on"),
        (rf"\b(?P<person>{PERSON})\s+(?i:went off|was substituted off)\b", "sub_off"),
    )
    for pattern, kind in direct:
        for match in re.finditer(pattern, text):
            actual_kind = kind
            if kind == "goal" and re.match(r"\s+(?i:an?\s+)?(?i:own goal)\b", text[match.end():]):
                actual_kind = "own_goal"
            add(match.group("person"), actual_kind)

    # Explicit named event lists are common in short sports summaries.
    lists = (
        (r"\b(?i:goals?)\s+(?i:came\s+)?(?i:from|by|through|via)\s+([^.!?;]{1,400})", "goal"),
        (r"\b(?i:yellow cards?)\s+(?i:for|to)\s+([^.!?;]{1,100})", "yellow_card"),
        (r"\b(?i:red cards?)\s+(?i:for|to)\s+([^.!?;]{1,100})", "red_card"),
    )
    for pattern, kind in lists:
        for match in re.finditer(pattern, text):
            if kind == "goal" and re.search(r"\bown\s+$", text[max(0, match.start()-5):match.start()], re.I):
                continue
            # A scorer list often attaches minutes to each name: "Smith (29'),
            # Jones (90+2')". Remove only these time annotations before
            # applying the ordinary clause boundary; other parentheses still
            # stop the list to avoid crediting unrelated player mentions.
            listed = re.sub(r"\(\s*\d{1,3}(?:\+\d{1,2})?\s*['’]?\s*\)", " ", match.group(1))
            segment = re.split(
                r"\(|,\s*(?:we|they|he|she|the|while|but|and an? red|and an? own)\b|"
                r"\b(?:but|while|before|after|put|ensured|helped|gave|sealed|followed|were|was)\b",
                listed, maxsplit=1, flags=re.I,
            )[0]
            for person in resolver.known_mentions(segment):
                names.add(person)
                events.add((person, kind))
            first = re.match(rf"(?P<person>{PERSON})\b", segment)
            if first and not resolver.known_mentions(first.group("person")):
                add(first.group("person"), kind)

    for match in re.finditer(rf"\b(?i:own goal)\s+(?i:by|from)\s+(?P<person>{PERSON})\b", text):
        add(match.group("person"), "own_goal")
    for match in re.finditer(rf"\b(?i:yellow card|red card)\s+(?i:for|to)\s+(?P<person>{PERSON})\b", text):
        add(match.group("person"), "yellow_card" if "yellow" in match.group(0).casefold() else "red_card")

    # Preserve who came on and who went off when the relation is explicit.
    for match in re.finditer(rf"\b(?P<incoming>{PERSON})\s+(?i:replaced)\s+(?P<outgoing>{PERSON})\b", text):
        add(match.group("incoming"), "sub_on")
        add(match.group("outgoing"), "sub_off")
    for match in re.finditer(rf"\b(?P<outgoing>{PERSON})\s+(?i:was replaced by)\s+(?P<incoming>{PERSON})\b", text):
        add(match.group("incoming"), "sub_on")
        add(match.group("outgoing"), "sub_off")
    return names, events, unresolved


def evaluate(record, summary):
    ref_names, ref_events = reference_sets(record)
    pred_names, pred_events, unresolved = extract_predictions(record, summary, ref_names)
    name_score = score_sets(ref_names, pred_names)
    event_score = score_sets(ref_events, pred_events)
    salient_kinds = {"goal", "own_goal", "red_card"}
    salient_ref = {event for event in ref_events if event[1] in salient_kinds}
    salient_pred = {event for event in pred_events if event[1] in salient_kinds}
    return {
        "metric_version": METRIC_VERSION, "reference_kind": "bbc_recorded_events",
        "goal_events_complete": bool(record.get("goal_events_complete")),
        "reference_names": sorted(ref_names), "predicted_names": sorted(pred_names),
        "reference_events": [list(x) for x in sorted(ref_events)],
        "predicted_events": [list(x) for x in sorted(pred_events)],
        "reference_salient_events": [list(x) for x in sorted(salient_ref)],
        "predicted_salient_events": [list(x) for x in sorted(salient_pred)],
        "unresolved_action_names": sorted(unresolved),
        "name": name_score, "event": event_score,
        "salient_event": score_sets(salient_ref, salient_pred),
    }


def aggregate(rows):
    result = []
    for condition in sorted({row["condition"] for row in rows}):
        for subset in ("all_recorded", "complete_goal_logs"):
            group = [r for r in rows if r["condition"] == condition and
                     (subset == "all_recorded" or r["goal_events_complete"])]
            output = {"metric_version": METRIC_VERSION, "reference_kind": "bbc_recorded_events",
                      "condition": condition, "subset": subset,
                      "n_summaries": len(group), "n_matches": len({r["match_id"] for r in group}),
                      "n_with_unresolved_action_names": sum(bool(r["unresolved_action_names"]) for r in group)}
            for name in ("name", "event", "salient_event"):
                tp = sum(r[name]["tp"] for r in group)
                fp = sum(r[name]["fp"] for r in group)
                fn = sum(r[name]["fn"] for r in group)
                micro = score_counts(tp, fp, fn)
                macro_values = [r[name]["f1"] for r in group if r[name]["f1"] is not None]
                output.update({f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn,
                               f"{name}_precision": micro["precision"], f"{name}_recall": micro["recall"],
                               f"{name}_f1": micro["f1"],
                               f"{name}_macro_f1": sum(macro_values)/len(macro_values) if macro_values else None})
            result.append(output)
    return result


def score_counts(tp, fp, fn):
    if tp + fp + fn == 0:
        return {"precision": None, "recall": None, "f1": None}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def format_report(table):
    """Show each metric's precision, recall, and harmonic-mean F1 explicitly."""
    lines = ["condition  subset               matches  metric          precision  recall  F1"]
    for row in table:
        for key, label in (("name", "NMS"), ("event", "EMS"),
                           ("salient_event", "salient EMS")):
            values = [row[f"{key}_{field}"] for field in ("precision", "recall", "f1")]
            shown = ["NA" if value is None else f"{value:.3f}" for value in values]
            lines.append(f"{row['condition']:<10} {row['subset']:<20} {row['n_matches']:>7}  "
                         f"{label:<14} {shown[0]:>9}  {shown[1]:>6}  {shown[2]:>5}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/processed/matches.jsonl")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--split-ids", default="data/splits/dev.json")
    parser.add_argument("--output", default="results/sportssum_dev.csv")
    parser.add_argument("--details", default="results/sportssum_dev_details.jsonl")
    args = parser.parse_args()
    ids = set(json.loads(Path(args.split_ids).read_text(encoding="utf-8")))
    records = {r["match_id"]: r for r in read_jsonl(args.matches) if r["match_id"] in ids}
    if set(records) != ids:
        raise ValueError("Split IDs are missing from source records")
    details = []
    seen = set()
    for row in read_jsonl(args.generations):
        if row["match_id"] not in ids or row.get("status") != "complete":
            raise ValueError("Generation file contains a non-split or incomplete row")
        evaluation_condition = row.get("evaluation_condition", row["condition"])
        key = (row["match_id"], evaluation_condition)
        if key in seen:
            raise ValueError(f"Duplicate generation for {key}")
        seen.add(key)
        details.append({"run_id": row["run_id"], "match_id": row["match_id"],
                        "condition": evaluation_condition,
                        "split_manifest_sha256": row.get("split_manifest_sha256"),
                        **evaluate(records[row["match_id"]], row["output_text"])})
    if not details:
        raise ValueError("No complete generations")
    conditions = {row["condition"] for row in details}
    missing = {(match_id, condition) for match_id in ids for condition in conditions} - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} match/condition generations; first: {min(missing)}")
    split_hashes = {row["split_manifest_sha256"] for row in details}
    if len(split_hashes) != 1:
        raise ValueError("Generation rows have inconsistent split manifest hashes")
    atomic_jsonl(args.details, details)
    table = aggregate(details)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {len(details)} SportsSum-style evaluations to {args.details} and {args.output}")
    print(format_report(table))


if __name__ == "__main__":
    main()
