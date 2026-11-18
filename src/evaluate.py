"""Conservative rule checker; ambiguous claims are unverifiable, not supported."""
import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path
from .data.common import read_jsonl, atomic_jsonl

def evaluate(record, text):
    a,b=record["team_a"],record["team_b"]
    s=record.get("score") or {}; claims=[]
    def add(kind,value,status): claims.append({"kind":kind,"value":value,"status":status})
    # Require both teams and their score order. A bare "Team 2-1" may be a
    # score for the preceding opponent, and "Team 14-12" may describe shots.
    for first,second,side in ((a,b,"A"),(b,a,"B")):
        first,second=re.escape(first),re.escape(second)
        patterns=(
            rf"\b{first}\s+(\d+)\s*[-–:]\s*(\d+)\s+{second}\b",
            rf"\b{first}\s+(?:beat|defeated)\s+{second}\s+(\d+)\s*[-–:]\s*(\d+)\b",
        )
        for pattern in patterns:
            for m in re.finditer(pattern,text,re.I):
                x,y=map(int,m.groups())
                expected=(s.get(side),s.get("B" if side=="A" else "A"))
                status="unverifiable" if None in expected else "supported" if (x,y)==expected else "contradicted"
                add("score",m.group(0),status)
    for winner, loser, side in ((a,b,"A"),(b,a,"B")):
        for m in re.finditer(rf"\b{re.escape(winner)}\s+(?:beat|defeated)\s+{re.escape(loser)}\b",text,re.I):
            status="unverifiable" if s.get("A") is None or s.get("B") is None else "supported" if (s[side]>s["B" if side=="A" else "A"]) else "contradicted"
            add("winner",m.group(0),status)
    lower=text.casefold(); events=[e for e in record.get("events",[]) if e.get("type")=="goal" and e.get("player")]
    def normalize_name(value):
        return "".join(c for c in unicodedata.normalize("NFKD",value.casefold()) if not unicodedata.combining(c))
    players={normalize_name(e["player"]) for e in events}
    def resolve_player(candidate):
        name=normalize_name(candidate)
        if name in players:return name
        surname_matches={p for p in players if p.split()[-1]==name}
        return next(iter(surname_matches)) if len(surname_matches)==1 else None
    team_names={normalize_name(team) for team in (a,b)}
    team_names.update(normalize_name(team).split()[-1] for team in (a,b))
    for match in re.finditer(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?)\s+(?:scored|netted)\b",text):
        player=match.group(1)
        if normalize_name(player) in team_names:continue
        resolved=resolve_player(player)
        known=[e for e in events if normalize_name(e["player"])==resolved]
        # The event log can be partial; an unlisted scorer is unknown unless completeness is certified.
        status="supported" if known else "contradicted" if record.get("goal_events_complete") else "unverifiable"
        add("scorer",player,status)
        minute_match=re.match(r"\s+(?:in\s+)?(?:the\s+)?(\d{1,3})(?:\+(\d{1,2}))?(?:st|nd|rd|th)?(?:\s+minute)?",text[match.end():],re.I)
        if minute_match:
            minute=int(minute_match.group(1)); stoppage=int(minute_match.group(2)) if minute_match.group(2) else None
            status="unverifiable" if not known else "supported" if any(e.get("minute")==minute and e.get("stoppage_minute")==stoppage for e in known) else "contradicted"
            add("goal_minute",minute_match.group(0).strip(),status)
    # This is scorer-name mention coverage, not full event recall: one name
    # mention may represent several goals, and shortened names may be missed.
    salient=[e for e in record.get("events",[]) if e.get("type")=="goal" and e.get("player")]
    covered=sum(e["player"].casefold() in lower for e in salient)
    verifiable=[c for c in claims if c["status"]!="unverifiable"]
    return {"claims":claims,"supported":sum(c["status"]=="supported" for c in claims),"contradicted":sum(c["status"]=="contradicted" for c in claims),
            "unverifiable":sum(c["status"]=="unverifiable" for c in claims),"any_factual_error":any(c["status"]=="contradicted" for c in claims),
            "fact_precision":sum(c["status"]=="supported" for c in verifiable)/len(verifiable) if verifiable else None,
            "event_coverage":covered/len(salient) if salient else None,"event_denominator":len(salient)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--matches",default="data/processed/matches.jsonl");p.add_argument("--generations",required=True);p.add_argument("--output",default="results/baselines_summary.csv");p.add_argument("--details",default="results/evaluation.jsonl")
    a=p.parse_args();records={r["match_id"]:r for r in read_jsonl(a.matches)};details=[]
    for g in read_jsonl(a.generations):
        if g.get("status")!="complete":continue
        if g["match_id"] not in records:raise ValueError(f"Unknown match {g['match_id']}")
        details.append({"run_id":g["run_id"],"match_id":g["match_id"],"condition":g.get("evaluation_condition",g["condition"]),**evaluate(records[g["match_id"]],g["output_text"])})
    atomic_jsonl(a.details,details); groups=collections.defaultdict(list)
    for d in details:groups[d["condition"]].append(d)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["condition","n_summaries","n_matches","n_verifiable_claims","n_summaries_with_verifiable_claims","n_unverifiable_claims","n_matches_with_recorded_goals","fact_precision","event_coverage","any_factual_error_rate","perspective"]);w.writeheader()
        for condition,ds in sorted(groups.items()):
            supported=sum(d["supported"] for d in ds);verified=supported+sum(d["contradicted"] for d in ds)
            cover=[d["event_coverage"] for d in ds if d["event_coverage"] is not None]
            w.writerow({"condition":condition,"n_summaries":len(ds),"n_matches":len({d["match_id"] for d in ds}),
                        "n_verifiable_claims":verified,"n_summaries_with_verifiable_claims":sum(d["supported"]+d["contradicted"]>0 for d in ds),
                        "n_unverifiable_claims":sum(d["unverifiable"] for d in ds),"n_matches_with_recorded_goals":len(cover),
                        "fact_precision":supported/verified if verified else "NA","event_coverage":sum(cover)/len(cover) if cover else "NA",
                        "any_factual_error_rate":sum(d["any_factual_error"] for d in ds)/len(ds),"perspective":"pending"})
    print(f"Wrote {len(details)} evaluations to {a.details} and {a.output}")
if __name__=="__main__":main()
