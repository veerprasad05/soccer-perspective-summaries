"""Conservative rule checker; ambiguous claims are unverifiable, not supported."""
import argparse
import collections
import csv
import json
import re
from pathlib import Path
from .data.common import read_jsonl, atomic_jsonl

def evaluate(record, text):
    a,b=record["team_a"],record["team_b"]
    s=record.get("score") or {}; claims=[]
    def add(kind,value,status): claims.append({"kind":kind,"value":value,"status":status})
    # Explicit 'A 2-1 B' or 'A 2, B 1' only. Scores without names are unverifiable.
    for m in re.finditer(rf"\b{re.escape(a)}\s+(\d+)\s*[-–,:]\s*(?:{re.escape(b)}\s+)?(\d+)\b",text,re.I):
        x,y=map(int,m.groups()); status="unverifiable" if s.get("A") is None or s.get("B") is None else "supported" if (x,y)==(s["A"],s["B"]) else "contradicted"
        add("score",m.group(0),status)
    for m in re.finditer(rf"\b{re.escape(b)}\s+(\d+)\s*[-–,:]\s*(?:{re.escape(a)}\s+)?(\d+)\b",text,re.I):
        x,y=map(int,m.groups());status="unverifiable" if s.get("A") is None or s.get("B") is None else "supported" if (y,x)==(s["A"],s["B"]) else "contradicted"
        add("score",m.group(0),status)
    for winner, loser, side in ((a,b,"A"),(b,a,"B")):
        for m in re.finditer(rf"\b{re.escape(winner)}\s+(?:beat|defeated)\s+{re.escape(loser)}\b",text,re.I):
            status="unverifiable" if s.get("A") is None or s.get("B") is None else "supported" if (s[side]>s["B" if side=="A" else "A"]) else "contradicted"
            add("winner",m.group(0),status)
    lower=text.casefold(); events=[e for e in record.get("events",[]) if e.get("type")=="goal" and e.get("player")]
    for match in re.finditer(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?)\s+(?:scored|netted)\b",text):
        player=match.group(1)
        known=[e for e in events if e["player"].casefold()==player.casefold()]
        # The event log can be partial; an unlisted scorer is unknown unless completeness is certified.
        status="supported" if known else "contradicted" if record.get("goal_events_complete") else "unverifiable"
        add("scorer",player,status)
        minute_match=re.match(r"\s+(?:in\s+)?(?:the\s+)?(\d{1,3})(?:\+(\d{1,2}))?(?:st|nd|rd|th)?(?:\s+minute)?",text[match.end():],re.I)
        if minute_match:
            minute=int(minute_match.group(1)); stoppage=int(minute_match.group(2)) if minute_match.group(2) else None
            status="unverifiable" if not known else "supported" if any(e.get("minute")==minute and e.get("stoppage_minute")==stoppage for e in known) else "contradicted"
            add("goal_minute",minute_match.group(0).strip(),status)
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
        details.append({"run_id":g["run_id"],"match_id":g["match_id"],"condition":g["condition"],**evaluate(records[g["match_id"]],g["output_text"])})
    atomic_jsonl(a.details,details); groups=collections.defaultdict(list)
    for d in details:groups[d["condition"]].append(d)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["condition","n_summaries","n_matches","fact_precision","event_coverage","any_factual_error_rate","perspective"]);w.writeheader()
        for condition,ds in sorted(groups.items()):
            supported=sum(d["supported"] for d in ds);verified=supported+sum(d["contradicted"] for d in ds)
            cover=[d["event_coverage"] for d in ds if d["event_coverage"] is not None]
            w.writerow({"condition":condition,"n_summaries":len(ds),"n_matches":len({d["match_id"] for d in ds}),"fact_precision":supported/verified if verified else "NA","event_coverage":sum(cover)/len(cover) if cover else "NA","any_factual_error_rate":sum(d["any_factual_error"] for d in ds)/len(ds),"perspective":"pending"})
    print(f"Wrote {len(details)} evaluations to {a.details} and {a.output}")
if __name__=="__main__":main()
