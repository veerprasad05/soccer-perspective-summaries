"""Adapter for the inspected 2026-11-15 SoccerNet-BBC ZIP (four BBC text files per match).

No article prose is parsed. Folder score is compared with commentary's explicit final score.
Unresolved conflicts remain in records and fail eligibility.
"""
import json
import re
import unicodedata
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from .common import PARSER_VERSION, sha256_file

FOLDER_RE = re.compile(r"^(?P<id>\d+)_(?P<competition>.*?) (?P<season>\d{4}-\d{4}) (?P<date>\d{4}-\d{2}-\d{2}) - (?P<clock>\d{2}-\d{2}) (?P<home>.*?) (?P<a>\d+) - (?P<b>\d+) (?P<away>.*?) BBC\.com$")
TIME_RE = re.compile(r"^(\d{1,3})'(?:\+(\d{1,2})')?$")
FINAL_RE = re.compile(r"^Match ends, (.+?) (\d+), (.+?) (\d+)\.$")
GOAL_RE = re.compile(r"^Goal! (.+?) (\d+), (.+?) (\d+)\. (.+)$")
OWN_RE = re.compile(r"^Own Goal by (.+?), (.+?)\. (.+?) (\d+), (.+?) (\d+)\.$")

def name_key(s):
    s="".join(c for c in unicodedata.normalize("NFKD",s.casefold()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)

def align_pair(names, home, away):
    """Return source order relative to folder A/B, or None for an unsafe join.

    Conservative string matching avoids treating a different fixture as the same
    match merely because two score tuples happen to be equal.
    """
    def sim(x,y): return SequenceMatcher(None,name_key(x),name_key(y)).ratio()
    direct=(sim(names[0],home),sim(names[1],away))
    reverse=(sim(names[0],away),sim(names[1],home))
    dscore=sum(direct);rscore=sum(reverse)
    best=direct if dscore>=rscore else reverse
    if min(best)<0.4 or sum(best)<0.9 or abs(dscore-rscore)<0.15:return None
    return "direct" if dscore>=rscore else "reverse"

def read_text(z, path):
    return z.read(path).decode("utf-8-sig", errors="replace")

def parse_stats(text):
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if len(lines)<4 or lines[:1]!=["Home Team"] or lines[2]!="Away Team":
        raise ValueError("Unexpected match_stats header")
    result={}; i=4
    while i+4<len(lines):
        label=lines[i]; section=lines[i+1:i+5]
        if section[0]!="Home" or section[2]!="Away": raise ValueError(f"Unexpected statistics block: {label}")
        def number(x):
            x=x.rstrip("%")
            return int(x) if x.isdigit() else None
        result[label.lower().replace(" ","_")]={"A":number(section[1]),"B":number(section[3])}
        i+=5
    if i!=len(lines): raise ValueError("Trailing statistics lines")
    return lines[1],lines[3],result

def commentary_groups(text):
    """Yield timed BBC commentary groups with 1-based source line numbers."""
    lines=text.splitlines(); current=None
    for line_no,line in enumerate(lines,1):
        item=line.strip()
        m=TIME_RE.fullmatch(item)
        if m:
            if current: yield current
            current={"minute":int(m.group(1)),"stoppage_minute":int(m.group(2)) if m.group(2) else None,"line":line_no,"lines":[]}
        elif current and item:
            current["lines"].append((line_no,item))
    if current: yield current

def team_side(raw, home_names, away_names):
    k=name_key(raw)
    home={name_key(x) for x in home_names};away={name_key(x) for x in away_names}
    if k in home and k not in away:return "A"
    if k in away and k not in home:return "B"
    return None

def parse_match(z, folder, archive_hash):
    base=PurePosixPath(folder).name
    m=FOLDER_RE.fullmatch(base)
    if not m: raise ValueError(f"Unrecognized match folder: {folder}")
    g=m.groupdict(); paths={kind:f"{folder}/{kind}.txt" for kind in ("info","match_stats","commentary","lineup")}
    stat_home,stat_away,stats=parse_stats(read_text(z,paths["match_stats"]))
    info=json.loads(read_text(z,paths["info"]))
    lines=read_text(z,paths["commentary"]).splitlines()
    final=[]
    for line_no,line in enumerate(lines,1):
        match=FINAL_RE.fullmatch(line.strip())
        if match:final.append({"line":line_no,"teams":[match.group(1),match.group(3)],"scores":[int(match.group(2)),int(match.group(4))]})
    a,b=int(g["a"]),int(g["b"])
    warnings=[]
    stat_alignment=align_pair((stat_home,stat_away),g["home"],g["away"])
    if stat_alignment is None:
        warnings.append("statistics_team_mismatch")
        stats={}
    elif stat_alignment=="reverse":
        stats={key:{"A":value["B"],"B":value["A"]} for key,value in stats.items()}
    home_names=[g["home"]];away_names=[g["away"]]
    if stat_alignment=="direct":home_names.append(stat_home);away_names.append(stat_away)
    elif stat_alignment=="reverse":home_names.append(stat_away);away_names.append(stat_home)
    final_alignment=None
    if not final:warnings.append("commentary_final_score_missing")
    else:
        final_alignment=align_pair(final[0]["teams"],g["home"],g["away"])
        if final_alignment is None:warnings.append("commentary_team_mismatch")
        elif final[0]["scores"] != ([a,b] if final_alignment=="direct" else [b,a]):warnings.append("commentary_final_score_conflict")
        if len({tuple(x["scores"]) for x in final})>1:warnings.append("multiple_final_scores_conflict")
    # Add final names only after both team alignment and score agreement.
    if final:
        f=final[0]
        if not any(w in warnings for w in ("commentary_team_mismatch","commentary_final_score_conflict")):
            if final_alignment=="direct": home_names.append(f["teams"][0]);away_names.append(f["teams"][1])
            elif final_alignment=="reverse":home_names.append(f["teams"][1]);away_names.append(f["teams"][0])
    events=[]
    for group in commentary_groups("\n".join(lines)):
        contents=group["lines"]
        if not contents:continue
        heading=contents[0][1].casefold()
        details=[(n,t) for n,t in contents if t.startswith("Goal! ") or t.startswith("Own Goal by")]
        kind=None;player=None;side=None;goal_type=None;score_after=None;source_line=group["line"]
        if details:
            source_line,detail=details[0];kind="goal"
            own=OWN_RE.fullmatch(detail)
            if own:
                player=own.group(1); credited_against=own.group(2)
                against=team_side(credited_against,home_names,away_names)
                side={"A":"B","B":"A"}.get(against)
                goal_type="own_goal";score_after=[int(own.group(4)),int(own.group(6))]
            else:
                gm=GOAL_RE.fullmatch(detail)
                if gm:
                    score_after=[int(gm.group(2)),int(gm.group(4))]
                    p=re.match(r"^(.+?) \((.+?)\) ",gm.group(5))
                    if p:player=p.group(1);side=team_side(p.group(2),home_names,away_names)
                    goal_type="penalty" if "converts the penalty" in detail.casefold() else "regular"
                else:warnings.append("unparsed_goal_detail")
        elif heading=="booking":
            kind="yellow_card"
            detail=contents[1][1] if len(contents)>1 else ""
            card=re.match(r"^(.+?) \((.+?)\) is shown the yellow card",detail)
            if card:player=card.group(1);side=team_side(card.group(2),home_names,away_names)
        elif heading=="dismissal":
            kind="red_card"
            detail=contents[1][1] if len(contents)>1 else ""
            card=re.match(r"^(.+?) \((.+?)\) is shown the red card",detail) or re.match(r"^Second yellow card to (.+?) \((.+?)\)",detail)
            if card:player=card.group(1);side=team_side(card.group(2),home_names,away_names)
        elif heading=="substitution":
            kind="substitution"
            detail=contents[1][1] if len(contents)>1 else ""
            sub=re.match(r"^Substitution, (.+?)\. (.+?) replaces (.+?)\.$",detail)
            if sub:side=team_side(sub.group(1),home_names,away_names);player=sub.group(2)
        if kind:
            event={"event_id":f"soccernet-bbc:{g['id']}:line:{source_line}","minute":group["minute"],"stoppage_minute":group["stoppage_minute"],
                   "type":kind,"team":side,"player":player,"source_reference":f"{paths['commentary']}:{source_line}"}
            if goal_type:event["goal_type"]=goal_type
            if kind=="substitution" and sub:event["replaced_player"]=sub.group(3)
            if score_after:
                event["source_score_after_ordered"] = score_after
                if final_alignment=="direct":event["score_after"]={"A":score_after[0],"B":score_after[1]}
                elif final_alignment=="reverse":event["score_after"]={"A":score_after[1],"B":score_after[0]}
            events.append(event)
    # BBC commentary sometimes repeats an earlier passage. Preserve all references
    # for repeated events while keeping each event once.
    unique=[]; seen={}
    for event in events:
        if event["type"]=="goal":
            key=(event["type"],event["minute"],event["stoppage_minute"],event["player"],tuple(event.get("source_score_after_ordered") or []))
        else:
            key=(event["type"],event["minute"],event["stoppage_minute"],event["team"],event["player"],event.get("replaced_player"))
        if key in seen:
            seen[key]["source_references"].append(event["source_reference"])
            warnings.append(f"duplicate_{event['type']}_commentary_removed")
            continue
        event["source_references"]=[event["source_reference"]]
        seen[key]=event
        unique.append(event)
    events=unique
    events.sort(key=lambda e:(e["minute"],e["stoppage_minute"] or 0,e["event_id"]))
    goal_events_complete=all(sum(e["type"]=="goal" and e["team"]==side for e in events)==target for side,target in (("A",a),("B",b)))
    if not goal_events_complete:warnings.append("goal_events_incomplete_or_conflicting")
    record={"match_id":f"soccernet-bbc:{int(g['id']):03d}","source":"soccernet-bbc","competition":g["competition"],"season":g["season"],"date":g["date"],
            "team_a":g["home"],"team_b":g["away"],"team_aliases":{"A":sorted(set(home_names)),"B":sorted(set(away_names))},"home_away_mapping":{"A": "home" if stat_alignment=="direct" else "away" if stat_alignment=="reverse" else None,"B":"away" if stat_alignment=="direct" else "home" if stat_alignment=="reverse" else None},
            "score":{"A":a,"B":b},"score_source_reference":f"{folder}/", "commentary_final":final[0] if final else None,
            "events":events,"goal_events_complete":goal_events_complete,"statistics":stats,"statistics_source_reference":paths["match_stats"] if stats else None,"source_files":list(paths.values()),"bbc_url":info.get("url"),"source_alignments":{"statistics":stat_alignment,"commentary":final_alignment},
            "parse_warnings":sorted(set(warnings)),"parser_version":PARSER_VERSION,"source_archive_sha256":archive_hash}
    return record

def load_archive(path):
    archive_hash=sha256_file(path)
    with zipfile.ZipFile(path) as z:
        folders=sorted({str(PurePosixPath(n).parent) for n in z.namelist() if n.endswith("match_stats.txt")})
        return [parse_match(z,folder,archive_hash) for folder in folders]
