# SportsSum-style evaluation on SoccerNet-BBC facts

This is an **adaptation**, not a reproduction of Huang, Li, and Chang's [SportsSum evaluation](https://aclanthology.org/2020.aacl-main.61.pdf). Their name matching score (NMS) is the set F-score of player names in reference news versus generated news; their event matching score (EMS) is the set F-score of `(subject, verb)` pairs, with selected synonymous verbs treated as equivalent. Our archive does not contain the reference article text as a local field. We compare generated summaries with the **structured BBC commentary events supplied to Qwen**. Scores cannot be compared numerically with the SportsSum paper.

## Exact operationalization (`sportssum-bbc-v1`)

- **NMS reference:** distinct named players in recorded goals, yellow/red cards, substitutions, and replaced-player fields. **NMS prediction:** those players found in a summary using normalized full names or unique surname suffixes, plus previously unseen capitalized names in explicit action phrases. Accent variants are folded. Duplicate mentions count once.
- **EMS reference:** distinct `(player, event type)` pairs from the same records. Event types are `goal`, `own_goal`, `yellow_card`, `red_card`, `sub_on`, and `sub_off`. **EMS prediction:** player–action pairs recognized by a fixed English pattern inventory. `scored`, `scoring`, `netted`, `equalized`, and `equalised` share the goal class; `booked` and `yellow card` share the yellow-card class; `sent off` and `red card` share the red-card class. Explicit `replaced` and `came on/went off` patterns produce substitution pairs. Named goal/card lists are also parsed with clause boundaries. This is an English synonym adaptation of SportsSum's Chinese verb clusters.
- **Scores:** per-match precision `|predicted ∩ reference| / |predicted|`, recall `|predicted ∩ reference| / |reference|`, and F1 `2PR/(P+R)`. Duplicate events by the same player/type collapse, following the paper's set formulation. Empty/empty is undefined and excluded from macro F1. The CSV and CLI report micro **precision, recall, and F1 separately** for NMS, EMS, and salient EMS, plus macro per-match F1 by condition. Each F1 combines that metric's own precision and recall; there is no pooled NMS-plus-EMS F1.
- **Subsets:** `all_recorded` uses all 50 development matches. `complete_goal_logs` uses the 33 matches whose recorded goal count matches the final score. This restriction reduces penalties for goals missing from the source log; it does not certify every card or substitution. `salient_event` additionally reports EMS restricted to goals, own goals, and red cards, a prespecified subset more relevant to a short match summary.

## Current development results

Micro scores from `results/sportssum_dev.csv`, **complete-goal-log subset (33 matches per condition)**:

| Condition | Metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Neutral | NMS | 1.000 | 0.286 | 0.445 |
| Neutral | All-event EMS | 0.966 | 0.103 | 0.186 |
| Neutral | Goal/red-card EMS | 0.963 | 0.650 | 0.776 |
| Fan A | NMS | 1.000 | 0.203 | 0.337 |
| Fan A | All-event EMS | 1.000 | 0.106 | 0.192 |
| Fan A | Goal/red-card EMS | 1.000 | 0.650 | 0.788 |
| Fan B | NMS | 1.000 | 0.221 | 0.362 |
| Fan B | All-event EMS | 0.982 | 0.101 | 0.183 |
| Fan B | Goal/red-card EMS | 0.981 | 0.662 | 0.791 |

The all-event EMS recall is low because the reference contains routine cards and substitutions that short summaries rarely mention. NMS recall has the same content-selection issue. Source-linked name precision is optimistic because unknown names are detected only in explicit action patterns; some hallucinated names may be missed. The event parser also misses paraphrases and may misattribute an event in complex sentences. These automated scores are **matching/coverage diagnostics, not comprehensive factuality or a SportsSum benchmark result**. SportsSum NMS/EMS do not check final score, minute, team-side attribution, or invented statistics; the separate rule checker remains an auxiliary diagnostic for those claims. No manual ratings or checks are part of this evaluation run.

## Reproduce

```bash
python3 -m src.sportssum --matches data/processed/matches.jsonl --generations results/dev_baselines.jsonl --split-ids data/splits/dev.json --output results/sportssum_dev.csv --details results/sportssum_dev_details.jsonl
python3 -m src.sportssum --matches data/processed/matches.jsonl --generations results/train_pairs.jsonl --split-ids data/splits/train.json --output results/sportssum_train.csv --details results/sportssum_train_details.jsonl
```

The JSONL files expose both sets and TP/FP/FN for every summary. The 49-match test fold has not been scored.
