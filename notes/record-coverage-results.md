# Record-coverage pilot and development results — November 18, 2026

The synced Qwen outputs contain both train-only pilots (6 summaries per mode, 2 matches × neutral/A/B) and both full development runs (150 summaries per mode, 50 matches × neutral/A/B). Every generation has `status=complete`, the expected pinned Qwen revision and split-manifest hash, and the same factual-input hash across modes for a match/condition. The 49-match test fold remains untouched.

| Mode | Dev summaries | Median words | Maximum words | Outputs at 250-token cap |
|---|---:|---:|---:|---:|
| Concise | 150 | 68 | 100 | 0 |
| Complete | 150 | 81 | 150 | 1 |

The one capped complete output is `soccernet-bbc:281`, fan A; it ends mid-statistic after `Corners: 5 (SC Freiburg), 3 (`. No pilot output reached the cap. Thus the 250-token limit is a real failure for one dev summary, but it does not explain most omissions: the complete mode usually stopped well below 150 words.

## SportsSum-style matching, metric version 2

The v1 event extractor missed timed goal lists such as `Goals came from Benzema (29'), Isco (40')`. Version `sportssum-bbc-v2` removes only parenthesized minute annotations before extracting the names in such lists. It was regression-tested and applied equally to both new modes and the old baseline; no model outputs were regenerated for this change. Files are `results/dev_record_{concise,complete}_sportssum_v2.csv` and corresponding JSONL details. The table below pools counts over 150 matched summaries per mode (50 independent matches, three audience conditions each), all 50 dev matches:

| Mode | Metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Concise | Player names (NMS) | 1.000 | 0.258 | 0.410 |
| Complete | Player names (NMS) | 0.997 | 0.495 | 0.662 |
| Concise | All recorded player-events (EMS) | 0.969 | 0.103 | 0.185 |
| Complete | All recorded player-events (EMS) | 0.924 | 0.132 | 0.230 |
| Concise | Goal/red-card player-events | 0.969 | 0.637 | 0.769 |
| Complete | Goal/red-card player-events | 0.938 | 0.705 | 0.805 |

The complete instruction increases recognized name and event coverage, with lower measured precision. It does **not** achieve complete coverage: no complete-mode dev summary received all-event recall 1.0, and only 9/150 received name recall 1.0. The all-event metric is also sensitive to paraphrases and does not check every field. Do not treat these numbers as comprehensive factual precision or independent 150-match observations. `results/dev_baselines_sportssum_v2.csv` rescored the old baseline with the same parser version for any baseline comparisons.

## Source-grounded examples and limitations

For training match `soccernet-bbc:433`, the BBC record has 15 events, including four substitutions and five yellow cards, plus five two-team statistics. The complete neutral output lists goal scorers but omits all substitution details and most card details. It says Real Madrid had three yellow cards; the record lists one. It says both teams had five corners; the record gives Getafe five and Real Madrid six. The complete fan A output places Pablo Sarabia's 83rd-minute goal just before halftime. The complete fan B output says Getafe had more corners while showing `5-6`. These examples show that an authoritative-record system prompt does not by itself prevent unsupported claims or enforce inclusion of every record field.

The separate conservative rule checker found some score/winner/minute contradictions, but it also mislabels some team possessives as player names and has narrow claim coverage. Its fractions are not a comprehensive error rate. SportsSum v2 still misses some event paraphrases and can misread complex lists. A full-field coverage and contradiction checker for score, times, team associations, and statistics is needed before claiming that the complete arm verbalizes the entire BBC record.

**Next decision:** keep these dev runs as the first measured comparison. Before generating test summaries, choose a revised `complete` output design and rerun a train-only pilot: either allow a longer structured match report with explicit event/statistic lines, or restrict the required fact set. The current 150-word/250-token setup has already failed its stated complete-coverage objective.
