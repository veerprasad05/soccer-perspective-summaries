# SoccerNet-BBC archive audit — 2026-11-15 local / 2026-11-16 UTC

## Provenance and archive identity

The user supplied `data/raw/SoccerNet-BBC.zip`. It is a valid ZIP (`unzip -t` found no errors), 1,271,176 bytes. `git hash-object` gives **175c5945b878b82c7093fea95e79437c730e677b**, exactly the Git blob SHA reported for `Datasets/SoccerNet-BBC.zip` in the [research repository](https://github.com/simula/soccer-summarization/tree/main/Datasets). The ZIP SHA-256 is **75b039655cf36d5b6d764ac67beb12a1cda4f10f71decf769a47f461d5f6960a**. The raw archive is ignored by Git. The `:Zone.Identifier` sidecar is an operating-system download marker and was not parsed.

The archive has **1,360 `.txt` files in 340 match folders**, exactly four files per folder: `commentary.txt`, `info.txt`, `lineup.txt`, and `match_stats.txt`. Match folder numeric IDs are unique and range from 1 to 500, with 160 IDs absent. `info.txt` supplies a BBC URL and a date string; there is no article body file. `match_stats.txt` provides home/away names and possession, shots, shots on target, corners, and fouls. `commentary.txt` supplies timestamped prose; the parser extracts explicit goals, bookings, dismissals, and substitutions. `lineup.txt` is present but not used. See `notes/archive-inspection.json` for all file paths, sizes, and samples.

The repository presents a CC-BY-4.0 license, but that alone does not settle rights in BBC text or SoccerNet media. The project stores the raw ZIP and parsed records under Git-ignored data directories. Do not publish those files until source-specific reuse terms are reviewed.

## Parsing and eligibility

`src/data/soccernet_bbc.py` is the source-specific parser. It uses folder IDs as stable match IDs, folder names and scores as the primary match identifier and result, and checks the BBC commentary's explicit final score. It aligns team names before accepting statistics or commentary: some files name the teams in reverse order, and some appear to describe another fixture. Unsafe joins are warnings and exclusions. Goals repeated in duplicated commentary passages are deduplicated using minute, scorer, and displayed score, while all source line references remain attached. Unknown event teams or players stay null. No goals are inferred from the final score.

For example, folder `soccernet-bbc:242` names Wolfsburg 2–1 Dortmund, while its commentary's final line names Borussia Dortmund 0–1 FC Bayern München; it is excluded as a team mismatch. In `soccernet-bbc:238`, the BBC files list Hertha Berlin before Bayern Munich, reversing the folder order; the parser maps teams and statistics back to A/B and keeps the match. Team-pair alignment uses normalized-name similarity with conservative thresholds (both sides at least 0.4, combined at least 0.9, orientation margin at least 0.15); borderline cases require manual review.

Eligibility was fixed as: distinct named teams; integer nonnegative final score; at least one timed, typed event; unique IDs; no recorded goal count greater than the final score; no unresolved commentary score/team mismatch; and no statistics file identified as another fixture. A final commentary line is desirable but not mandatory when the score is explicit in the folder name; six eligible records lack it. Criteria and all exclusions are machine-readable in `data/processed/audit.json`.

| Audit item | Actual count |
|---|---:|
| Raw match folders / normalized records | 340 / 340 |
| Eligible / excluded | **299 / 41** |
| Eligible outcomes | 134 A wins, 113 B wins, 52 draws |
| Eligible events | 796 goals, 1,569 substitutions, 1,009 yellow cards, 45 red cards |
| Eligible records with complete, team-attributed goal count | 154 / 299 |
| Eligible records with incomplete goal event list | 145 / 299 |
| Eligible records with no statistics | 0 |

Exclusion reasons overlap: 29 score contradictions, 8 statistics team mismatches, 7 commentary team mismatches, 8 goal counts above the folder score, and 4 records without timed events. The parser flagged duplicated goal commentary in 174 match records. These counts describe parser diagnostics, not independent scientific results. The **processed JSONL SHA-256** for this run is **26c4dd8bbdde870d694e0da690d5bee642f37dff906782dcc41dcedfe98febf3** (parser version 0.1.0). Re-running normalization should yield the same checksum.

The event list is often incomplete: 796 goal entries versus 999 total goals in final scores across eligible matches. The prompt explicitly warns that unlisted events must not be inferred. Named-scorer coverage is evaluated only over recorded goal events.

## Source-line spot checks

The following six eligible records were compared against `commentary.txt` at the stored line references, including their explicit final score lines:

| Match ID | Result | Checked goal evidence |
|---|---|---|
| `soccernet-bbc:102` | Manchester City 1–2 CSKA Moscow | Line 277: Seydou Doumbia, CSKA Moscow goal at 2' |
| `soccernet-bbc:010` | Chelsea 1–2 Crystal Palace | Line 188: Bakary Sako, Crystal Palace goal at 65'; repeated passages deduplicated |
| `soccernet-bbc:198` | Reims 1–1 Paris SG | Line 46: Theoson Siebatcheu, Reims goal at 83' |
| `soccernet-bbc:235` | Paderborn 0–6 Bayern Munich | Line 286: Robert Lewandowski, Bayern goal at 37'; source goal list incomplete |
| `soccernet-bbc:290` | Inter 0–0 AC Milan | Final line confirms 0–0; no goal event |
| `soccernet-bbc:384` | Real Madrid 2–0 Dep. La Coruna | Line 110: Karim Benzema, Real Madrid goal at 73'; source goal list incomplete |

These checks confirm the cited examples, not every parsed event. String-based team alignment and event extraction still need broader manual error review before model evaluation.

## Split decision and result

The original **300/100/100** command failed as expected: `Need 500 distinct eligible matches ... found 299, shortage 201`. The user then explicitly approved a **200/50/49** match-level split of the 299 eligible BBC matches. This is a documented change from the submitted proposal and research specification, not a claim that the original 500-match target was met.

`data/splits/train.json`, `dev.json`, `test.json`, and `manifest.json` now exist. Their counts are 200/50/49; all 299 eligible IDs appear once, with no overlap. Seed 42, deterministic ordering, source checksum, eligible-ID checksum, exclusions, and team/outcome diagnostics are in the manifest. Manifest SHA-256: **2a4aa142931c986c1695b8e40e4fc715ded4d49ecfbf6ba4de2a2ced442ef1ab**. Outcome counts are train 93 A wins / 76 B wins / 31 draws; dev 21 / 19 / 10; test 20 / 18 / 11. The test fold is frozen for final evaluation.

The official [SoccerNet SN-Labels dataset](https://huggingface.co/datasets/SoccerNet/SN-Labels) remains a possible future extension, but it is **not part of this approved BBC-only split**.
