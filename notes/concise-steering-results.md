# Concise steering: development run inspection

Inspected the synced, Git-ignored `results/` files locally. This is a **development** analysis, not a final test result.

## Integrity and execution

- `results/train_record_concise_pairs.jsonl`: 400 completed concise fan generations, validated as 200 aligned train A/B pairs by `src.activations --validate-only`.
- `results/concise_vectors_full.json`: full (non-smoke) 200-pair mean A-minus-B vector, pinned Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, layer norms 2.049 (12), 3.954 (20), and 18.525 (26). The recorded training-pair and vector SHA-256 checksums match the synced files.
- `results/concise_steering_dev_l20.jsonl`: 850/850 completed, nonempty outputs on exactly the 50 dev match IDs: 50 no-hook, 200 learned-vector, and 600 random-vector generations. All rows use one experiment ID, vector checksum, and split manifest. No match overlaps train or test. No output hit the 250-new-token cap.
- The no-hook output matches the existing `results/dev_record_concise.jsonl` neutral summary text and prompt hash for all 50 matches. This also exercises hook removal across successive matches in the generation loop.

## Observable changes and limited factual checks

| Arm | Text changed from no hook | SportsSum salient event precision / recall / F1 | Narrow rule precision |
| --- | ---: | ---: | ---: |
| No hook | 0/50 | .960 / .632 / .762 | .989 |
| Learned −0.5 | 29/50 | .961 / .640 / .768 | .989 |
| Learned +0.5 | 39/50 | .985 / .579 / .729 | .989 |
| Learned −1 | 39/50 | .961 / .649 / .775 | .989 |
| Learned +1 | 42/50 | .972 / .605 / .746 | .988 |

At coefficient magnitude 1, individual random controls changed 36–41 of 50 texts and scored .732–.814 on salient-event F1. Thus text changes and these factuality scores do not establish a learned-vector advantage. The SportsSum scorer recognizes only a limited inventory of names and event wording; its full event recall is low because concise summaries omit many listed BBC events. The rule precision covers only recognized explicit claims and is not comprehensive factual precision.

The rule checker flags one contradiction in every arm for `soccernet-bbc:229`: the text says Metz defeated Paris SG while the record and stated score are Metz 2, Paris SG 3. This error is already present in the no-hook baseline. One random arm (`random_alpha_+1_seed_43`) additionally says Athletic Bilbao defeated Real Madrid despite a stated 1–2 loss in `soccernet-bbc:407`; the rule checker flags the sentence under its scorer rule, illustrating its limited classification. No additional contradiction was flagged in the learned arms, but these checks do not rule out other unsupported facts.

## Interpretation and next measurement

The current results verify that the extraction and steering pipeline ran and altered generated text. They **do not yet measure team perspective**. A blinded, match-aware −3 (Team A) to +3 (Team B) stance rubric, with human calibration of an independent LLM judge or direct human ratings, is required before choosing a layer/strength or claiming predictable fan-perspective steering. Compare learned arms with no-hook and norm-matched random controls by match; keep the 49-match test fold untouched until the method is fixed on dev.
