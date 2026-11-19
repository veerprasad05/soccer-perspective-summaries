# First Luna stance-judge pilot inspection

The first synced pilot, `results/stance_judge_pilot_luna.jsonl`, contains 20
ratings over four development matches and five arms. It used
`bbc-supporter-stance-v1`, with 16 API calls and four reused ratings for
identical BBC-record/summary requests. All 20 rows match the saved source
summary hashes; the 49-match test split was not involved.

The pilot exposed a **sign-mapping failure**. In `soccernet-bbc:463`, Real
Madrid is Team A and Granada CF is Team B. Four summaries were rated `+1`
even though their rationales say the text favors Real Madrid. On the stated
scale, `+1` means slight Team B preference. This makes the v1 arm means
unreliable for choosing a steering strength or claiming a perspective shift.
The v1 report listed mean stance 0.5 for no hook, 0.5 for learned +1, and
0.25 for learned -1, but these figures should not be interpreted.

The judge also rated two random-arm summaries for a 6-0 Real Madrid win as
`+2` (Team B) because they ended with "dominating" language. That may be
defensible if the wording is celebratory, but it may also reflect outcome
bias. A blinded human review is required to decide whether this rubric and
model distinguish supporter framing from objective dominance.

The revised `bbc-supporter-stance-v2` asks for the **exact favored team name**
or `neither`, plus a 0-3 strength. The request schema restricts the answer
to the two team names in that match; code maps the name to the signed -3 to
+3 score. This removes the observed numerical polarity mismatch. It does
not guarantee that the model identifies the correct favored team, so rerun
the pilot with a separate `..._v2.jsonl` file and compare it with blinded
human scores before judging the full 850 development outputs.
