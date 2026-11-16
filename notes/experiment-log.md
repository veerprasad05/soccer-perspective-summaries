# Experiment log

2026-11-15: CPU pipeline tests only. No model inference, no API calls, no cluster access, no real-match results.

2026-11-15: User approved 299 eligible BBC matches split 200/50/49. Created frozen manifest with seed 42; verified all eligible IDs used once and no overlap. Repeated the split command and confirmed the manifest SHA-256 stayed `2a4aa142931c986c1695b8e40e4fc715ded4d49ecfbf6ba4de2a2ced442ef1ab`. Ran a two-match train-only dry-run prompt preview: six condition rows, identical factual-input hash across conditions for each match, and no manifest change. No Qwen inference or API calls.
