"""Future activation extraction; no experimental results are implemented."""
def validate_train_pairs(rows):
    pairs = {}
    for row in rows:
        if row.get("split") != "train": raise ValueError("Activation pairs must use train only")
        if row.get("condition") not in ("fan_A","fan_B") or row.get("status") != "complete": raise ValueError("Need completed fan_A/fan_B rows")
        pairs.setdefault(row["match_id"], {})[row["condition"]] = row
    missing=[mid for mid,p in pairs.items() if set(p)!={"fan_A","fan_B"}]
    if missing: raise ValueError(f"Incomplete train pairs: {missing[:5]}")
    return pairs
if __name__=="__main__": raise SystemExit("Activation extraction pending: source data and baseline pairs are required")
