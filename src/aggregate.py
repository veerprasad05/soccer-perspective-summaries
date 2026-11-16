"""Join rows by run ID without crossing split boundaries."""
def join(generations, evaluations):
    by_id={e["run_id"]:e for e in evaluations}
    return [{**g,"evaluation":by_id.get(g["run_id"])} for g in generations]
