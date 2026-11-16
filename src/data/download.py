"""Fetch only the research repository's public archive, with checksum logging."""
import argparse
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path
URL = "https://raw.githubusercontent.com/simula/soccer-summarization/main/Datasets/SoccerNet-BBC.zip"
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/SoccerNet-BBC.zip")
    a = p.parse_args()
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, tmp.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block); h.update(block)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)
    meta = {"url": URL, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sha256": h.hexdigest(), "bytes": target.stat().st_size}
    target.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
if __name__ == "__main__": main()
