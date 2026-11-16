"""Non-extracting ZIP inspection. No source schema assumptions."""
import argparse
import collections
import json
import re
import zipfile
from pathlib import Path
from .common import sha256_file

def inspect(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archive missing: {path}. Run python3 -m src.data.download first.")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        suffix = collections.Counter(Path(i.filename).suffix.lower() or "[none]" for i in files)
        top = collections.Counter(i.filename.split("/")[0] for i in files)
        directories=collections.Counter(str(Path(i.filename).parent) for i in files)
        candidate_ids=sorted({int(m.group(1)) for folder in directories if (m:=re.match(r"^([0-9]+)_",Path(folder).name))})
        samples = []
        for i in files[:10]:
            item = {"path": i.filename, "bytes": i.file_size}
            if i.file_size < 100000 and Path(i.filename).suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".xml"}:
                content = z.read(i).decode("utf-8", errors="replace")
                item["sample_first_500_chars"] = content[:500]
            samples.append(item)
        return {"archive": str(path), "sha256": sha256_file(path), "file_count": len(files), "total_uncompressed_bytes": sum(i.file_size for i in files), "types": dict(suffix), "top_level": dict(top), "candidate_match_count":len(candidate_ids),"candidate_match_ids":candidate_ids,"files_per_match_folder":dict(collections.Counter(directories.values())), "samples": samples, "all_paths": [i.filename for i in files]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    result = inspect(a.input)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary={k:v for k,v in result.items() if k not in ("all_paths","candidate_match_ids")}
    summary["candidate_match_id_sample"]=result["candidate_match_ids"][:10]
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
