"""
Convert nugget annotation files to the pipeline's per-query qrel format.

Supported input formats:

1. AutoNuggetizer JSONL (TREC RAG 2024/2025):
   {"qid": "2027497", "nuggets": [{"text": "...", "importance": "vital", "docids": ["d1", ...]}, ...]}

2. NIST nugget pool (legacy TREC QA format):
   qid nugget_id vital|okay doc_id [doc_id ...]

Output per query (our format):  docid  nugid  relevance
Written to: {output_dir}/{qid}.qrel
"""

import json
import os
import argparse
from collections import defaultdict


def convert_autonuggetizer(input_path: str, output_dir: str):
    """Convert AutoNuggetizer JSONL to per-query qrel files."""
    os.makedirs(output_dir, exist_ok=True)
    total_queries = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj["qid"])
            nuggets = obj.get("nuggets", [])

            doc_nuggets: dict[str, set] = defaultdict(set)
            for nug_idx, nug in enumerate(nuggets):
                nug_id = nug.get("nugget_id", f"nug_{nug_idx:04d}")
                importance = nug.get("importance", "okay")
                rel = 2 if importance == "vital" else 1
                for docid in nug.get("docids", []):
                    doc_nuggets[docid].add((nug_id, rel))

            out_path = os.path.join(output_dir, f"{qid}.qrel")
            with open(out_path, "w") as fout:
                for docid, nug_set in doc_nuggets.items():
                    for nug_id, rel in sorted(nug_set):
                        fout.write(f"{docid} {nug_id} {rel}\n")
            total_queries += 1

    print(f"Wrote {total_queries} qrel files to {output_dir}/")


def convert_nist_pool(input_path: str, output_dir: str):
    """Convert legacy NIST nugget pool (all queries in one file) to per-query qrel files."""
    os.makedirs(output_dir, exist_ok=True)
    by_query: dict[str, list[str]] = defaultdict(list)

    with open(input_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, nug_id, importance = parts[0], parts[1], parts[2]
            rel = 2 if importance == "vital" else 1
            for docid in parts[3:]:
                by_query[qid].append(f"{docid} {nug_id} {rel}")

    for qid, lines in by_query.items():
        out_path = os.path.join(output_dir, f"{qid}.qrel")
        with open(out_path, "w") as fout:
            fout.write("\n".join(lines) + "\n")

    print(f"Wrote {len(by_query)} qrel files to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("format", choices=["autonuggetizer", "nist"])
    p.add_argument("input", help="Input nugget file")
    p.add_argument("output_dir", help="Directory to write per-query .qrel files")
    args = p.parse_args()

    if args.format == "autonuggetizer":
        convert_autonuggetizer(args.input, args.output_dir)
    else:
        convert_nist_pool(args.input, args.output_dir)
