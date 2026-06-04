#!/usr/bin/env python3
"""
Preprocess raw track data into the pipeline's unified JSONL format.

For each track, produces:
  {track}/topics/queries_unified.jsonl   {qid, query, meta}       JSONL
  {track}/corpus/corpus.jsonl            {docid, title, text, url, meta}  JSONL

Input files are read from the raw locations laid out by download.sh.
Gzip-compressed inputs (.jsonl.gz) are handled automatically.
Multi-shard corpus directories are merged into a single output file.

Usage:
  python data/preprocess.py rag2024  [--data-dir data]
  python data/preprocess.py biogen   [--data-dir data]
  python data/preprocess.py ragtime  [--data-dir data]
  python data/preprocess.py all      [--data-dir data]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# Repo root so scripts/converters/ is importable regardless of cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.converters.rag2024 import (
    convert_corpus_line as rag_corpus_line,
    convert_queries_line as rag_queries_line,
)
from scripts.converters.biogen import (
    convert_corpus_line as biogen_corpus_line,
    convert_queries_line as biogen_queries_line,
)
from scripts.converters.ragtime import (
    convert_corpus_line as ragtime_corpus_line,
    convert_queries_line as ragtime_queries_line,
)


def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def find_raw_jsonl(directory: Path) -> list[Path]:
    """Return all .jsonl / .jsonl.gz files in directory, excluding unified outputs."""
    exts = ("*.jsonl", "*.jsonl.gz")
    files = []
    for pattern in exts:
        for p in sorted(directory.glob(pattern)):
            if "unified" not in p.name:
                files.append(p)
    return files


def convert_files(
    inputs: list[Path],
    output: Path,
    convert_fn,
    **kwargs,
) -> int:
    """Apply convert_fn to every non-empty line in inputs; write results to output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as fout:
        for path in inputs:
            with open_maybe_gz(path) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = convert_fn(line, **kwargs) if kwargs else convert_fn(line)
                        fout.write(result + "\n")
                        count += 1
                    except Exception as exc:
                        print(
                            f"  [warn] Skipping malformed line in {path.name}: {exc}",
                            file=sys.stderr,
                        )
    return count


def tsv_to_unified(tsv_path: Path, output: Path) -> int:
    """Convert anserini TSV (qid<TAB>query) to unified JSONL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(tsv_path, encoding="utf-8") as fin, open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                fout.write(json.dumps({"qid": parts[0], "query": parts[1], "meta": {}}) + "\n")
                count += 1
    return count


# ---------------------------------------------------------------------------
# Per-track preprocessing
# ---------------------------------------------------------------------------

def preprocess_rag2024(data_dir: Path) -> None:
    track_dir = data_dir / "rag2024"
    print("=== RAG 2024 ===")

    # Topics ─ prefer JSONL (ir_datasets) over TSV (anserini-tools)
    jsonl_topics = track_dir / "topics" / "queries.jsonl"
    tsv_topics   = track_dir / "topics" / "queries.tsv"
    out_topics   = track_dir / "topics" / "queries_unified.jsonl"

    if jsonl_topics.exists():
        n = convert_files([jsonl_topics], out_topics, rag_queries_line)
        print(f"  topics  : {n} queries  →  {out_topics.relative_to(data_dir.parent)}")
    elif tsv_topics.exists():
        n = tsv_to_unified(tsv_topics, out_topics)
        print(f"  topics  : {n} queries (TSV)  →  {out_topics.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No topics found. Expected:")
        print(f"         {jsonl_topics}  ({{\"topic_id\": ..., \"topic\": ...}})")
        print(f"         {tsv_topics}    (qid<TAB>query)")

    # Corpus ─ may be multiple shards (loop & merge)
    corpus_dir = track_dir / "corpus"
    out_corpus  = corpus_dir / "corpus.jsonl"
    shards = find_raw_jsonl(corpus_dir)

    if shards:
        n = convert_files(shards, out_corpus, rag_corpus_line)
        print(f"  corpus  : {n} docs from {len(shards)} shard(s)  →  {out_corpus.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No corpus shards found in {corpus_dir}")
        print(f"         Download MS MARCO v2.1 segmented JSONL shards there.")
        print(f"         See: bash data/download.sh rag2024")


def preprocess_biogen(data_dir: Path) -> None:
    track_dir = data_dir / "biogen"
    print("=== BioGen 2025 ===")

    # Topics
    raw_topics = track_dir / "topics" / "queries.jsonl"
    out_topics  = track_dir / "topics" / "queries_unified.jsonl"

    if raw_topics.exists():
        n = convert_files([raw_topics], out_topics, biogen_queries_line)
        print(f"  topics  : {n} queries  →  {out_topics.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No topics found at {raw_topics}")
        print(f"         Download from TREC active participants: https://trec.nist.gov/act_part/")
        print(f"         Format: {{\"qa_id\": \"BG2025_001\", \"question\": \"...\"}}")

    # Corpus ─ may be extracted to multiple JSONL files
    corpus_dir = track_dir / "corpus"
    out_corpus  = corpus_dir / "corpus.jsonl"
    raw_files = find_raw_jsonl(corpus_dir)

    if raw_files:
        n = convert_files(raw_files, out_corpus, biogen_corpus_line)
        print(f"  corpus  : {n} docs from {len(raw_files)} file(s)  →  {out_corpus.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No corpus files found in {corpus_dir}")
        print(f"         Download from: http://bionlp.nlm.nih.gov/biogen-2025-document-collection.zip")
        print(f"         Unzip to {corpus_dir}/")
        print(f"         Format: {{\"pmid\": \"12345678\", \"title\": \"...\", \"abstract\": \"...\"}}")


def preprocess_ragtime(data_dir: Path) -> None:
    track_dir = data_dir / "ragtime"
    print("=== RAGTIME 2025 ===")

    # Topics
    raw_topics = track_dir / "topics" / "queries.jsonl"
    out_topics  = track_dir / "topics" / "queries_unified.jsonl"

    if raw_topics.exists():
        n = convert_files([raw_topics], out_topics, ragtime_queries_line)
        print(f"  topics  : {n} queries  →  {out_topics.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No topics found at {raw_topics}")
        print(f"         Download from TREC active participants: https://trec.nist.gov/act_part/")
        print(f"         Format: {{\"topic_id\": \"...\", \"problem_statement\": \"...\", \"background\": \"...\", \"report_length\": 2000}}")

    # Corpus ─ one or more language-split JSONL files
    corpus_dir = track_dir / "corpus"
    out_corpus  = corpus_dir / "corpus.jsonl"
    raw_files = find_raw_jsonl(corpus_dir)

    if raw_files:
        n = convert_files(raw_files, out_corpus, ragtime_corpus_line)
        print(f"  corpus  : {n} docs from {len(raw_files)} file(s)  →  {out_corpus.relative_to(data_dir.parent)}")
    else:
        print(f"  [warn] No corpus files found in {corpus_dir}")
        print(f"         Download via HuggingFace: huggingface-cli download trec-ragtime/ragtime2 \\")
        print(f"           --repo-type dataset --local-dir {corpus_dir}")
        print(f"         Format: {{\"docid\": \"...\", \"title\": \"...\", \"text\": \"...\", \"lang\": \"ar|zh|en|ru\"}}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "track",
        choices=["rag2024", "biogen", "ragtime", "all"],
        help="Track to preprocess",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Root data/ directory (default: directory containing this script)",
    )
    args = parser.parse_args()

    if args.track in ("rag2024", "all"):
        preprocess_rag2024(args.data_dir)
        print()
    if args.track in ("biogen", "all"):
        preprocess_biogen(args.data_dir)
        print()
    if args.track in ("ragtime", "all"):
        preprocess_ragtime(args.data_dir)
        print()

    print("Done. Run the pipeline with:")
    if args.track == "all":
        for cfg in ("rag2024", "biogen", "ragtime"):
            print(f"  python run_rag.py config/{cfg}.yaml")
    else:
        print(f"  python run_rag.py config/{args.track}.yaml")


if __name__ == "__main__":
    main()
