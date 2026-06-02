#!/usr/bin/env python3
"""Normalize TREC RAG-style track data into this repository's format.

Normalized formats:
- queries.tsv: qid<TAB>query_text
- corpus.jsonl: {"docid": "...", "contents": "..."}
- run.trec: qid Q0 docid rank score run_id
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator, TextIO


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_input_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix in {".jsonl", ".json", ".gz", ".txt", ".tsv", ".run", ".trec"}:
            yield p


def read_jsonl(path: Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _first(obj: dict, keys: list[str], default: str = "") -> str:
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return str(v)
    return default


def _query_text(track: str, obj: dict) -> str:
    base = ["query", "title", "question", "narrative", "problem_statement", "request"]
    if track == "autojudge":
        parts = [
            _first(obj, ["title", "query", "question"]),
            _first(obj, ["problem_statement", "narrative"]),
            _first(obj, ["background"]),
        ]
        return " ".join([p.strip() for p in parts if p and p.strip()])
    return _first(obj, base)


def normalize_queries(track: str, input_path: Path, output_path: Path) -> int:
    rows: list[tuple[str, str]] = []
    for p in iter_input_files(input_path):
        if p.suffix in {".txt", ".tsv"}:
            with open_text(p) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if "\t" in line:
                        qid, text = line.split("\t", 1)
                        rows.append((qid.strip(), text.strip()))
            continue

        for obj in read_jsonl(p):
            meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
            qid = _first(
                obj,
                ["id", "topic_id", "narrative_id", "qa_id", "request_id", "query_id"],
                default=_first(meta, ["topic_id", "narrative_id", "qa_id", "request_id"]),
            )
            text = _query_text(track, obj) or _first(meta, ["question", "narrative", "title"])
            if qid and text:
                rows.append((qid.strip(), " ".join(text.split())))

    unique: dict[str, str] = {}
    for qid, text in rows:
        if qid not in unique:
            unique[qid] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for qid in sorted(unique):
            f.write(f"{qid}\t{unique[qid]}\n")
    return len(unique)


def _docid(obj: dict) -> str:
    return _first(obj, ["docid", "id", "doc_id", "document_id", "pmid"])


def _text(obj: dict) -> str:
    candidates = [
        "contents",
        "text",
        "segment",
        "body",
        "abstract",
        "passage",
        "content",
    ]
    text = _first(obj, candidates)
    title = _first(obj, ["title"])
    if title and text and not text.startswith(title):
        return f"{title}\n\n{text}".strip()
    return (text or title).strip()


def normalize_corpus(input_path: Path, output_path: Path) -> int:
    docs: dict[str, str] = {}
    for p in iter_input_files(input_path):
        if p.suffix in {".txt", ".tsv", ".run", ".trec"}:
            continue
        for obj in read_jsonl(p):
            docid = _docid(obj)
            text = _text(obj)
            if docid and text and docid not in docs:
                docs[docid] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for docid in sorted(docs):
            f.write(json.dumps({"docid": docid, "contents": docs[docid]}, ensure_ascii=False) + "\n")
    return len(docs)


def _parse_trec_line(line: str) -> tuple[str, str, int, float, str] | None:
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    try:
        return parts[0], parts[2], int(parts[3]), float(parts[4]), parts[5]
    except ValueError:
        return None


def normalize_run(input_path: Path, output_path: Path, default_run_id: str = "run") -> int:
    entries: list[tuple[str, str, int, float, str]] = []
    for p in iter_input_files(input_path):
        with open_text(p) as f:
            first = f.readline()
            if not first:
                continue
            line = first.strip()
            f.seek(0)

            if line.startswith("{"):
                for obj in read_jsonl(p):
                    qid = _first(obj, ["qid", "query_id", "topic_id", "request_id"])
                    run_id = _first(obj, ["run_id"], default=default_run_id)
                    ranked = obj.get("ranked_docs", obj.get("hits", []))
                    if not isinstance(ranked, list):
                        continue
                    for i, item in enumerate(ranked, 1):
                        if isinstance(item, dict):
                            docid = _first(item, ["docid", "id", "doc_id"])
                            score = float(item.get("score", 0.0))
                            rank = int(item.get("rank", i))
                        elif isinstance(item, list) and len(item) >= 2:
                            docid = str(item[0])
                            score = float(item[1])
                            rank = i
                        else:
                            continue
                        if qid and docid:
                            entries.append((qid, docid, rank, score, run_id))
            else:
                for ln in f:
                    parsed = _parse_trec_line(ln)
                    if parsed:
                        entries.append(parsed)

    entries.sort(key=lambda x: (x[0], x[2], -x[3], x[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for qid, docid, rank, score, run_id in entries:
            f.write(f"{qid} Q0 {docid} {rank} {score:.8f} {run_id}\n")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_q = sub.add_parser("queries", help="normalize topics/queries")
    p_q.add_argument("--track", choices=["rag", "ragtime", "biogen", "autojudge"], required=True)
    p_q.add_argument("--input", type=Path, required=True)
    p_q.add_argument("--output", type=Path, required=True)

    p_c = sub.add_parser("corpus", help="normalize corpus")
    p_c.add_argument("--input", type=Path, required=True)
    p_c.add_argument("--output", type=Path, required=True)

    p_r = sub.add_parser("run", help="normalize retrieval run file")
    p_r.add_argument("--input", type=Path, required=True)
    p_r.add_argument("--output", type=Path, required=True)
    p_r.add_argument("--default-run-id", default="run")

    args = parser.parse_args()

    if args.cmd == "queries":
        n = normalize_queries(args.track, args.input, args.output)
        print(f"Wrote {n} queries to {args.output}")
    elif args.cmd == "corpus":
        n = normalize_corpus(args.input, args.output)
        print(f"Wrote {n} documents to {args.output}")
    elif args.cmd == "run":
        n = normalize_run(args.input, args.output, default_run_id=args.default_run_id)
        print(f"Wrote {n} run rows to {args.output}")


if __name__ == "__main__":
    main()
