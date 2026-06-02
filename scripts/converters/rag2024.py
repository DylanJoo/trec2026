"""
Convert TREC RAG 2024/2025 native formats to unified pipeline format.

Native corpus (JSONL per shard):
  {"docid": "msmarco_v2.1_doc_XX_YYYY#N_ZZZZ",
   "url": "...", "title": "...", "headings": "...",
   "segment": "...", "start_char": N, "end_char": N}

Native queries (JSONL):
  {"topic_id": "2027497", "topic": "..."}

Native generation run (JSONL):
  {"run_id": "...", "topic_id": "...", "topic": "...",
   "references": ["docid", ...],
   "response_length": N,
   "answer": [{"text": "sentence.", "citations": ["docid", ...]}, ...]}
"""

import json
import sys


def convert_corpus_line(line: str) -> str:
    doc = json.loads(line)
    unified = {
        "docid": doc["docid"],
        "title": doc.get("title", ""),
        "text": doc.get("segment", ""),
        "url": doc.get("url", ""),
        "meta": {
            "headings": doc.get("headings", ""),
            "start_char": doc.get("start_char"),
            "end_char": doc.get("end_char"),
        },
    }
    return json.dumps(unified)


def convert_queries_line(line: str) -> str:
    q = json.loads(line)
    unified = {"qid": str(q["topic_id"]), "query": q["topic"], "meta": {}}
    return json.dumps(unified)


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    sentences = [
        {"text": s["text"], "citations": s.get("citations", [])}
        for s in r.get("answer", [])
    ]
    unified = {
        "qid": str(r["topic_id"]),
        "response": " ".join(s["text"] for s in sentences),
        "sentences": sentences,
    }
    return json.dumps(unified)


def convert_file(input_path: str, output_path: str, mode: str):
    converter = {"corpus": convert_corpus_line,
                 "queries": convert_queries_line,
                 "response": convert_response_line}[mode]
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if line:
                fout.write(converter(line) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["corpus", "queries", "response"])
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    convert_file(args.input, args.output, args.mode)
    print(f"Converted {args.mode}: {args.input} -> {args.output}")
