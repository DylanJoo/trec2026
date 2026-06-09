"""
Convert TREC NeuCLIR 2024 native formats to unified pipeline format.

Native corpus (JSONL via ir_datasets neuclir/1/zh, neuclir/1/fa, neuclir/1/ru):
  {"doc_id": "...", "title": "...", "text": "...", "url": "...", "time": "...", "cc_file": "..."}

Native report-generation requests (JSONL, TREC 2024 format):
  {"request_id": "300", "collection_ids": ["neuclir/1/all"],
   "background": "I am a researcher...",
   "problem_statement": "I need a report on...",
   "limit": 2000, "title": "Japan suicide rate COVID-19"}

Legacy ad-hoc queries (JSONL):
  {"query_id": "NeuCLIR-2024-01", "title": "...", "description": "...", "lang": "zh"}

Native report run (JSONL):
  {"run_id": "...", "query_id": "...",
   "report": "English report text",
   "citations": ["doc_id1", "doc_id2"]}
"""

import json


def convert_corpus_line(line: str) -> str:
    doc = json.loads(line)
    unified = {
        "docid": doc.get("doc_id", doc.get("docid", "")),
        "title": doc.get("title", ""),
        "text": doc.get("text", ""),
        "url": doc.get("url", ""),
        "meta": {
            "time": doc.get("time", ""),
            "lang": doc.get("lang", ""),
        },
    }
    return json.dumps(unified)


def convert_queries_line(line: str) -> str:
    q = json.loads(line)
    # Report-generation request format (TREC 2024 NeuCLIR)
    if "request_id" in q:
        unified = {
            "qid": str(q["request_id"]),
            "query": q.get("problem_statement", ""),
            "meta": {
                "title": q.get("title", ""),
                "background": q.get("background", ""),
                "collection_ids": q.get("collection_ids", []),
                "limit": q.get("limit", 2000),
                "track": "neuclir",
            },
        }
    else:
        # Legacy ad-hoc query format
        unified = {
            "qid": str(q.get("query_id", q.get("qid", ""))),
            "query": q.get("title", q.get("query", "")),
            "meta": {
                "description": q.get("description", ""),
                "target_lang": q.get("lang", ""),
                "track": "neuclir",
            },
        }
    return json.dumps(unified)


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    report = r.get("report", "")
    citations = r.get("citations", [])
    unified = {
        "qid": str(r.get("query_id", r.get("qid", ""))),
        "response": report,
        "sentences": [{"text": report, "citations": citations}],
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
