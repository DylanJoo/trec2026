"""
Convert TREC DRAGUN 2025 native formats to unified pipeline format.

DRAGUN topics (JSONL):
  {"topic_id": "...", "article_title": "...", "article_text": "...", "source": "..."}

DRAGUN response (JSONL):
  {"run_id": "...", "topic_id": "...",
   "report": "Full report text with inline [doc_id] citations.",
   "citations": ["docid1", ...]}
"""

import json
import re


def convert_queries_line(line: str) -> str:
    q = json.loads(line)
    unified = {
        "qid": str(q["topic_id"]),
        "query": q.get("article_title", "") + " " + q.get("article_text", ""),
        "meta": {
            "article_title": q.get("article_title", ""),
            "source": q.get("source", ""),
            "track": "dragun",
        },
    }
    return json.dumps(unified)


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    report = r.get("report", "")
    inline_citations = re.findall(r'\[([^\]]+)\]', report)
    explicit_citations = r.get("citations", [])
    all_citations = list(dict.fromkeys(inline_citations + explicit_citations))

    unified = {
        "qid": str(r["topic_id"]),
        "response": report,
        "sentences": [{"text": report, "citations": all_citations}],
    }
    return json.dumps(unified)


def convert_file(input_path: str, output_path: str, mode: str):
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if mode == "queries":
                fout.write(convert_queries_line(line) + "\n")
            elif mode == "response":
                fout.write(convert_response_line(line) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["queries", "response"])
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    convert_file(args.input, args.output, args.mode)
    print(f"Converted {args.mode}: {args.input} -> {args.output}")
