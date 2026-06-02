"""
Convert TREC RAGTIME 2025 / DRAGUN 2025 native formats to unified pipeline format.

RAGTIME corpus (multilingual news JSONL, AR/ZH/EN/RU):
  {"docid": "...", "title": "...", "text": "...", "lang": "ar"|"zh"|"en"|"ru", "date": "..."}

RAGTIME topics (JSONL):
  {"topic_id": "...", "problem_statement": "...", "background": "...",
   "report_length": 2000 | 10000}

DRAGUN topics (JSONL):
  {"topic_id": "...", "article_title": "...", "article_text": "...", "source": "..."}

RAGTIME/DRAGUN response (JSONL):
  {"run_id": "...", "topic_id": "...",
   "report": "Full report text with inline [doc_id] citations.",
   "citations": ["docid1", ...]}
"""

import json
import re


def convert_corpus_line(line: str) -> str:
    doc = json.loads(line)
    unified = {
        "docid": doc["docid"],
        "title": doc.get("title", ""),
        "text": doc.get("text", ""),
        "url": doc.get("url", ""),
        "meta": {
            "lang": doc.get("lang", ""),
            "date": doc.get("date", ""),
        },
    }
    return json.dumps(unified)


def convert_queries_line(line: str, track: str = "ragtime") -> str:
    q = json.loads(line)
    if track == "dragun":
        # DRAGUN: news article as the topic
        unified = {
            "qid": str(q["topic_id"]),
            "query": q.get("article_title", "") + " " + q.get("article_text", ""),
            "meta": {
                "article_title": q.get("article_title", ""),
                "source": q.get("source", ""),
                "track": "dragun",
            },
        }
    else:
        # RAGTIME
        unified = {
            "qid": str(q["topic_id"]),
            "query": q.get("problem_statement", ""),
            "meta": {
                "background": q.get("background", ""),
                "report_length": q.get("report_length", 2000),
                "track": "ragtime",
            },
        }
    return json.dumps(unified)


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    report = r.get("report", "")
    # Extract inline citations like [docid] from report text
    inline_citations = re.findall(r'\[([^\]]+)\]', report)
    explicit_citations = r.get("citations", [])
    all_citations = list(dict.fromkeys(inline_citations + explicit_citations))

    unified = {
        "qid": str(r["topic_id"]),
        "response": report,
        "sentences": [{"text": report, "citations": all_citations}],
    }
    return json.dumps(unified)


def convert_file(input_path: str, output_path: str, mode: str, track: str = "ragtime"):
    if mode == "corpus":
        converter = convert_corpus_line
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(converter(line) + "\n")
    elif mode == "queries":
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(convert_queries_line(line, track=track) + "\n")
    elif mode == "response":
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(convert_response_line(line) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["corpus", "queries", "response"])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--track", choices=["ragtime", "dragun"], default="ragtime")
    args = p.parse_args()
    convert_file(args.input, args.output, args.mode, track=args.track)
    print(f"Converted {args.mode}: {args.input} -> {args.output}")
