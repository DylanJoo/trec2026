"""
Convert TREC BioGen 2024/2025 native formats to unified pipeline format.

Native corpus (JSONL from PySerini processed PubMed):
  {"pmid": "12345678", "title": "...", "abstract": "..."}

Native queries (JSONL or TSV):
  {"qa_id": "BG2024_001", "question": "..."}

Native generation run (JSONL):
  {"run_id": "...", "qa_id": "...", "question": "...",
   "answer": [{"text": "sentence.", "citations": ["pmid1", "pmid2", ...]}, ...]}
"""

import json


def convert_corpus_line(line: str) -> str:
    doc = json.loads(line)
    pmid = str(doc.get("pmid", doc.get("id", "")))
    unified = {
        "docid": pmid,
        "title": doc.get("title", ""),
        "text": doc.get("abstract", doc.get("text", "")),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "meta": {"pmid": pmid},
    }
    return json.dumps(unified)


def convert_queries_line(line: str) -> str:
    q = json.loads(line)
    unified = {
        "qid": str(q.get("qa_id", q.get("id", ""))),
        "query": q.get("question", q.get("query", "")),
        "meta": {},
    }
    return json.dumps(unified)


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    sentences = [
        {"text": s["text"], "citations": s.get("citations", s.get("supported_citations", []))}
        for s in r.get("answer", [])
    ]
    unified = {
        "qid": str(r.get("qa_id", r.get("id", ""))),
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
