"""
Convert TREC iKAT 2024 native formats to unified pipeline format.

Native corpus passage (JSONL):
  {"id": "clueweb22-en0004-50-00170:0", "contents": "...", "url": "..."}

Native topics (JSON, multi-turn conversations):
  [{"topic_id": "1", "title": "...",
    "ptkb": {"1": "statement1", "2": "statement2"},
    "turns": [
      {"turn_id": "1_1", "utterance": "...", "response": "...",
       "ptkb_provenance": ["1", "2"], ...}
    ]}]

Native run (JSONL, one object per turn):
  {"turn_id": "1_1",
   "responses": [{"rank": 1, "text": "..."}],
   "references": ["passage_id", ...]}
"""

import json


def convert_corpus_line(line: str) -> str:
    doc = json.loads(line)
    unified = {
        "docid": doc["id"],
        "title": "",
        "text": doc.get("contents", ""),
        "url": doc.get("url", ""),
        "meta": {},
    }
    return json.dumps(unified)


def convert_topics_file(input_path: str, output_path: str):
    """Flatten multi-turn JSON topics into per-turn JSONL queries."""
    with open(input_path) as fin:
        topics = json.load(fin)

    with open(output_path, "w") as fout:
        for topic in topics:
            ptkb = topic.get("ptkb", {})
            ptkb_text = " | ".join(ptkb.values())
            for turn in topic.get("turns", []):
                utterance = turn["utterance"]
                # Prepend PTKB context into query meta
                unified = {
                    "qid": turn["turn_id"],
                    "query": utterance,
                    "meta": {
                        "topic_id": topic["topic_id"],
                        "ptkb": ptkb,
                        "ptkb_text": ptkb_text,
                        "conversation_history": [],  # populated below
                    },
                }
                fout.write(json.dumps(unified) + "\n")


def convert_response_line(line: str) -> str:
    r = json.loads(line)
    responses = r.get("responses", [])
    top_response = responses[0]["text"] if responses else ""
    unified = {
        "qid": r["turn_id"],
        "response": top_response,
        "sentences": [{"text": top_response, "citations": r.get("references", [])}],
    }
    return json.dumps(unified)


def convert_file(input_path: str, output_path: str, mode: str):
    if mode == "corpus":
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(convert_corpus_line(line) + "\n")
    elif mode == "queries":
        convert_topics_file(input_path, output_path)
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
    args = p.parse_args()
    convert_file(args.input, args.output, args.mode)
    print(f"Converted {args.mode}: {args.input} -> {args.output}")
