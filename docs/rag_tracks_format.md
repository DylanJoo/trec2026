# TREC RAG-style track integration guide

This repo currently uses:
- **Corpus**: JSONL, one doc per line: `{"docid": "...", "contents": "..."}`
- **Queries**: TSV: `qid<TAB>query_text`
- **Retriever result**: TREC run: `qid Q0 docid rank score run_id`
- **Generation output**: JSONL: `{"qid": "...", "response": "..."}`

## Recommended normalized format

To run multiple tracks consistently, keep the internal format above and add a normalized submission format for citation-based tracks:

```json
{
  "metadata": {
    "team_id": "team",
    "run_id": "run",
    "topic_id": "topic"
  },
  "responses": [
    {"text": "sentence", "citations": ["doc1", "doc2"]}
  ],
  "references": ["doc1", "doc2", "doc3"]
}
```

This is aligned with `rag-run-validator` default schema and is accepted across RAG-related tracks as a common structure.

## Track-by-track format summary

## 1) TREC RAG (2024/2025)

- **Corpus**: MS MARCO v2.1 document + segmented corpus (JSONL shards in tar archives).
- **Topics**: JSONL (e.g., `trec_rag_2025_queries.jsonl`) with fields like `id`, `title` (narrative query).
- **Retrieval submission**: standard 6-column TREC run file over segment IDs.
- **AG/RAG submission**: JSONL with metadata + references + sentence-level citations.
- **Evaluation signals**:
  - Retrieval uses qrels + `trec_eval` style measures (`ndcg_cut`, `recall` from released guidance/posts).
  - AG/RAG are citation-grounded long-form response evaluations.

## 2) TREC RAGTIME (2025/2026)

- **Corpus**: multilingual news/report collection (2026 release available as `trec-ragtime/ragtime2` on Hugging Face).
- **Topics**: distributed via TREC active participants site.
- **Submission**: citation-based report generation + retrieval tasks; organizers provide API/search support.
- **Evaluation**: citation-based report quality and retrieval subtasks.

## 3) TREC BioGen (2025)

- **Corpus**: processed PubMed collection (`biogen-2025-document-collection.zip`).
- **Task A input**: question-answer pairs with existing supporting PMIDs.
- **Task B input**: biomedical topic/question.
- **Submission format**:
  - Task A: JSONL with `metadata`, per-sentence `supported_citations` and `contradicted_citations`.
  - Task B: JSONL with `metadata`, `responses[{text, citations}]`.
- **Evaluation**: PMID-grounding/reference attribution metrics (Task B follows BioGen 2024-style evaluation).

## 4) TREC AutoJudge

- **Purpose**: benchmark LLM-as-a-judge systems across RAG-related tasks.
- **Core data objects** (`autojudge-base`):
  - `Report` (RAG system output)
  - `Request` (topic/query)
  - `Leaderboard` / `Qrels` / `NuggetBanks`
- **Evaluation tooling** (`auto-judge-evaluate`): leaderboard correlation, qrel agreement, format conversion.
- **Note**: official yearly datasets are distributed separately; starter kit ships synthetic `kiddie` data.

## Normalization workflow in this repo

Use the converter script:

```bash
# queries/topics -> queries.tsv
python scripts/normalize_track_data.py queries --track rag --input <topics> --output data/queries.tsv

# corpus shards -> corpus.jsonl
python scripts/normalize_track_data.py corpus --input <corpus_dir_or_file> --output data/corpus.jsonl

# run files -> TREC run
python scripts/normalize_track_data.py run --input <run_input> --output runs/bm25.trec
```

Supported inputs include JSONL/JSON(.gz), TSV/TXT queries, and TREC/jsonl-style retrieval outputs.

## Dataset download script

```bash
bash scripts/download_rag_tracks_data.sh [output_dir]
```

Default output directory: `data/external`.

The script downloads public resources for RAG, RAGTIME, BioGen, AutoJudge, and a shared run validator, then writes `README.restricted.txt` for files that must still be obtained through TREC participant portals.
