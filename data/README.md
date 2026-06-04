# Data

This directory contains data for three TREC tracks supported by this pipeline:
**TREC RAG 2024**, **TREC BioGen 2025**, and **TREC RAGTIME 2025**.

---

## Tracks

### TREC RAG 2024

**Task**: Given a narrative query and a set of retrieved MS MARCO v2.1 document segments, generate a grounded answer (≤400 words) with sentence-level inline citations (e.g. `[1][3]`).

**Corpus**: MS MARCO v2.1 segmented — 113 million passage segments (~200–300 tokens each) from 11 million web documents. Raw size ~84 GB uncompressed across 70 tar archives.

**Topics**: 301 test topics drawn from TREC Deep Learning 2021–2023 plus new queries. Dev set (raggy-dev) is publicly available; test topics require TREC participant registration.

**Evaluation**: UMBRELA-based citation quality; nugget recall.

| Resource | URL |
|---|---|
| Corpus — Azure (raw, 70 × ~1.2 GB tar) | `https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented_XX.tar.gz` |
| Corpus — HuggingFace (Cohere mirror) | `https://huggingface.co/datasets/Cohere/trec-rag-2024-index` |
| Topics — dev (anserini-tools) | `https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.rag24.raggy-dev.txt` |
| Topics — test | TREC active participants: `https://trec.nist.gov/act_part/` |
| Qrels — UMBRELA | `https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.rag24.test-umbrela-all.txt` |
| Track homepage | `https://trec-rag.github.io/` |
| Track repository | `https://github.com/TREC-RAG/TREC-RAG-2024` |
| ir\_datasets id | `msmarco-passage-v2/trec-rag-2024` |

**Native corpus format** (JSONL, one segment per line):
```json
{"docid": "msmarco_v2.1_doc_00_0#0_0", "url": "https://...", "title": "...",
 "headings": "...", "segment": "passage text", "start_char": 0, "end_char": 312}
```

**Native topics format** — TSV (anserini-tools) or JSONL (ir\_datasets):
```
2027497	what are the side effects of taking tylenol
```
```json
{"topic_id": "2027497", "topic": "what are the side effects of taking tylenol"}
```

---

### TREC BioGen 2025

**Task**: Biomedical report generation. Given a biomedical question, generate a paragraph-length answer with sentence-level citations to PubMed abstracts (PMIDs).
- **Task A**: verify/extend an existing answer (citation attribution).
- **Task B**: generate from scratch (open-ended RAG).

**Corpus**: PubMed/MEDLINE abstracts — ~27 million biomedical articles.

| Resource | URL |
|---|---|
| Corpus (official, zip) | `http://bionlp.nlm.nih.gov/biogen-2025-document-collection.zip` |
| Corpus — NCBI PubMed FTP | `https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/` |
| Starter kit (2025) | `https://github.com/trec-biogen/starter-kit-2025` |
| Starter kit (2024 reference) | `https://github.com/trec-biogen/starter-kit-2024` |
| Topics | TREC active participants: `https://trec.nist.gov/act_part/` |
| Track homepage | `https://trec-biogen.github.io/` |
| ir\_datasets id | `trec-biogen/2024`, `trec-biogen/2025` |

> **Note**: The official test topic set is distributed to registered TREC participants only. Sample topics are available in the starter kit.

**Native corpus format** (JSONL):
```json
{"pmid": "12345678", "title": "Article title", "abstract": "Abstract text..."}
```

**Native topics format** (JSONL):
```json
{"qa_id": "BG2025_001", "question": "What is the mechanism of action of metformin in type 2 diabetes?"}
```

---

### TREC RAGTIME 2025

**Task**: Multilingual report generation. Given a problem statement and background, generate a structured report (~2000 or ~10000 characters) grounded in multilingual news documents (AR/ZH/EN/RU) with inline citations.

**Corpus**: ~4 million multilingual news documents across Arabic, Chinese, English, and Russian. Available on HuggingFace without registration.

| Resource | URL |
|---|---|
| Corpus — HuggingFace | `https://huggingface.co/datasets/trec-ragtime/ragtime2` |
| Corpus — NIST (registered participants) | `https://trec.nist.gov/act_part/tracks25.html` |
| Topics | TREC active participants: `https://trec.nist.gov/act_part/` |
| Track homepage | `https://trec-ragtime.github.io/` |
| Registration | `https://trec.nist.gov/act_part/tracks25.html` |

> **Note**: Official topics require NIST participant registration. The corpus is available on HuggingFace (`trec-ragtime/ragtime2`) without registration.

**Native corpus format** (JSONL):
```json
{"docid": "...", "title": "...", "text": "...", "lang": "ar", "date": "2024-01-15"}
```

**Native topics format** (JSONL):
```json
{"topic_id": "RT2025_001", "problem_statement": "Describe recent developments in...",
 "background": "Context...", "report_length": 2000}
```

---

## Unified Format

All track data is normalized to the same internal format so the pipeline handles any track uniformly.

### `corpus/corpus.jsonl`

One document per line:
```json
{"docid": "...", "title": "...", "text": "...", "url": "...", "meta": {}}
```

| Track | `docid` source | `text` source |
|---|---|---|
| RAG 2024 | `docid` | `segment` |
| BioGen 2025 | `pmid` | `abstract` |
| RAGTIME 2025 | `docid` | `text` |

### `topics/queries_unified.jsonl`

One query per line:
```json
{"qid": "...", "query": "...", "meta": {}}
```

Track-specific fields are stored under `meta`: RAGTIME includes `background` and `report_length`; BioGen stores the raw `qa_id`.

### `runs/run.txt`

Standard TREC 6-column retrieval run:
```
qid Q0 docid rank score run_tag
```

### `qrels/{qid}.txt` (RAGTIME, oracle mode)

Per-query nugget annotations used by the oracle context selector:
```
docid nugget_id relevance
```

---

## Directory Structure

After downloading and preprocessing, the expected layout is:

```
data/
├── download.sh               ← download script (bash)
├── preprocess.py             ← preprocessing script (python)
├── rag2024/
│   ├── corpus/
│   │   └── corpus.jsonl            ← unified (pipeline input)
│   ├── topics/
│   │   ├── queries.jsonl           ← raw (ir_datasets JSONL)
│   │   ├── queries.tsv             ← raw (anserini TSV, alternative)
│   │   └── queries_unified.jsonl   ← unified (pipeline input)
│   ├── runs/
│   │   └── run.txt
│   └── qrels/
│       └── qrels.rag24.test-umbrela-all.txt
├── biogen/
│   ├── corpus/
│   │   └── corpus.jsonl
│   ├── topics/
│   │   ├── queries.jsonl
│   │   └── queries_unified.jsonl
│   └── runs/
│       └── run.txt
└── ragtime/
    ├── corpus/
    │   └── corpus.jsonl
    ├── topics/
    │   ├── queries.jsonl
    │   └── queries_unified.jsonl
    ├── runs/
    │   └── run.txt
    └── qrels/
        └── {qid}.txt
```

---

## Getting Started

### 1. Download

```bash
bash data/download.sh rag2024    # RAG 2024: topics + qrels (corpus instructions printed)
bash data/download.sh biogen     # BioGen 2025: corpus + topics
bash data/download.sh ragtime    # RAGTIME 2025: corpus via HuggingFace (topics instructions printed)
bash data/download.sh all        # all three tracks
```

See [`download.sh`](download.sh) for access notes on restricted resources.

### 2. Preprocess

Convert each track's raw data to the unified pipeline format:

```bash
python data/preprocess.py rag2024
python data/preprocess.py biogen
python data/preprocess.py ragtime
python data/preprocess.py all
```

After this step each track directory contains `topics/queries_unified.jsonl` and `corpus/corpus.jsonl`.

### 3. Run the pipeline

```bash
python run_rag.py config/rag2024.yaml
python run_rag.py config/biogen.yaml
python run_rag.py config/ragtime.yaml
```

---

## Converter Reference

Low-level converters live in `scripts/converters/` and can be called directly:

```bash
# RAG 2024
python scripts/converters/rag2024.py queries   raw.jsonl  topics/queries_unified.jsonl
python scripts/converters/rag2024.py corpus    shard.jsonl corpus/corpus.jsonl

# BioGen
python scripts/converters/biogen.py  queries   raw.jsonl  topics/queries_unified.jsonl
python scripts/converters/biogen.py  corpus    raw.jsonl  corpus/corpus.jsonl

# RAGTIME
python scripts/converters/ragtime.py queries   raw.jsonl  topics/queries_unified.jsonl
python scripts/converters/ragtime.py corpus    raw.jsonl  corpus/corpus.jsonl

# Nugget annotations (RAGTIME qrels)
python scripts/converters/nuggets.py autonuggetizer  nuggets.jsonl   data/ragtime/qrels/
python scripts/converters/nuggets.py nist            nist_pool.txt   data/ragtime/qrels/
```
