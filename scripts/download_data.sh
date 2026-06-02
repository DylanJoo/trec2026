#!/bin/bash
# =============================================================================
# Download datasets for all supported TREC RAG-style tracks.
#
# Usage:
#   bash scripts/download_data.sh [TRACK]
#
# TRACK options:
#   rag2024     TREC RAG 2024  (MS MARCO v2.1 segmented + topics)
#   rag2025     TREC RAG 2025  (same corpus, new topics)
#   ragtime     TREC RAGTIME 2025  (multilingual news corpus + topics)
#   dragun      TREC DRAGUN 2025  (MS MARCO v2.1 + news article topics)
#   biogen      TREC BioGen 2024/2025  (PubMed abstracts)
#   ikat        TREC iKAT 2024  (ClueWeb22-B — requires license)
#   neuclir     TREC NeuCLIR 2024  (multilingual news via ir_datasets)
#   all         Download everything (except ikat which requires manual license)
#
# Requirements: python, pip install ir_datasets pyserini, huggingface-cli
# =============================================================================

set -e

DATA_DIR="${DATA_DIR:-$(dirname "$0")/../data}"
TRACK="${1:-all}"

mkdir -p "$DATA_DIR"

# =============================================================================
# TREC RAG 2024 / 2025
# Corpus: MS MARCO v2.1 segmented (113M segments, ~84GB uncompressed)
# Hosted by Microsoft; easiest access via HuggingFace or Pyserini prebuilt index
# Topics: trec-rag.github.io (released via participants mailing list)
# =============================================================================
download_rag() {
    local year="${1:-2024}"
    local outdir="$DATA_DIR/rag${year}"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC RAG ${year} ==="

    # Corpus: download via HuggingFace (Cohere mirror, segmented with embeddings stripped)
    # Full raw corpus from Microsoft: https://msmarco.blob.core.windows.net/msmarcoranking/
    echo "[corpus] Downloading MS MARCO v2.1 segmented via HuggingFace..."
    python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Cohere/trec-rag-2024-index",
    repo_type="dataset",
    local_dir="${outdir}/corpus",
    ignore_patterns=["*.parquet"],   # skip embedding files to save space
)
PYEOF

    # Topics (JSONL) — publicly available at trec-rag.github.io
    # RAG 2024: 301 test topics (DL 2021-2023 + new topics)
    # RAG 2025: narrative-style topics
    echo "[topics] Downloading RAG ${year} topics..."
    if [ "$year" = "2024" ]; then
        python -c "
import ir_datasets, json
# trec-rag 2024 test topics via ir_datasets
try:
    ds = ir_datasets.load('msmarco-passage-v2/trec-rag-2024')
    with open('${outdir}/topics/queries.jsonl', 'w') as f:
        for q in ds.queries_iter():
            f.write(json.dumps({'topic_id': q.query_id, 'topic': q.text}) + '\n')
    print(f'Wrote {sum(1 for _ in open(\"${outdir}/topics/queries.jsonl\"))} topics')
except Exception as e:
    print(f'ir_datasets not available for rag-2024, download manually: {e}')
"
    fi
    echo "[done] RAG ${year} data in ${outdir}"
    echo "  NOTE: Convert with: python scripts/converters/rag2024.py queries ${outdir}/topics/queries.jsonl ${outdir}/topics/queries_unified.jsonl"
    echo "  NOTE: Full corpus JSONL from Microsoft: https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented_*.tar.gz"
}

# =============================================================================
# TREC RAGTIME 2025
# Corpus: multilingual news (AR/ZH/EN/RU), ~4M docs
# Released via NIST to registered participants
# Topics: JSONL (problem_statement + background, two lengths)
# =============================================================================
download_ragtime() {
    local outdir="$DATA_DIR/ragtime"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC RAGTIME 2025 ==="
    echo "[corpus] RAGTIME corpus requires NIST registration."
    echo "  1. Register at: https://trec.nist.gov/act_part/tracks25.html"
    echo "  2. Download corpus from NIST secure site once registered."
    echo "  3. Place corpus JSONL files in: ${outdir}/corpus/"
    echo ""
    echo "[topics] Downloading RAGTIME 2025 topics..."
    python -c "
import ir_datasets, json
try:
    # RAGTIME topics may be available via ir_datasets after track releases
    ds = ir_datasets.load('trec-ragtime/2025')
    with open('${outdir}/topics/queries.jsonl', 'w') as f:
        for q in ds.queries_iter():
            f.write(json.dumps({'topic_id': q.query_id, 'topic': q.text}) + '\n')
    print('Topics downloaded.')
except Exception as e:
    print(f'Not yet in ir_datasets: {e}')
    print('Download topics from: https://trec-ragtime.github.io/')
"
    echo "[done] RAGTIME data dir: ${outdir}"
}

# =============================================================================
# TREC DRAGUN 2025
# Corpus: MS MARCO v2.1 segmented (same as RAG 2024)
# Topics: 30 news articles (JSONL)
# =============================================================================
download_dragun() {
    local outdir="$DATA_DIR/dragun"
    mkdir -p "$outdir/topics" "$outdir/runs"

    echo "=== TREC DRAGUN 2025 ==="
    echo "[corpus] DRAGUN uses MS MARCO v2.1 segmented — run 'download_data.sh rag2024' for corpus."
    echo "[topics] Downloading DRAGUN 2025 topics..."
    python -c "
import urllib.request, json, os
# Topics file from DRAGUN track page
url = 'https://trec-dragun.github.io/trec-2025-dragun-topics.jsonl'
try:
    urllib.request.urlretrieve(url, '${outdir}/topics/queries.jsonl')
    n = sum(1 for _ in open('${outdir}/topics/queries.jsonl'))
    print(f'Downloaded {n} topics.')
except Exception as e:
    print(f'Failed: {e}')
    print('Download manually from: https://trec-dragun.github.io/')
"
    echo "[done] DRAGUN data dir: ${outdir}"
    echo "  NOTE: Convert with: python scripts/converters/ragtime.py --track dragun queries ${outdir}/topics/queries.jsonl ${outdir}/topics/queries_unified.jsonl"
}

# =============================================================================
# TREC BioGen 2024 / 2025
# Corpus: PubMed/MEDLINE ~27M abstracts
# Available via PySerini starter kit
# =============================================================================
download_biogen() {
    local year="${1:-2025}"
    local outdir="$DATA_DIR/biogen${year}"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC BioGen ${year} ==="

    # Corpus via PySerini prebuilt index
    echo "[corpus] Downloading PubMed index via PySerini..."
    python -c "
from pyserini.search.lucene import LuceneSearcher
try:
    searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')
    print('PySerini BioASQ index downloaded.')
except Exception as e:
    print(f'Failed: {e}')
"
    echo "[corpus] For full PubMed ${year} corpus (JSONL), download from BioGen starter kit:"
    echo "  https://trec-biogen.github.io/docs/"
    echo "  Place JSONL files in: ${outdir}/corpus/"

    # Topics
    echo "[topics] Downloading BioGen ${year} topics..."
    python -c "
import ir_datasets, json
try:
    tag = 'trec-biogen/2024' if '${year}' == '2024' else 'trec-biogen/2025'
    ds = ir_datasets.load(tag)
    with open('${outdir}/topics/queries.jsonl', 'w') as f:
        for q in ds.queries_iter():
            f.write(json.dumps({'qa_id': q.query_id, 'question': q.text}) + '\n')
    print('Topics downloaded via ir_datasets.')
except Exception as e:
    print(f'Not yet in ir_datasets: {e}')
    print('Download from: https://trec-biogen.github.io/docs/')
"
    echo "[done] BioGen data dir: ${outdir}"
    echo "  NOTE: Convert with: python scripts/converters/biogen.py queries ${outdir}/topics/queries.jsonl ${outdir}/topics/queries_unified.jsonl"
}

# =============================================================================
# TREC iKAT 2024
# Corpus: ClueWeb22-B subset (~117M passages)
# REQUIRES manual license from CMU — cannot be auto-downloaded
# Topics: JSON conversation trees with PTKB
# =============================================================================
download_ikat() {
    local outdir="$DATA_DIR/ikat"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC iKAT 2024 ==="
    echo "[corpus] ClueWeb22-B requires a data license from CMU:"
    echo "  1. Sign license at: https://lemurproject.org/clueweb22/"
    echo "  2. Email signed form to: jlm4@andrew.cmu.edu"
    echo "  3. You will receive a download link."
    echo "  4. Place passage JSONL files in: ${outdir}/corpus/"
    echo ""
    echo "[topics] Downloading iKAT 2024 topics from TREC..."
    python -c "
import urllib.request, json
url = 'https://trec.nist.gov/data/ikat/2024-qrels.txt'
try:
    # Topics JSON
    topics_url = 'https://raw.githubusercontent.com/irlabamsterdam/iKAT/main/2024/data/2024_test_topics.json'
    urllib.request.urlretrieve(topics_url, '${outdir}/topics/topics.json')
    print('Downloaded iKAT 2024 topics.')
except Exception as e:
    print(f'Failed: {e}')
    print('Download from: https://www.trecikat.com/data/')
"
    echo "[done] iKAT data dir: ${outdir}"
    echo "  NOTE: Convert with: python scripts/converters/ikat.py queries ${outdir}/topics/topics.json ${outdir}/topics/queries_unified.jsonl"
}

# =============================================================================
# TREC NeuCLIR 2024
# Corpus: ~10M news docs (ZH/FA/RU) via ir_datasets
# Topics: English queries
# =============================================================================
download_neuclir() {
    local outdir="$DATA_DIR/neuclir"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC NeuCLIR 2024 ==="

    python - <<PYEOF
import ir_datasets, json, os

langs = [
    ("neuclir/1/zh/trec-2024", "zh"),
    ("neuclir/1/fa/trec-2024", "fa"),
    ("neuclir/1/ru/trec-2024", "ru"),
]

for ds_name, lang in langs:
    print(f"Loading {ds_name} ...")
    try:
        ds = ir_datasets.load(ds_name)

        # Corpus
        corpus_path = os.path.join("${outdir}/corpus", f"corpus_{lang}.jsonl")
        with open(corpus_path, "w") as f:
            for doc in ds.docs_iter():
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "title": getattr(doc, "title", ""),
                    "text": doc.text,
                    "url": getattr(doc, "url", ""),
                    "time": getattr(doc, "time", ""),
                    "lang": lang,
                }) + "\n")
        print(f"  Corpus written to {corpus_path}")

        # Topics (same English queries across all languages)
        queries_path = os.path.join("${outdir}/topics", f"queries_{lang}.jsonl")
        with open(queries_path, "w") as f:
            for q in ds.queries_iter():
                f.write(json.dumps({
                    "query_id": q.query_id,
                    "title": q.title,
                    "description": getattr(q, "description", ""),
                    "lang": lang,
                }) + "\n")
        print(f"  Queries written to {queries_path}")

        # Qrels
        qrels_path = os.path.join("${outdir}", f"qrels_{lang}.txt")
        with open(qrels_path, "w") as f:
            for qrel in ds.qrels_iter():
                f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
        print(f"  Qrels written to {qrels_path}")

    except Exception as e:
        print(f"  Failed: {e}")
PYEOF

    echo "[done] NeuCLIR data dir: ${outdir}"
    echo "  NOTE: Convert with: python scripts/converters/neuclir.py corpus ${outdir}/corpus/corpus_zh.jsonl ${outdir}/corpus/corpus_zh_unified.jsonl"
}

# =============================================================================
# MAIN
# =============================================================================

case "$TRACK" in
    rag2024)   download_rag 2024 ;;
    rag2025)   download_rag 2025 ;;
    ragtime)   download_ragtime ;;
    dragun)    download_dragun ;;
    biogen)    download_biogen 2025 ;;
    biogen2024) download_biogen 2024 ;;
    ikat)      download_ikat ;;
    neuclir)   download_neuclir ;;
    all)
        echo "Downloading all tracks (skipping ikat — requires manual license)"
        download_rag 2024
        download_rag 2025
        download_ragtime
        download_dragun
        download_biogen 2025
        download_neuclir
        echo ""
        echo "=== iKAT requires manual license — run: bash scripts/download_data.sh ikat ==="
        ;;
    *)
        echo "Unknown track: $TRACK"
        echo "Usage: $0 [rag2024|rag2025|ragtime|dragun|biogen|ikat|neuclir|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Data directory layout ==="
find "$DATA_DIR" -maxdepth 3 -type d | sort
