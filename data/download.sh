#!/bin/bash
# Download data for TREC RAG 2024, BioGen 2025, and RAGTIME 2025.
#
# Usage:
#   bash data/download.sh rag2024    # MS MARCO v2.1 corpus + RAG 2024 topics/qrels
#   bash data/download.sh biogen     # PubMed corpus + BioGen 2025 topics
#   bash data/download.sh ragtime    # Multilingual news corpus + RAGTIME 2025 topics
#   bash data/download.sh all        # all three tracks
#
# Requirements: curl, python3, pip install ir_datasets huggingface_hub
# See data/README.md for format details and manual-download notes.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
TRACK="${1:-all}"

fetch() {
    local url="$1" out="$2"
    if [[ -f "$out" ]]; then
        echo "  [skip] $(basename "$out") already exists"
        return
    fi
    echo "  [download] $url"
    curl -fsSL --retry 3 --retry-delay 5 -o "$out" "$url"
}

# =============================================================================
# TREC RAG 2024
# Corpus:  MS MARCO v2.1 segmented — 113M segments, ~84 GB (70 tar archives)
#          Recommended: HuggingFace Cohere mirror (no Azure account needed)
#          Azure raw: https://msmarco.blob.core.windows.net/msmarcoranking/
# Topics:  301 topics; dev set is public, test set requires TREC registration
# Qrels:   UMBRELA judgments, publicly available via anserini-tools
# =============================================================================
download_rag2024() {
    local outdir="$DATA_DIR/rag2024"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs" "$outdir/qrels"

    echo "=== TREC RAG 2024 ==="

    # Topics (dev set) — publicly available from anserini-tools
    echo "[topics] Downloading RAG 2024 dev topics (anserini-tools)..."
    fetch \
        "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.rag24.raggy-dev.txt" \
        "$outdir/topics/queries.tsv"

    # Qrels — UMBRELA judgments
    echo "[qrels] Downloading UMBRELA qrels..."
    fetch \
        "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.rag24.test-umbrela-all.txt" \
        "$outdir/qrels/qrels.rag24.test-umbrela-all.txt"

    # Topics via ir_datasets (JSONL format — preferred for preprocess.py)
    echo "[topics] Attempting ir_datasets download (trec-rag-2024)..."
    python3 - "$outdir" <<'PYEOF'
import sys, json
outdir = sys.argv[1]
try:
    import ir_datasets
    ds = ir_datasets.load("msmarco-passage-v2/trec-rag-2024")
    with open(f"{outdir}/topics/queries.jsonl", "w") as f:
        for q in ds.queries_iter():
            f.write(json.dumps({"topic_id": q.query_id, "topic": q.text}) + "\n")
    n = sum(1 for _ in open(f"{outdir}/topics/queries.jsonl"))
    print(f"  Wrote {n} topics to {outdir}/topics/queries.jsonl")
except Exception as e:
    print(f"  [warn] ir_datasets failed: {e}")
    print(f"  [info] Falling back to TSV topics at {outdir}/topics/queries.tsv")
PYEOF

    # Corpus — instruct user (too large for automated download by default)
    echo ""
    echo "[corpus] MS MARCO v2.1 segmented is ~84 GB. Choose one option:"
    echo ""
    echo "  Option A — HuggingFace (Cohere mirror, recommended):"
    echo "    pip install huggingface_hub"
    echo "    huggingface-cli download Cohere/trec-rag-2024-index \\"
    echo "      --repo-type dataset \\"
    echo "      --local-dir $outdir/corpus \\"
    echo "      --ignore-patterns '*.parquet'"
    echo ""
    echo "  Option B — Azure blob storage (raw tar archives, ~1.2 GB each):"
    echo "    for i in \$(seq -w 00 69); do"
    echo "      curl -O https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented_\${i}.tar.gz"
    echo "      tar -xzf msmarco_v2.1_doc_segmented_\${i}.tar.gz -C $outdir/corpus/"
    echo "    done"
    echo ""
    echo "  Option C — Pyserini prebuilt dense index (for retrieval only):"
    echo "    python3 -c \"from pyserini.search.faiss import FaissSearcher; FaissSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented.bge-base-en-v1.5', None)\""
    echo ""
    echo "[done] RAG 2024 → $outdir"
}

# =============================================================================
# TREC BioGen 2025
# Corpus:  PubMed/MEDLINE abstracts — ~27M documents
#          Official zip from NLM; also available via NCBI FTP
# Topics:  Distributed to TREC registered participants only
#          Sample topics in starter kit: https://github.com/trec-biogen/starter-kit-2025
# =============================================================================
download_biogen() {
    local outdir="$DATA_DIR/biogen"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs"

    echo "=== TREC BioGen 2025 ==="

    # Corpus — official NLM package
    echo "[corpus] Downloading BioGen 2025 document collection..."
    local corpus_zip="$outdir/corpus/biogen-2025-document-collection.zip"
    fetch "http://bionlp.nlm.nih.gov/biogen-2025-document-collection.zip" "$corpus_zip" || {
        echo "  [warn] Direct download failed. Manual steps:"
        echo "    1. Visit https://trec-biogen.github.io/docs/"
        echo "    2. Download biogen-2025-document-collection.zip"
        echo "    3. Unzip to $outdir/corpus/"
    }
    if [[ -f "$corpus_zip" ]]; then
        echo "  [extract] Unzipping corpus..."
        unzip -q -n "$corpus_zip" -d "$outdir/corpus/"
        echo "  [done] Corpus extracted to $outdir/corpus/"
    fi

    # Topics via ir_datasets
    echo "[topics] Attempting ir_datasets download (trec-biogen/2025)..."
    python3 - "$outdir" <<'PYEOF'
import sys, json
outdir = sys.argv[1]
try:
    import ir_datasets
    ds = ir_datasets.load("trec-biogen/2025")
    with open(f"{outdir}/topics/queries.jsonl", "w") as f:
        for q in ds.queries_iter():
            f.write(json.dumps({"qa_id": q.query_id, "question": q.text}) + "\n")
    n = sum(1 for _ in open(f"{outdir}/topics/queries.jsonl"))
    print(f"  Wrote {n} topics to {outdir}/topics/queries.jsonl")
except Exception as e:
    print(f"  [warn] ir_datasets failed ({e})")
    # Fall back to starter kit sample topics
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/trec-biogen/starter-kit-2025/main/data/sample_topics.jsonl"
        urllib.request.urlretrieve(url, f"{outdir}/topics/queries.jsonl")
        print(f"  Downloaded sample topics from starter kit.")
    except Exception as e2:
        print(f"  [warn] Sample topics also failed: {e2}")
        print(f"  [info] Download topics manually from: https://trec.nist.gov/act_part/")
        print(f"         Place as: {outdir}/topics/queries.jsonl")
        print(f"         Format:  {{\"qa_id\": \"...\", \"question\": \"...\"}}")
PYEOF

    echo "[done] BioGen 2025 → $outdir"
}

# =============================================================================
# TREC RAGTIME 2025
# Corpus:  ~4M multilingual news docs (AR/ZH/EN/RU)
#          Available on HuggingFace: trec-ragtime/ragtime2
#          Official release via NIST (registered participants)
# Topics:  Distributed to TREC registered participants only
#          Registration: https://trec.nist.gov/act_part/tracks25.html
# =============================================================================
download_ragtime() {
    local outdir="$DATA_DIR/ragtime"
    mkdir -p "$outdir/corpus" "$outdir/topics" "$outdir/runs" "$outdir/qrels"

    echo "=== TREC RAGTIME 2025 ==="

    # Corpus — HuggingFace (public)
    echo "[corpus] Downloading RAGTIME corpus from HuggingFace (trec-ragtime/ragtime2)..."
    python3 - "$outdir" <<'PYEOF'
import sys
outdir = sys.argv[1]
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="trec-ragtime/ragtime2",
        repo_type="dataset",
        local_dir=f"{outdir}/corpus",
        local_dir_use_symlinks=False,
    )
    print(f"  Corpus downloaded to {outdir}/corpus/")
except ImportError:
    print("  [warn] huggingface_hub not installed. Run: pip install huggingface_hub")
    print("  Then: huggingface-cli download trec-ragtime/ragtime2 \\")
    print(f"          --repo-type dataset --local-dir {outdir}/corpus")
except Exception as e:
    print(f"  [warn] HuggingFace download failed: {e}")
    print("  [info] Manual option — NIST registered participants:")
    print("    1. Register at https://trec.nist.gov/act_part/tracks25.html")
    print("    2. Download corpus from the NIST secure portal")
    print(f"    3. Place JSONL files in {outdir}/corpus/")
    print("    Format: {\"docid\": \"...\", \"title\": \"...\", \"text\": \"...\", \"lang\": \"ar|zh|en|ru\", \"date\": \"...\"}")
PYEOF

    # Topics — restricted (participants only)
    echo ""
    echo "[topics] RAGTIME 2025 topics require NIST participant registration."
    echo "  1. Register at: https://trec.nist.gov/act_part/tracks25.html"
    echo "  2. Download official topics from the NIST secure portal"
    echo "  3. Place as: $outdir/topics/queries.jsonl"
    echo "     Format: {\"topic_id\": \"...\", \"problem_statement\": \"...\", \"background\": \"...\", \"report_length\": 2000}"
    echo ""
    echo "  Track homepage: https://trec-ragtime.github.io/"
    echo ""
    echo "[done] RAGTIME 2025 → $outdir"
}

# =============================================================================
# MAIN
# =============================================================================

case "$TRACK" in
    rag2024)  download_rag2024 ;;
    biogen)   download_biogen ;;
    ragtime)  download_ragtime ;;
    all)
        download_rag2024
        echo ""
        download_biogen
        echo ""
        download_ragtime
        ;;
    *)
        echo "Unknown track: $TRACK"
        echo "Usage: $0 [rag2024|biogen|ragtime|all]"
        exit 1
        ;;
esac

echo ""
echo "Next step: python data/preprocess.py $TRACK"
