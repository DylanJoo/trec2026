#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/external}"
mkdir -p "$OUT_DIR"/{rag,ragtime,biogen,autojudge,shared}

fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return
  fi
  echo "[download] $url"
  curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
}

echo "== TREC RAG =="
fetch "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc.tar" \
  "$OUT_DIR/rag/msmarco_v2.1_doc.tar"
fetch "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar" \
  "$OUT_DIR/rag/msmarco_v2.1_doc_segmented.tar"
fetch "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.rag24.test.txt" \
  "$OUT_DIR/rag/topics.rag24.test.txt"
fetch "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.rag24.test-umbrela-all.txt" \
  "$OUT_DIR/rag/qrels.rag24.test-umbrela-all.txt"

echo "== TREC RAGTIME =="
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download trec-ragtime/ragtime2 \
    --repo-type dataset \
    --local-dir "$OUT_DIR/ragtime/ragtime2" \
    --local-dir-use-symlinks False
else
  echo "[warn] huggingface-cli not found; install with: pip install -U huggingface_hub"
  echo "[warn] then run: huggingface-cli download trec-ragtime/ragtime2 --repo-type dataset --local-dir $OUT_DIR/ragtime/ragtime2"
fi

echo "== TREC BioGen =="
fetch "http://bionlp.nlm.nih.gov/biogen-2025-document-collection.zip" \
  "$OUT_DIR/biogen/biogen-2025-document-collection.zip"
fetch "https://codeload.github.com/trec-biogen/starter-kit-2025/zip/refs/heads/main" \
  "$OUT_DIR/biogen/starter-kit-2025.zip"

echo "== TREC AutoJudge =="
fetch "https://codeload.github.com/trec-auto-judge/auto-judge-starter-kit/zip/refs/heads/main" \
  "$OUT_DIR/autojudge/auto-judge-starter-kit.zip"
fetch "https://codeload.github.com/trec-auto-judge/auto-judge-base/zip/refs/heads/main" \
  "$OUT_DIR/autojudge/auto-judge-base.zip"
fetch "https://codeload.github.com/trec-auto-judge/auto-judge-evaluate/zip/refs/heads/main" \
  "$OUT_DIR/autojudge/auto-judge-evaluate.zip"

echo "== Shared validator and tooling =="
fetch "https://codeload.github.com/hltcoe/rag-run-validator/zip/refs/heads/main" \
  "$OUT_DIR/shared/rag-run-validator.zip"

cat > "$OUT_DIR/README.restricted.txt" <<'TXT'
Some required files are not publicly downloadable and must be obtained from TREC Active Participants portals:
- TREC RAG: official 2025/2026 topic JSONL, AG/RAG candidate sets, official judgments
- TREC RAGTIME: official topic files and submission portal artifacts
- TREC BioGen: official Task A/Task B topic files from TREC active participants site
- TREC AutoJudge: official benchmark datasets/judgments for the active year
TXT

echo "Done. Data root: $OUT_DIR"
