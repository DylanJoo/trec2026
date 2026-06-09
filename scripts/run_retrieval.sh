#!/bin/bash -l
#SBATCH --job-name=retrieval
#SBATCH --output=logs/retrieval.out
#SBATCH --error=logs/retrieval.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

queries=(
"data/neuclir2024.topics.test.jsonl"
"data/ragtime2025.topics.test.jsonl"
"data/ragtime2026.topics.test.jsonl"
)
index_dirs=(
"/home/dju/scratch/neuclir1/modernbert-base.cover-5k/corpus_emb*pkl"
"/home/dju/scratch/ragtime1/modernbert-base.cover-5k/corpus_emb*pkl"
"/home/dju/scratch/ragtime2/modernbert-base.cover-5k/corpus_emb*pkl"
)
run_outputs=(
"runs/runs.neuclir2024.cover.test.txt"
"runs/runs.ragtime2025.cover.test.txt"
"runs/runs.ragtime2026.cover.test.txt"
)

for shard in {0..2}; do
    query="${queries[shard]}"
    index_dir="${index_dirs[shard]}"
    run_output="${run_outputs[shard]}"

    python run_retrieval.py \
        --queries "$query" \
        --model_name_or_path DylanJHJ/modernbert-base.cover-5k \
        --passage_reps "$index_dir" \
        --output "$run_output" \
        --query_prefix "search_query: " \
        --pooling mean \
        --normalize \
        --query_max_len 128 \
        --save_text \
        --batch_size 64 \
        --depth 1000
done
