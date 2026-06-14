#!/bin/bash -l
#SBATCH --job-name=rerank
#SBATCH --output=logs/rerank.out
#SBATCH --error=logs/rerank.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40:4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate vllm

queries=(
"data/neuclir2024.topics.test.jsonl"
"data/ragtime2025.topics.test.jsonl"
# "data/ragtime2026.topics.test.jsonl"
)
run_outputs=(
"runs/runs.neuclir2024.cover.test.txt"
"runs/runs.ragtime2025.cover.test.txt"
# "runs/runs.ragtime2026.cover.test.txt"
)

cd ${HOME}/trec2026

for rerank in lancer lancer_expr;do
for topk in 200 500; do
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    INPUT_RUN=${HOME}/trec2026/runs/runs.neuclir2024.cover.test.txt
    OUTPUT_RUN=${HOME}/trec2026/runs/runs.neuclir2024.cover.${rerank}-top${topk}.test.txt
    OUTPUT_SQ=${HOME}/trec2026/runs/sq.neuclir2024.cover.${rerank}-top${topk}.test.jsonl

    python3 -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${rerank}.yaml \
        --llm.backend=vllm \
        --data.loader_type=neuclir \
        --data.input_run=${INPUT_RUN} \
        --data.output_run=${OUTPUT_RUN} \
        --data.output_subquestions=${OUTPUT_SQ} \
        --data.topk=1000 \
        --llm.model_name_or_path=$MODEL \
        --num_runs=2 \
        --top_k=$topk \
        --rank_end=$topk \
        --max_doc_length=1024
done
done
