#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=logs/search.out.%a
#SBATCH --error=logs/search.err.%a
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-1
#SBATCH --mem=256G
#SBATCH --time=2:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

MODEL_DIRS=(
DylanJHJ/modernbert-base.cover-5k
)

SCRATCH_DIRS=(
"neuclir1"
"ragtime1"
"ragtime2"
)

RUN_NAMES=(
"runs/runs.neuclir2024.cover.test.txt"
"runs/runs.ragtime2025.cover.test.txt"
"runs/runs.ragtime2026.cover.test.txt"
)

SCRATCH=${SCRATCH_DIRS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}

for model_dir in "${MODEL_DIRS[@]}"; do
    echo "Processing model: $model_dir, dataset: $RUN_NAME"
    output_dir=${HOME}/scratch/${SCRATCH}/${model_dir##*/}
    mkdir -p $output_dir

    python -m tevatron.retriever.driver.search \
        --query_reps $output_dir/query_emb.pkl \
        --passage_reps "$output_dir/corpus_emb*pkl" \
        --depth 1000 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to ${RUN_NAME}.tmp

    python -m tevatron.utils.format.convert_result_to_trec \
        --input ${RUN_NAME}.tmp --output ${RUN_NAME}.orig

    echo "Done: ${RUN_NAME}.orig"
done
