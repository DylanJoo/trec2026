#!/bin/bash -l
#SBATCH --job-name=rac-eval
#SBATCH --output=logs/rac-eval.out
#SBATCH --error=logs/rac-eval.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

source ${HOME}/.bashrc
initconda
conda activate inference

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Track-specific resources
NEUCLIR_QREL="data/neuclir/neuclir24-test-request.qrel"
NEUCLIR_JUDGE="data/neuclir/neuclir24-relevant-docs.jsonl"
CRUX_ROOT=${HOME}/datasets/crux

cd ${HOME}/trec2026/

# Retrieval run files in runs/
for run in runs/runs.neuclir*; do
    python -m src.evaluator.rac_eval \
        --run $run \
        --qrel data/neuclir/neuclir24-test-request.qrel \
        --judge data/neuclir/neuclir24.ratings.human.jsonl
done

for run in outputs/neuclir/*/context.run; do
    python -m src.evaluator.rac_eval \
        --run $run \
        --qrel data/neuclir/neuclir24-test-request.qrel \
        --judge data/neuclir/neuclir24.ratings.human.jsonl
done
