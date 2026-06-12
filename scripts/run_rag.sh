#!/bin/bash -l
#SBATCH --job-name=baseline-gen
#SBATCH --output=logs/baseline-gen.out
#SBATCH --error=logs/baseline-gen.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate vllm

cd $HOME/trec2026

TRACK=neuclir
TOPKS=(1 2 3 5 7 10 15 20 25 30 50)
for k in "${TOPKS[@]}"; do
    echo "[$(date)] run neuclir cover-top${k}"
    python run_rag.py config/cover-topk/neuclir-cover-top${k}.yaml
done
