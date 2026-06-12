#!/bin/bash -l
#SBATCH --job-name=oracle-gen
#SBATCH --output=logs/oracle-gen.out
#SBATCH --error=logs/oracle-gen.err
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

cd ${HOME}/trec2026

TRACK=neuclir
CONFIG=config/oracle-complete/${TRACK}-oracle-complete.yaml
echo "[$(date)] === config=${CONFIG} ==="
python run_oracle.py "$CONFIG"
echo "[$(date)] === done ${CONFIG} ==="
