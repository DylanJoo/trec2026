#!/bin/bash -l
#SBATCH --job-name=direct-gen
#SBATCH --output=logs/direct-gen.out
#SBATCH --error=logs/direct-gen.err
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

CONFIG=${1:-config/direct/ragtime-direct.yaml}

echo "[$(date)] === config=${CONFIG} ==="
python run_direct.py "$CONFIG"
STATUS=$?

[ $STATUS -ne 0 ] && OVERALL_STATUS=$STATUS
echo "[$(date)] === done ${CONFIG} (exit ${STATUS}) ==="
