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

OVERALL_STATUS=0

for TRACK in \
    neuclir \
    ragtime \
    ; do

    # oracle topk
    for CONFIG in config/oracle-topk/${TRACK}-oracle-top*.yaml; do
        echo "[$(date)] === config=${CONFIG} ==="
        python run_oracle.py "$CONFIG"
        STATUS=$?
        [ $STATUS -ne 0 ] && OVERALL_STATUS=$STATUS
        echo "[$(date)] === done ${CONFIG} (exit ${STATUS}) ==="
    done

    # oracle complete
    CONFIG=config/oracle-complete/${TRACK}-oracle-complete.yaml
    echo "[$(date)] === config=${CONFIG} ==="
    python run_oracle.py "$CONFIG"
    STATUS=$?
    [ $STATUS -ne 0 ] && OVERALL_STATUS=$STATUS
    echo "[$(date)] === done ${CONFIG} (exit ${STATUS}) ==="

done

exit $OVERALL_STATUS
