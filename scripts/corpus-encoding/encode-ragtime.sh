#!/bin/bash -l
#SBATCH --job-name=ragtime-encode
#SBATCH --output=logs/ragtime-encode.out.%a
#SBATCH --error=logs/ragtime-encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 

MODEL_DIRS=(
DylanJHJ/modernbert-base.cover-5k
)

LANGS=(
"eng-docs"
"arb-trans"
"rus-trans"
"zho-trans"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/scratch/ragtime1/${model_dir##*/}
    mkdir -p $output_dir

    # for SHARD_ID in 0 1 2 3 4;do
    #     echo Encoding ragtime1 corpus $SHARD_ID
    #     python -m tevatron.retriever.driver.encode \
    #         --output_dir=temp \
    #         --tokenizer_name answerdotai/ModernBERT-base \
    #         --model_name_or_path $model_dir \
    #         --per_device_eval_batch_size 1024 \
    #         --passage_max_len 1024 \
    #         --pooling mean --bf16 --normalize  \
    #         --passage_prefix "search_document: " \
    #         --dataset_path ${HOME}/scratch/ragtime1/${LANG}.processed.jsonl.gz \
    #         --encode_output_path $output_dir/corpus_emb.${LANG}-${SHARD_ID}.pkl \
    #         --dataset_shard_index ${SHARD_ID} \
    #         --dataset_number_of_shards 5
    # done

    topic_path=/home/dju/trec2026/data/ragtime/ragtime25_main_all.jsonl 
    if [[ $LANG -eq "eng-docs" ]]; then
        echo "Encoding $topic_path"
        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --pooling mean --normalize --bf16 \
            --query_prefix "search_query: " \
            --per_device_eval_batch_size 64 \
            --dataset_path $topic_path \
            --encode_output_path $output_dir/query_emb.pkl \
            --query_max_len 128 \
            --encode_is_query
    fi

done
