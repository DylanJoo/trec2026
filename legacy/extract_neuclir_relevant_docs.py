dataset='neuclir24'
docids = []

# Relevant one
with open(f'/home/dju/trec2026/data/neuclir/{dataset}-test-request.qrel', 'r') as f:
    for line in f:
        parts = line.split()
        docids.append(parts[2])

# Runs
files = [
    '/home/dju/trec2026/runs/runs.neuclir2024.cover.test.txt'
]
for file in files:
    with open(file, 'r') as f:
        for line in f:
            parts = line.split()
            docids.append(parts[2])

docids_set = set(docids)
print(len(docids_set))

from datasets import load_dataset
ds = load_dataset('json', data_files='/home/dju/scratch/neuclir1/*.processed.jsonl.gz', num_proc=8)['train']

corpus = {}
for doc in ds.filter(lambda x: x['id'] in docids_set):
    corpus[doc['id']] = {'title': doc['title'], 'text': doc['text']}

print(len(corpus))

import json
import gzip
output_path = f'/home/dju/trec2026/data/neuclir/{dataset}-relevant-docs.jsonl.gz'
with gzip.open(output_path, 'wt', encoding='utf-8') as f:
    for doc_id, doc in corpus.items():
        f.write(json.dumps({'id': doc_id, 'title': doc['title'], 'text': doc['text']}) + '\n')
print(f"Written to {output_path}")

# dataset='ragtime25'
# docids = []
# with open(f'data/ragtime/2025.mlir.qrels', 'r') as f:
#     for line in f:
#         parts = line.split()
#         if int(parts[3]) > 0:
#             docids.append(parts[2])
#
# docids_set = set(docids)
# print(len(docids_set))
#
# from datasets import load_dataset
# ds = load_dataset('json', data_files='/home/dju/scratch/ragtime1/*.processed.jsonl.gz', num_proc=8)['train']
#
# corpus = {}
# for doc in ds.filter(lambda x: x['id'] in docids_set):
#     corpus[doc['id']] = {'title': doc['title'], 'text': doc['text']}
#
# print(len(corpus))
#
# import json
# output_path = f'data/ragtime/{dataset}-relevant-docs.jsonl'
# with open(output_path, 'w') as f:
#     for doc_id, doc in corpus.items():
#         f.write(json.dumps({'id': doc_id, 'title': doc['title'], 'text': doc['text']}) + '\n')
# print(f"Written to {output_path}")
