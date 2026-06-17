# Evaluation Results — NeuCLIR

## Baseline

| Run | File | StRecall@10 | StRecall@20 | α-nDCG@10 | α-nDCG@20 |
|-----|------|------------:|------------:|----------:|----------:|
| runs | neuclir2024.cover.test.txt               | 0.6879 | 0.7890 | 0.5711 | 0.5996 |
| runs | neuclir2024.cover.lancer-top100.test.txt | 0.7479 | 0.8169 | 0.6343 | 0.6596 |
| runs | neuclir2024.cover.lancer-top200.test.txt | 0.7416 | 0.8324 | 0.6286 | 0.6597 |
| runs | neuclir2024.cover.lancer-top500.test.txt | 0.7197 | 0.8137 | 0.6156 | 0.6507 |
| runs | neuclir2024.cover.lancer_expr-top100.test.txt | 0.7591 | 0.8250 | 0.6876 | 0.7092 |
| runs | neuclir2024.cover.lancer_expr-top200.test.txt | 0.7631 | 0.8390 | 0.6877 | 0.7092 |

## Cover Experiments (New Metrics)

| Run | StRecall@10 | StRecall@20 | α-nDCG@10 | α-nDCG@20 |
|-----|------------:|------------:|----------:|----------:|
| cover-top1  | 0.2835 | 0.2835 | 0.2516 | 0.2441 |
| cover-top2  | 0.3639 | 0.3639 | 0.3449 | 0.3347 |
| cover-top3  | 0.4908 | 0.4908 | 0.4316 | 0.4189 |
| cover-top5  | 0.5613 | 0.5613 | 0.4992 | 0.4845 |
| cover-top7  | 0.5949 | 0.5949 | 0.5256 | 0.5101 |
| cover-top10 | 0.6879 | 0.6879 | 0.5711 | 0.5545 |
| cover-top15 | 0.6879 | 0.7584 | 0.5711 | 0.5867 |
| cover-top20 | 0.6879 | 0.7890 | 0.5711 | 0.5996 |
| cover-top25 | 0.6879 | 0.7890 | 0.5711 | 0.5996 |
| cover-top30 | 0.6879 | 0.7890 | 0.5711 | 0.5996 |
| cover-top50 | 0.6879 | 0.7890 | 0.5711 | 0.5996 |

## Oracle Experiments (New Metrics)

| Run | StRecall@10 | StRecall@20 | α-nDCG@10 | α-nDCG@20 |
|-----|------------:|------------:|----------:|----------:|
| oracle-top1     | 0.6108 | 0.6108 | 0.5367 | 0.5224 |
| oracle-top2     | 0.7928 | 0.7928 | 0.7085 | 0.6897 |
| oracle-top3     | 0.8860 | 0.8860 | 0.7885 | 0.7670 |
| oracle-top5     | 0.9510 | 0.9510 | 0.8448 | 0.8208 |
| oracle-top7     | 0.9736 | 0.9736 | 0.8595 | 0.8347 |
| oracle-top10    | 0.9905 | 0.9905 | 0.8690 | 0.8435 |
| oracle-top15    | 0.9905 | 0.9968 | 0.8690 | 0.8474 |
| oracle-top20    | 0.9905 | 1.0000 | 0.8690 | 0.8491 |
| oracle-top25    | 0.9905 | 1.0000 | 0.8690 | 0.8491 |
| oracle-top30    | 0.9905 | 1.0000 | 0.8690 | 0.8491 |
| oracle-top50    | 0.9905 | 1.0000 | 0.8690 | 0.8491 |
| oracle-complete | 0.9905 | 1.0000 | 0.8690 | 0.8491 |
