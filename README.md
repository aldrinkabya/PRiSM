# PRiSM

Code for "[PRiSM: Partial Ranking via Inter-layer Semantic Measurement for Efficient Fine-tuning of Language Models](https://doi.org/10.63317/3eyz8rr5qun6)", LREC 2026.

## Overview

PRiSM fine-tunes only a few layers of a pre-trained language model and picks them without any training. Each layer is scored by how similar its [CLS] representation is to those of the other layers, and the highest-scoring layer in each of $B$ consecutive blocks is fine-tuned.

## Repository layout

```
.
├── prism/
│   ├── scoring.py      layer scores from inter-layer [CLS] similarity
│   ├── selection.py    block-wise and other selection rules
│   ├── model.py        BERT classifier with per-layer freezing
│   ├── data.py         dataset loading for the 15 classification tasks
│   └── engine.py       training and evaluation loops
├── train.py            command-line entry point
└── download_data.py    fetches the datasets
```

## Quick start

```bash
pip install -r requirements.txt
python download_data.py
python train.py --task sst2
```

`download_data.py` fetches the datasets as packaged by [LM-BFF](https://github.com/princeton-nlp/LM-BFF) into `data/glue_data/`. Python 3.10 is recommended.

`train.py` prints the layer scores, the selected layers and per-epoch results, then a summary of the best dev accuracy and macro-F1, which is also saved to `results/<task>_<method>_k<k>.csv`.

## Options

| Option | Default | |
| :--- | :--- | :--- |
| `--task` | required | `sst2`, `sst5`, `mr`, `cr`, `mpqa`, `subj`, `trec`, `cola`, `mnli`, `mnli-mm`, `snli`, `qnli`, `rte`, `mrpc`, `qqp` |
| `--method` | `blockwise_highest` | Layer selection rule, see below |
| `--k` | 3 | Layers to fine-tune |
| `--B` | 3 | Blocks; `k` and the layer count must both be divisible by it |
| `--epochs` | 5 | |
| `--lr` | 2e-5 | |
| `--batch_size` | 32 | |
| `--seed` | 42 | |

`--method` takes PRiSM's rule, `blockwise_highest`, or one of the comparisons from the paper: `blockwise_lowest`, `highest` and `lowest` (the top or bottom $k$ layers by score, without blocks), `random`, and `full_ft` (full fine-tuning).

```bash
python train.py --task rte --method random
python train.py --task mrpc --method full_ft
```

## Citation

```bibtex
@inproceedings{biswas2026prism,
  title     = {{PRiSM}: Partial Ranking via Inter-layer Semantic Measurement for Efficient Fine-tuning of Language Models},
  author    = {Biswas, Aldrin Kabya and Fahim, Md and Amin, Md. Ashraful and Ali, Amin Ahsan and Rahman, AKM Mahbubur},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  pages     = {10313--10323},
  year      = {2026},
  address   = {Palma de Mallorca, Spain},
  publisher = {ELRA Language Resources Association},
  doi       = {10.63317/3eyz8rr5qun6}
}
```
