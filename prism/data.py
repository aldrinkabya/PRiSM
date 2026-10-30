import os
import pandas as pd
import torch
from torch.utils.data import Dataset

# task: (number of classes, max sequence length)
TASKS = {
    'sst2': (2, 256), 'sst5': (5, 256), 'mr': (2, 256), 'cr': (2, 256),
    'mpqa': (2, 128), 'subj': (2, 256), 'trec': (6, 128), 'cola': (2, 256),
    'mnli': (3, 256), 'mnli-mm': (3, 256), 'snli': (3, 256), 'qnli': (2, 256),
    'rte': (2, 256), 'mrpc': (2, 256), 'qqp': (2, 256),
}

PAIR_TASKS = ['rte', 'mrpc', 'mnli', 'mnli-mm', 'snli']
GLUE_FOLDERS = {'sst2': 'SST-2', 'cola': 'CoLA', 'qnli': 'QNLI', 'rte': 'RTE', 'mrpc': 'MRPC', 'qqp': 'QQP', 'snli': 'SNLI'}


def task_files(data_dir, task):
    """Returns the (train, dev) file paths for a task."""
    # The LM-BFF archive extracts into an 'original' subfolder
    root = os.path.join(data_dir, 'original')
    if not os.path.exists(root):
        root = data_dir

    if task == 'mnli':
        return os.path.join(root, 'MNLI/train.tsv'), os.path.join(root, 'MNLI/dev_matched.tsv')
    if task == 'mnli-mm':
        return os.path.join(root, 'MNLI/train.tsv'), os.path.join(root, 'MNLI/dev_mismatched.tsv')
    if task in GLUE_FOLDERS:
        folder = os.path.join(root, GLUE_FOLDERS[task])
        return os.path.join(folder, 'train.tsv'), os.path.join(folder, 'dev.tsv')
    folder = os.path.join(root, task)
    return os.path.join(folder, 'train.csv'), os.path.join(folder, 'test.csv')


def read_task(path, task):
    if task in ['rte', 'qnli']:
        df = pd.read_csv(path, sep='\t', quoting=3)
        df['label'] = df['label'].map({'entailment': 0, 'not_entailment': 1})
    elif task == 'mrpc':
        df = pd.read_csv(path, sep='\t', quoting=3)
        df.columns = ['label', 'id1', 'id2', 'sentence1', 'sentence2']
        df['label'] = df['label'].astype(int)
    elif task == 'cola':
        df = pd.read_csv(path, sep='\t', quoting=3, header=None, names=['id', 'label', 'misc', 'sentence'])
    elif task == 'sst2':
        df = pd.read_csv(path, sep='\t', quoting=3)
        df.columns = ['sentence', 'label']
        df['label'] = df['label'].astype(int)
    elif task in ['sst5', 'mr', 'cr', 'mpqa', 'subj', 'trec']:
        df = pd.read_csv(path, header=None, names=['label', 'sentence'])
        df['label'] = df['label'].astype(int)
    elif task in ['mnli', 'mnli-mm']:
        df = pd.read_csv(path, sep='\t', quoting=3)
        df = df[['sentence1', 'sentence2', 'gold_label']]
        df['label'] = df['gold_label'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2})
    elif task == 'snli':
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    rows.append({'sentence1': parts[7], 'sentence2': parts[8], 'gold_label': parts[-1]})
        df = pd.DataFrame(rows)
        df['label'] = df['gold_label'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2})
    elif task == 'qqp':
        df = pd.read_csv(path, sep='\t', quoting=3)
        df.columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
        df['label'] = df['is_duplicate'].astype(int)
    else:
        raise ValueError(f"Unsupported task: {task}")

    for col in ['sentence', 'sentence1', 'sentence2']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    return df


class ClassificationDataset(Dataset):
    def __init__(self, path, task, tokenizer, max_len):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.task = task
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = read_task(path, task)

    def __len__(self):
        return len(self.data)

    def texts(self, row):
        if self.task in PAIR_TASKS:
            return row['sentence1'], row['sentence2']
        if self.task == 'qqp':
            return row['question1'], row['question2']
        if self.task == 'qnli':
            return row['question'], row['sentence']
        return row['sentence'], None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_a, text_b = self.texts(row)
        enc = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long),
        }
