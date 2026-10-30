import argparse
import csv
import os
import time

import torch
from tabulate import tabulate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from prism.data import TASKS, ClassificationDataset, task_files
from prism.engine import evaluate, train_epoch
from prism.model import CLSClassifier
from prism.scoring import score_layers
from prism.selection import random_layers, select_layers
from prism.utils import count_parameters, set_seed

METHODS = ['blockwise_highest', 'blockwise_lowest', 'highest', 'lowest', 'random', 'full_ft']


def parse_args():
    parser = argparse.ArgumentParser(description="PRiSM: training-free layer selection for partial fine-tuning of BERT")
    parser.add_argument('--task', type=str, required=True, choices=list(TASKS), help="Classification task")
    parser.add_argument('--data_dir', type=str, default='./data/glue_data')
    parser.add_argument('--method', type=str, choices=METHODS, default='blockwise_highest',
                        help="Layer selection strategy; blockwise_highest is PRiSM")
    parser.add_argument('--k', type=int, default=3, help="Number of layers to fine-tune")
    parser.add_argument('--B', type=int, default=3, help="Number of blocks")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Task: {args.task} | method: {args.method} | k={args.k}")

    n_classes, max_len = TASKS[args.task]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_path, dev_path = task_files(args.data_dir, args.task)
    train_set = ClassificationDataset(train_path, args.task, tokenizer, max_len)
    dev_set = ClassificationDataset(dev_path, args.task, tokenizer, max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    model = CLSClassifier(n_classes).to(device)

    # Stage 1: choose the layers to fine-tune
    selection_time = 0.0
    if args.method == 'random':
        model.train_only(random_layers(model.num_layers, args.k))
    elif args.method != 'full_ft':
        start = time.time()
        scores = score_layers(model.bert, train_loader, device)
        layers = select_layers(args.method, scores, args.k, args.B)
        selection_time = time.time() - start
        print(f"Layer scores: {[round(s, 4) for s in scores.tolist()]}")
        print(f"Selected layers: {layers}")
        model.train_only(layers)

    # Stage 2: fine-tune
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * args.epochs)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    best_acc, best_f1 = 0, 0
    start = time.time()
    for epoch in range(args.epochs):
        _, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        acc, _, f1 = evaluate(model, dev_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f} | dev acc {acc:.4f} | F1 {f1:.4f}")
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
    train_time = time.time() - start

    headers = ["Task", "Method", "k", "Selection time", "Training time", "Acc", "F1"]
    row = [args.task, args.method, args.k, f"{selection_time:.2f}s", f"{train_time:.2f}s",
           f"{best_acc * 100:.2f}", f"{best_f1 * 100:.2f}"]
    print("\n" + tabulate([row], headers=headers, tablefmt="grid"))

    out_file = os.path.join(args.output_dir, f"{args.task}_{args.method}_k{args.k}.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


if __name__ == "__main__":
    main()
