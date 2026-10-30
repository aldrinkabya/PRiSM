import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses = []
    correct = 0

    bar = tqdm(loader, desc="Training", leave=False)
    for batch in bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        correct += torch.sum(logits.argmax(dim=1) == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return correct.double() / len(loader.dataset), np.mean(losses)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Returns accuracy, mean loss and macro-F1."""
    model.eval()
    losses = []
    correct = 0
    labels_all, preds_all = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=1)
        losses.append(loss_fn(logits, labels).item())
        correct += torch.sum(preds == labels)
        labels_all.extend(labels.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

    acc = correct.double() / len(loader.dataset)
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return acc, np.mean(losses), f1
