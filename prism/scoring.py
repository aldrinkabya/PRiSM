import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def score_layers(encoder, loader, device):
    """Scores each layer by the mean cosine similarity of its [CLS] vector with
    those of every other layer, averaged over the batches of `loader`."""
    encoder.eval()
    num_layers = encoder.config.num_hidden_layers
    totals = torch.zeros(num_layers, device=device)
    batches = 0

    for batch in tqdm(loader, desc="Scoring layers"):
        out = encoder(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        cls = torch.stack([h[:, 0, :] for h in out.hidden_states[1:]])

        for i in range(num_layers):
            sims = [F.cosine_similarity(cls[i], cls[j], dim=1).mean() for j in range(num_layers) if j != i]
            totals[i] += sum(sims) / len(sims)
        batches += 1

    return totals / batches
