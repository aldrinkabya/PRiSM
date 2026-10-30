import random
import numpy as np
import torch


def top_layers(scores, k, highest=True):
    """The k highest- (or lowest-) scoring layers."""
    k = min(k, len(scores))
    return torch.topk(scores if highest else -scores, k=k).indices.cpu().numpy()


def blockwise_layers(scores, k, num_blocks, highest=True):
    """Splits the layers into equal consecutive blocks and takes the k / num_blocks
    highest- (or lowest-) scoring layers from each. PRiSM uses highest=True."""
    num_layers = len(scores)
    if num_layers % num_blocks != 0:
        raise ValueError(f"Layers ({num_layers}) must be divisible by B ({num_blocks}).")
    if k % num_blocks != 0:
        raise ValueError(f"k ({k}) must be divisible by B ({num_blocks}).")

    size = num_layers // num_blocks
    per_block = k // num_blocks
    chosen = []
    for b in range(num_blocks):
        block = scores[b * size:(b + 1) * size]
        idx = torch.topk(block if highest else -block, k=per_block).indices
        chosen.append(idx + b * size)
    return np.sort(torch.cat(chosen).cpu().numpy())


def random_layers(num_layers, k):
    return random.sample(range(num_layers), k)


def select_layers(method, scores, k, num_blocks):
    if method == 'blockwise_highest':
        return blockwise_layers(scores, k, num_blocks, highest=True)
    if method == 'blockwise_lowest':
        return blockwise_layers(scores, k, num_blocks, highest=False)
    if method == 'highest':
        return top_layers(scores, k, highest=True)
    if method == 'lowest':
        return top_layers(scores, k, highest=False)
    raise ValueError(f"Unknown method: {method}")
