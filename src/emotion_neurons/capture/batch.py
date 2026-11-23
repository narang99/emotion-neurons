"""Right-padded batches of token ids, and masks of the positions to read."""

import torch
from jaxtyping import Bool, Int
from torch import Tensor


def padded(rows: list[list[int]], pad_id: int) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    """(input_ids, attention_mask), right padded."""
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for j, r in enumerate(rows):
        ids[j, : len(r)] = torch.tensor(r)
        mask[j, : len(r)] = 1
    return ids, mask


def position_mask(positions: list[list[int]], width: int) -> Bool[Tensor, "batch seq"]:
    mask = torch.zeros((len(positions), width), dtype=torch.bool)
    for j, p in enumerate(positions):
        mask[j, p] = True
    return mask
