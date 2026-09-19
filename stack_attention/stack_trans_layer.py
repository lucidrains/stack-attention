from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class StackTransLayer(Module):
    # Kechi Zhang et al. https://arxiv.org/abs/2507.15343

    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
