from __future__ import annotations
from functools import partial

import torch
from torch import nn
from torch.nn import Module, Linear, RMSNorm
import torch.nn.functional as F

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import pack_with_inverse

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# gumbel

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def gumbel_noise_like(t):
    return -log(-log(torch.rand_like(t)))

def straight_through(src, tgt):
    return src + (tgt - src).detach()

# classes

class StackTransLayer(Module):
    # Kechi Zhang et al. https://arxiv.org/abs/2507.15343

    def __init__(
        self,
        dim,
        *,
        num_stacks = 4,     # heads
        dim_stack = 16,     # dim head
        stack_size = 24,
        prenorm = False
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        # action related

        num_actions = 3 # (push, pop, noop)
        self.to_action_logits = LinearNoBias(dim, num_stacks * num_actions)
        self.split_action_logits = Rearrange('... (num_stacks num_actions) -> ... num_stacks num_actions', num_actions = num_actions)
        self.num_actions = num_actions

        # to hidden per stack for maybe push

        dim_inner = num_stacks * dim_stack
        self.to_stack_inputs = LinearNoBias(dim, dim_inner)
        self.split_num_stacks = Rearrange('... (num_stacks dim_stack) -> ... num_stacks dim_stack', num_stacks = num_stacks)

        # combine

        self.combine = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens,
        stack_states = None,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False
    ):
        orig = tokens

        tokens = self.norm(tokens)

        tokens, inverse_pack = pack_with_inverse(tokens, '* d')

        # project to stack inputs

        stack_inputs = self.to_stack_inputs(tokens)
        stack_inputs = self.split_num_stacks(stack_inputs) # (b h d)

        # the actions per stack per token

        action_logits = self.to_action_logits(tokens)
        action_logits = self.split_action_logits(action_logits)
        action_logits = action_logits / action_temperature

        if stochastic_action:
            action_logits = action_logits + gumbel_noise_like(action_logits)

        actions = action_logits.softmax(dim = -1)

        if hard_action:
            soft_actions = actions
            hard_actions = F.one_hot(actions.argmax(dim = -1), self.num_actions)
            actions = straight_through(soft_actions, hard_actions)

        # stack related logic (todo ...)

        # combine

        out = rearrange(stack_inputs, '... h d -> ... (h d)')

        out = self.combine(out)

        return inverse_pack(out)
