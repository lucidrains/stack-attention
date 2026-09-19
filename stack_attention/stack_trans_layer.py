from __future__ import annotations
from functools import partial

import torch
from torch import cat, nn, Tensor
from torch.nn import Module, Linear, RMSNorm, Parameter
import torch.nn.functional as F

from torch_einops_utils import pack_with_inverse, pad_left_at_dim, pad_right_at_dim

import einx
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

# ein equations

# b - batch seq
# h - heads or num stacks
# s - stack depth
# d - dim stack
# a - actions

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

def mask_value(t):
    return -torch.finfo(t.dtype).max

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

        self.num_stacks = num_stacks
        self.stack_size = stack_size
        self.dim_stack = dim_stack

        # attention based read from stack

        self.to_read_stack_attn = LinearNoBias(dim_stack, 1)

        self.null_stack = Parameter(torch.randn(dim_stack) * 1e-2)

        # combine

        self.combine = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens,
        stack_states: tuple[Tensor, Tensor] | None = None,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False
    ):
        assert action_temperature > 0.

        orig, device = tokens, tokens.device

        # maybe pre norm

        tokens = self.norm(tokens)

        tokens, inverse_pack = pack_with_inverse(tokens, '* d')

        # dimensions

        batch, num_stacks, stack_size, dim_stack = tokens.shape[0], self.num_stacks, self.stack_size, self.dim_stack

        # project to stack inputs

        stack_inputs = self.to_stack_inputs(tokens)
        stack_inputs = self.split_num_stacks(stack_inputs) # (b h d)

        # the actions per stack per token

        action_logits = self.to_action_logits(tokens)
        action_logits = self.split_action_logits(action_logits) # (b h 3)
        action_logits = action_logits / action_temperature

        if stochastic_action:
            action_logits = action_logits + gumbel_noise_like(action_logits)

        actions = action_logits.softmax(dim = -1)

        if hard_action:
            soft_actions = actions
            hard_actions = F.one_hot(actions.argmax(dim = -1), self.num_actions)
            actions = straight_through(soft_actions, hard_actions)

        # handle initial stack state
        # but the transformer could dynamically generate this initial state as well

        if not exists(stack_states):
            init_stack = torch.zeros((batch, num_stacks, stack_size, dim_stack), device = device)
            init_stack_mask = torch.zeros((batch, num_stacks, stack_size), dtype = torch.bool, device = device)
            stack_states = (init_stack, init_stack_mask)

        # differentiable stack - eq 3 (applies to mask as well)

        stack, stack_mask = stack_states
        zeros = torch.zeros_like(stack_inputs)

        stack_with_inputs, _ = pack_with_inverse((stack_inputs, stack, zeros), 'b h * d')

        stack_mask_with_input = pad_left_at_dim(stack_mask, 1, value = True, dim = -1)
        stack_mask_with_input = pad_right_at_dim(stack_mask_with_input, 1, value = False, dim = -1)

        # superposition of all three possibilities, weighted summed by the action

        pushed_stack = stack_with_inputs[..., :-2, :]
        popped_stack = stack_with_inputs[..., 2:,:]
        noop_stack = stack

        stack_superpositions = torch.stack((pushed_stack, popped_stack, noop_stack), dim = -1)

        pushed_mask = stack_mask_with_input[..., :-2]
        popped_mask = stack_mask_with_input[..., 2:]
        noop_mask = stack_mask

        stack_mask_superpositions = torch.stack((pushed_mask, popped_mask, noop_mask), dim = -1)

        next_stack = einsum(stack_superpositions, actions, 'b h s d a, b h a -> b h s d')
        next_mask = einsum(stack_mask_superpositions.float(), actions.float(), 'b h s a, b h a -> b h s').bool()

        next_stack_states = (next_stack, next_mask)

        # read from the stacks, using their attention like method
        # however, will do some adjustments as it does not look quite right

        null_stack = repeat(self.null_stack, 'd -> b h 1 d', b = batch, h = num_stacks)

        next_stack_with_null = cat((next_stack, null_stack), dim = -2)
        next_mask_with_null = pad_right_at_dim(next_mask, 1, value = True)

        read_stack_attn_logits = self.to_read_stack_attn(next_stack_with_null)
        read_stack_attn_logits = rearrange(read_stack_attn_logits, '... 1 -> ...')

        read_stack_attn_logits = read_stack_attn_logits.masked_fill(next_mask_with_null, mask_value(next_stack_with_null))

        read_stack_attn = read_stack_attn_logits.softmax(dim = -1)

        read_stack_out = einsum(next_stack_with_null, read_stack_attn, 'b h s d, b h s -> b h d')

        # combine

        out = rearrange(stack_inputs, 'b h d -> b (h d)')

        out = self.combine(out)

        out = inverse_pack(out)

        return out, next_stack_states
