from __future__ import annotations
from functools import partial

import torch
from torch import cat, stack, nn, tensor
from torch.nn import Module, Linear, RMSNorm, Parameter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_einops_utils import (
    pack_with_inverse,
    pad_right_at_dim,
    entropy
)

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

def mask_value(t):
    return -torch.finfo(t.dtype).max

def masked_log(t, eps = 1e-20):
    # exact zeros are fully masked, while fractional masks from the superposition stay differentiable

    return torch.where(t > 0, log(t, eps), mask_value(t))

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
        state_conditioned = False,
        return_action_entropies = False,
        prenorm = False,
        add_residual = True,
        learned_residual_gate = True
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        # action related

        self.return_action_entropies = return_action_entropies

        num_actions = 3 # (push, pop, noop)
        dim_inner = num_stacks * dim_stack

        # the actions may optionally be conditioned on a global read of the current stacks (closed loop control)

        self.state_conditioned = state_conditioned
        dim_action_input = dim + (dim_inner if state_conditioned else 0)

        self.to_action_logits = LinearNoBias(dim_action_input, num_stacks * num_actions)
        self.split_action_logits = Rearrange('... (num_stacks num_actions) -> ... num_stacks num_actions', num_actions = num_actions)
        self.num_actions = num_actions

        # to hidden per stack for maybe push

        self.to_stack_inputs = LinearNoBias(dim, dim_inner)
        self.split_num_stacks = Rearrange('... (num_stacks dim_stack) -> ... num_stacks dim_stack', num_stacks = num_stacks)

        self.num_stacks = num_stacks
        self.stack_size = stack_size
        self.dim_stack = dim_stack

        # attention based read from stack

        self.to_read_stack_attn = LinearNoBias(dim_stack, 1)

        self.null_stack = Parameter(torch.randn(dim_stack) * 1e-2)

        # combining stack reads across number of stacks

        self.combine = LinearNoBias(dim_inner, dim)

        # maybe combining with residual

        self.add_residual = add_residual
        self.learned_residual_gate = learned_residual_gate and add_residual

        self.residual_scale = Parameter(tensor(0.)) if self.learned_residual_gate else None

    def global_read(self, stack):
        # global read - eq 4
        # the mask rides along as the last channel of the stack and is split off here

        batch, num_stacks = stack.shape[:2]

        stack, mask = stack[..., :-1], stack[..., -1]

        null_stack = repeat(self.null_stack, 'd -> b h 1 d', b = batch, h = num_stacks)


        stack_with_null = cat((stack, null_stack), dim = -2)
        mask_with_null = pad_right_at_dim(mask, 1, value = 1.)

        attn_logits = self.to_read_stack_attn(stack_with_null)
        attn_logits = rearrange(attn_logits, '... 1 -> ...')
        attn_logits = attn_logits + masked_log(mask_with_null)

        attn = attn_logits.softmax(dim = -1)

        return einsum(stack_with_null, attn, 'b h s d, b h s -> b h d')

    def step(
        self,
        tokens,
        stack_state,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False
    ):
        residual = tokens
        tokens = self.norm(tokens)

        # project to stack inputs, appending a 1. into the mask channel for the pushed element

        stack_inputs = self.to_stack_inputs(tokens)
        stack_inputs = self.split_num_stacks(stack_inputs) # (b h d)
        stack_inputs = F.pad(stack_inputs, (0, 1), value = 1.)

        # the actions per stack per token, optionally conditioned on a read of the current stacks

        action_inputs = tokens

        if self.state_conditioned:
            state_read = self.global_read(stack_state)
            action_inputs = cat((tokens, rearrange(state_read, 'b h d -> b (h d)')), dim = -1)

        action_logits = self.to_action_logits(action_inputs)
        action_logits = self.split_action_logits(action_logits) # (b h 3)
        action_logits = action_logits / action_temperature

        if stochastic_action:
            action_logits = action_logits + gumbel_noise_like(action_logits)

        actions = action_logits.softmax(dim = -1)
        action_entropies = entropy(actions, reduce = False)

        if hard_action:
            soft_actions = actions
            hard_actions = F.one_hot(actions.argmax(dim = -1), self.num_actions)
            actions = straight_through(soft_actions, hard_actions)

        # differentiable stack - eq 2 and 3, mask included as the last channel

        zeros = torch.zeros_like(stack_inputs)

        stack_with_inputs, _ = pack_with_inverse((stack_inputs, stack_state, zeros), 'b h * d')

        # superposition of all three possibilities, weighted summed by the action

        pushed_stack = stack_with_inputs[..., :-2, :]
        popped_stack = stack_with_inputs[..., 2:, :]
        noop_stack = stack_state

        stack_superpositions = torch.stack((pushed_stack, popped_stack, noop_stack), dim = -1)

        next_stack_state = einsum(stack_superpositions, actions, 'b h s d a, b h a -> b h s d')

        # global read of the updated stacks

        read_stack_out = self.global_read(next_stack_state)

        # combine heads

        out = rearrange(read_stack_out, 'b h d -> b (h d)')
        out = self.combine(out)

        # maybe add residual

        if self.add_residual:
            residual_scale = self.residual_scale.exp() if exists(self.residual_scale) else 1.
            out = out + residual * residual_scale

        return out, next_stack_state, actions, action_entropies, read_stack_out

    def _forward_recurrent(
        self,
        tokens,
        stack_state,
        block_size = None,
        checkpoint_blocks = False,
        **step_kwargs
    ):
        assert tokens.ndim == 3, 'tokens must be (batch, seq, dim) for recurrent execution'
        num_steps = tokens.shape[1]

        # initial stack state

        if not exists(stack_state):
            stack_state = torch.zeros(
                (tokens.shape[0], self.num_stacks, self.stack_size, self.dim_stack + 1),
                device = tokens.device,
                dtype = tokens.dtype
            )

        # unblocked runs in single chunk

        if not exists(block_size) or block_size >= num_steps:
            return self._forward_recurrent_chunk(tokens, stack_state, **step_kwargs)

        # chunked recurrence with optional gradient checkpointing

        outputs, entropies = [], []

        chunk_forward = partial(
            self._forward_recurrent_chunk,
            **step_kwargs
        )

        chunk_runner = partial(checkpoint, chunk_forward, use_reentrant = False) if checkpoint_blocks else chunk_forward

        for chunk in tokens.split(block_size, dim = 1):
            chunk_out, stack_state, chunk_entropies = chunk_runner(chunk, stack_state)
            outputs.append(chunk_out)
            entropies.append(chunk_entropies)

        return cat(outputs, dim = 1), stack_state, cat(entropies, dim = 1)

    def _forward_recurrent_chunk(self, tokens, stack_state, **step_kwargs):
        outputs, entropies = [], []

        # recurrent over sequence

        for step in range(tokens.shape[1]):
            token = tokens[:, step]
            out, stack_state, actions, action_entropy, read_stack_out = self.step(
                token, stack_state, **step_kwargs
            )
            outputs.append(out)
            entropies.append(action_entropy)

        outputs = stack(outputs, dim = 1)
        entropies = stack(entropies, dim = 1)

        return outputs, stack_state, entropies

    def forward(
        self,
        tokens,
        stack_state = None,
        recurrent = False,
        block_size = None,
        checkpoint_blocks = False,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False,
        return_action_entropies = None
    ):
        assert action_temperature > 0.

        return_action_entropies = default(return_action_entropies, self.return_action_entropies)

        step_kwargs = dict(
            action_temperature = action_temperature,
            stochastic_action = stochastic_action,
            hard_action = hard_action
        )

        # parallel is recurrent with seq len 1 on packed tokens

        if not recurrent:
            tokens, inverse_pack = pack_with_inverse(tokens, '* d')
            tokens = rearrange(tokens, 'b d -> b 1 d')

        out, stack_state, entropies = self._forward_recurrent(
            tokens,
            stack_state,
            block_size = block_size,
            checkpoint_blocks = checkpoint_blocks,
            **step_kwargs
        )

        if not recurrent:
            out = inverse_pack(rearrange(out, 'b 1 d -> b d'))
            entropies = inverse_pack(rearrange(entropies, 'b 1 h -> b h'), '* h')

        if not return_action_entropies:
            return out, stack_state

        return out, stack_state, entropies
