from __future__ import annotations
from functools import partial

import torch
from torch import cat, nn, tensor
from torch.nn import Module, Linear, RMSNorm, Parameter
import torch.nn.functional as F
import torch.utils._pytree as pytree

from torch_einops_utils import pack_with_inverse
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

# ein equations

# b - batch seq
# h - heads
# a - actions

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def gumbel_noise_like(t):
    return -log(-log(torch.rand_like(t)))

def straight_through(src, tgt):
    return src + (tgt - src).detach()

# combine state superpositions across actions

def combine_superpositions(candidates, actions):
    if isinstance(candidates, list):
        flat_states, spec = zip(*[pytree.tree_flatten(s) for s in candidates])
        stacked_leaves = [torch.stack(leaves, dim = -1) for leaves in zip(*flat_states)]
        spec = spec[0]
    else:
        stacked_leaves, spec = pytree.tree_flatten(candidates)

    out = []
    for leaf in stacked_leaves:
        if leaf.dtype == torch.bool:
            comb = einsum(leaf.float(), actions.float(), 'b h ... a, b h a -> b h ...') > 0.5
        else:
            comb = einsum(leaf, actions, 'b h ... a, b h a -> b h ...')
        out.append(comb)

    return pytree.tree_unflatten(out, spec)

# generalized layer

class DataStructureTransLayer(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions,
        transition,
        readout,
        num_heads = 4,
        dim_inputs = 16,
        dim_readout = None,
        state_conditioned = False,
        init_state = None,
        prenorm = False,
        add_residual = True,
        learned_residual_gate = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.num_actions = num_actions
        self.num_heads = num_heads

        # transition, readout, and initial state

        self.transition = transition
        self.readout = readout
        self.init_state = init_state

        # combining readout back to token dimension

        dim_readout = default(dim_readout, dim_inputs)
        self.combine = LinearNoBias(num_heads * dim_readout, dim) if exists(dim_readout) else nn.Identity()

        # action logits (optionally state-conditioned for closed-loop control)

        self.state_conditioned = state_conditioned
        dim_action_input = dim + (num_heads * dim_readout if state_conditioned else 0)

        self.to_action_logits = LinearNoBias(dim_action_input, num_heads * num_actions)
        self.split_action_logits = Rearrange('... (h a) -> ... h a', a = num_actions)

        # inputs

        self.dim_inputs = dim_inputs
        self.to_inputs = LinearNoBias(dim, num_heads * dim_inputs) if exists(dim_inputs) else None
        self.split_inputs = Rearrange('... (h d) -> ... h d', h = num_heads) if exists(dim_inputs) else None

        # residual

        self.add_residual = add_residual
        self.learned_residual_gate = learned_residual_gate and add_residual
        self.residual_scale = Parameter(tensor(0.)) if self.learned_residual_gate else None

    def forward(
        self,
        tokens,
        state = None,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False
    ):
        residual = tokens

        tokens = self.norm(tokens)
        tokens, inverse_pack = pack_with_inverse(tokens, '* d')

        batch, num_heads = tokens.shape[0], self.num_heads

        # initial state

        if not exists(state):
            assert exists(self.init_state), 'init_state must be provided when state is None'
            state = self.init_state(batch, num_heads, tokens.device, tokens.dtype)

        # inputs per head

        inputs = None
        if exists(self.to_inputs):
            inputs = self.split_inputs(self.to_inputs(tokens))

        # action distribution (optionally conditioned on state observation)

        action_inputs = tokens

        if self.state_conditioned:
            state_obs = self.readout(state)
            if state_obs.ndim == 3:
                state_obs = rearrange(state_obs, 'b h d -> b (h d)')
            action_inputs = cat((tokens, state_obs), dim = -1)

        action_logits = self.split_action_logits(self.to_action_logits(action_inputs)) / action_temperature

        if stochastic_action:
            action_logits = action_logits + gumbel_noise_like(action_logits)

        actions = action_logits.softmax(dim = -1)

        if hard_action:
            actions = straight_through(actions, F.one_hot(actions.argmax(dim = -1), self.num_actions))

        # compute candidate next states and superpose

        candidates = self.transition(state, inputs)
        next_state = combine_superpositions(candidates, actions)

        # read out from updated state

        readout = self.readout(next_state)

        if readout.ndim == 3:
            readout = rearrange(readout, 'b h d -> b (h d)')

        out = self.combine(readout)
        out = inverse_pack(out)

        # residual

        if self.add_residual:
            scale = self.residual_scale.exp() if exists(self.residual_scale) else 1.
            out = out + residual * scale

        return out, next_state
