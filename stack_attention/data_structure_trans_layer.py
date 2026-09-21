from __future__ import annotations
from functools import partial

import torch
from torch import cat, stack, nn, tensor
from torch.nn import Module, Linear, RMSNorm, Parameter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_einops_utils import (
    pack_with_inverse,
    tree_flatten_with_inverse,
    entropy
)

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from stack_attention.data_structure import (
    DataStructure,
    IntrospectionTrajectory
)

# ein equations

# b - batch
# n - sequence length
# h - number of heads
# a - number of actions
# d - readout dim per head

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
    if not isinstance(candidates, list):
        candidates = [candidates]

    flat_states, inverses = zip(*[tree_flatten_with_inverse(s) for s in candidates])
    stacked_leaves = [stack(leaves, dim = -1) for leaves in zip(*flat_states)]
    inverse = inverses[0]

    assert all(leaf.is_floating_point() for leaf in stacked_leaves), 'all state tensor leaves must be floating point to support differentiable superposition'

    out = [
        einsum(leaf, actions, 'b h ... a, b h a -> b h ...')
        for leaf in stacked_leaves
    ]

    return inverse(out)

# classes

class DataStructureTransLayer(Module):

    def __init__(
        self,
        dim,
        *,
        data_structure: DataStructure,
        num_heads = 4,
        state_conditioned = False,
        return_action_entropies = False,
        prenorm = False,
        add_residual = True,
        learned_residual_gate = True
    ):
        super().__init__()

        assert isinstance(data_structure, DataStructure), 'data_structure must be a DataStructure'

        compiled = data_structure.compile()
        dim_inputs, dim_readout = compiled.dim_inputs, compiled.dim_readout

        assert exists(dim_inputs) and exists(dim_readout), (
            'could not infer dim_inputs / dim_readout from the data structure, '
            'please pass dim = ... to data_structure(...)'
        )

        self.data_structure = data_structure
        self.action_names = compiled.action_names
        self.num_actions = compiled.num_actions
        self.num_heads = num_heads
        self.dim_inputs = dim_inputs
        self.dim_readout = dim_readout
        self.transition = compiled.transition
        self.readout = compiled.readout
        self.init_state = compiled.init_state

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        # combine the per-head readouts back to the token dimension

        dim_inner = num_heads * dim_readout
        self.combine = LinearNoBias(dim_inner, dim)

        # action logits, optionally conditioned on a read of the current state

        self.return_action_entropies = return_action_entropies
        self.state_conditioned = state_conditioned

        dim_action_input = dim + (dim_inner if state_conditioned else 0)
        self.to_action_logits = LinearNoBias(dim_action_input, num_heads * self.num_actions)
        self.split_action_logits = Rearrange('... (h a) -> ... h a', a = self.num_actions)

        # per-head inputs for actions that take an item

        self.to_inputs = LinearNoBias(dim, num_heads * dim_inputs)
        self.split_inputs = Rearrange('... (h d) -> ... h d', h = num_heads)

        # residual

        self.add_residual = add_residual
        self.learned_residual_gate = learned_residual_gate and add_residual
        self.residual_scale = Parameter(tensor(0.)) if self.learned_residual_gate else None

    def step(
        self,
        tokens,
        state,
        action_temperature = 1.,
        stochastic_action = False,
        hard_action = False
    ):
        residual = tokens
        tokens = self.norm(tokens)

        # per-head inputs

        inputs = self.split_inputs(self.to_inputs(tokens))

        # action distribution, optionally conditioned on the current readout

        action_inputs = tokens

        if self.state_conditioned:
            state_readout = self.readout(state)
            action_inputs = cat((tokens, rearrange(state_readout, 'b h d -> b (h d)')), dim = -1)

        action_logits = self.split_action_logits(self.to_action_logits(action_inputs)) / action_temperature

        if stochastic_action:
            action_logits = action_logits + gumbel_noise_like(action_logits)

        actions = action_logits.softmax(dim = -1)
        action_entropies = entropy(actions, reduce = False)

        if hard_action:
            actions = straight_through(actions, F.one_hot(actions.argmax(dim = -1), self.num_actions))

        # superpose the candidate next states

        next_state = combine_superpositions(self.transition(state, inputs), actions)

        # read out from the updated state and combine heads

        readout = self.readout(next_state)
        out = self.combine(rearrange(readout, 'b h d -> b (h d)'))

        # maybe add residual

        if self.add_residual:
            residual_scale = self.residual_scale.exp() if exists(self.residual_scale) else 1.
            out = out + residual * residual_scale

        return out, next_state, actions, action_entropies, readout

    def _forward_recurrent(
        self,
        tokens,
        state,
        num_recurrent_steps,
        introspect,
        block_size = None,
        checkpoint_blocks = False,
        **step_kwargs
    ):
        # latent loop vs sequence recurrence

        is_latent_loop = exists(num_recurrent_steps)

        if is_latent_loop:
            if tokens.ndim == 3 and tokens.shape[1] == 1:
                tokens = rearrange(tokens, 'b 1 d -> b d')

            assert tokens.ndim == 2, 'tokens must be (batch, dim) when using num_recurrent_steps'
            num_steps = num_recurrent_steps
        else:
            assert tokens.ndim == 3, 'tokens must be (batch, seq, dim) for recurrent execution'
            num_steps = tokens.shape[1]

        # initial state

        if not exists(state):
            state = self.init_state(tokens.shape[0], self.num_heads, tokens.device, tokens.dtype)

        # unblocked or latent loop executes in a single chunk

        if is_latent_loop or not exists(block_size) or block_size >= num_steps:
            return self._forward_recurrent_chunk(
                tokens,
                state,
                is_latent_loop = is_latent_loop,
                introspect = introspect,
                num_steps = num_steps,
                **step_kwargs
            )

        # chunked recurrence with optional gradient checkpointing

        outputs, entropies, trajectories = [], [], []

        chunk_forward = partial(
            self._forward_recurrent_chunk,
            is_latent_loop = False,
            introspect = introspect,
            **step_kwargs
        )

        chunk_runner = partial(checkpoint, chunk_forward, use_reentrant = False) if checkpoint_blocks else chunk_forward

        for chunk in tokens.split(block_size, dim = 1):
            chunk_out, state, chunk_entropies, chunk_traj = chunk_runner(chunk, state)

            outputs.append(chunk_out)
            entropies.append(chunk_entropies)

            if introspect:
                trajectories.append(chunk_traj)

        outputs = cat(outputs, dim = 1)
        entropies = cat(entropies, dim = 1)

        # stitch introspection trajectories across chunks

        trajectory = None
        if introspect:
            trajectory = IntrospectionTrajectory(
                action_names = self.action_names,
                action_probs = cat([t.action_probs for t in trajectories], dim = 1),
                entropies = cat([t.entropies for t in trajectories], dim = 1),
                readouts = cat([t.readouts for t in trajectories], dim = 1),
                outputs = cat([t.outputs for t in trajectories], dim = 1),
                states = sum([t.states for t in trajectories], []),
                data_structure = self.data_structure
            )

        return outputs, state, entropies, trajectory

    def _forward_recurrent_chunk(
        self,
        tokens,
        state,
        is_latent_loop = False,
        introspect = False,
        num_steps = None,
        **step_kwargs
    ):
        num_steps = default(num_steps, tokens.shape[1])

        outputs, actions, entropies, readouts, states = [], [], [], [], []
        current = tokens

        # recurrent over sequence or latent loop

        for step in range(num_steps):
            token = current if is_latent_loop else tokens[:, step]

            out, state, action, action_entropy, readout = self.step(token, state, **step_kwargs)

            outputs.append(out)
            actions.append(action)
            entropies.append(action_entropy)
            readouts.append(readout)

            if introspect:
                states.append(state)

            if is_latent_loop:
                current = out

        outputs = stack(outputs, dim = 1)
        entropies = stack(entropies, dim = 1)

        # introspection trajectory

        trajectory = None
        if introspect:
            trajectory = IntrospectionTrajectory(
                action_names = self.action_names,
                action_probs = stack(actions, dim = 1),
                entropies = entropies,
                readouts = stack(readouts, dim = 1),
                outputs = outputs,
                states = states,
                data_structure = self.data_structure
            )

        return outputs, state, entropies, trajectory

    def introspect(self, tokens, **kwargs):
        kwargs.setdefault('introspect', True)

        if 'recurrent' not in kwargs and 'num_recurrent_steps' not in kwargs:
            kwargs['recurrent'] = tokens.ndim == 3 and tokens.shape[1] > 1

        _, _, trajectory = self.forward(tokens, **kwargs)
        return trajectory

    def forward(
        self,
        tokens,
        state = None,
        recurrent = False,
        block_size = None,
        checkpoint_blocks = False,
        num_recurrent_steps = None,
        introspect = False,
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

        is_parallel = not recurrent and not exists(num_recurrent_steps)

        if is_parallel:
            tokens, inverse_pack = pack_with_inverse(tokens, '* d')
            tokens = rearrange(tokens, 'b d -> b 1 d')

        out, state, entropies, trajectory = self._forward_recurrent(
            tokens,
            state,
            num_recurrent_steps = num_recurrent_steps,
            introspect = introspect,
            block_size = block_size,
            checkpoint_blocks = checkpoint_blocks,
            **step_kwargs
        )

        if is_parallel:
            out = inverse_pack(rearrange(out, 'b 1 d -> b d'))
            entropies = inverse_pack(rearrange(entropies, 'b 1 h -> b h'), '* h')

        if introspect:
            return out, state, trajectory

        if return_action_entropies:
            return out, state, entropies

        return out, state
