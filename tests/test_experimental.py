import pytest
param = pytest.mark.parametrize

import torch
from torch import cat
from einops import rearrange
from torch_einops_utils import pack_with_inverse
from stack_attention import DataStructureTransLayer
from stack_attention.stack_trans_layer import exists

# helpers

def push(stack, item):
    return cat((item.unsqueeze(-2), stack[..., :-1, :]), dim = -2)

def pop(stack):
    return cat((stack[..., 1:, :], torch.zeros_like(stack[..., :1, :])), dim = -2)

# 5-action stack (push 2, push 1, noop, pop 1, pop 2)

@param('return_action_entropies', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_5_action_stack(return_action_entropies, stochastic_action, hard_action):
    dim = 128
    dim_stack = 16
    stack_size = 16

    def transition(stack, inputs):
        inputs = rearrange(inputs, 'b h (n d) -> b h n d', n = 2)
        zeros = torch.zeros_like(inputs)

        stack_with_inputs, _ = pack_with_inverse((inputs, stack, zeros), 'b h * d')

        return [stack_with_inputs[:, :, i:i + stack_size] for i in range(5)]

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 4,
        num_actions = 5,
        dim_inputs = 2 * dim_stack,
        dim_readout = dim_stack,
        transition = transition,
        init_state = lambda batch, num_heads, device, dtype: torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype),
        readout = lambda stack: stack[..., 0, :]
    )

    tokens = torch.randn(2, 32, dim, requires_grad = True)

    ret1 = layer(
        tokens,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out1, state, entropies1 = ret1
        assert entropies1.shape == (2, 32, layer.num_heads)
    else:
        out1, state = ret1

    ret2 = layer(
        tokens,
        state = state,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out2, state, entropies2 = ret2
        assert entropies2.shape == (2, 32, layer.num_heads)
        (out2.sum() + entropies2.sum()).backward()
    else:
        out2, state = ret2
        out2.sum().backward()

    assert out1.shape == out2.shape == tokens.shape
    assert state.shape == (2 * 32, 4, stack_size, dim_stack)
    assert exists(tokens.grad)

# dual stack vm (call stack + eval stack)

@param('return_action_entropies', (False, True))
@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_dual_stack_vm(return_action_entropies, state_conditioned, stochastic_action, hard_action):
    dim = 128
    dim_stack = 16
    stack_size = 16

    def vm_transition(state, inputs):
        call_stack, eval_stack = state

        top_call = call_stack[..., 0, :]
        top_eval = eval_stack[..., 0, :]

        return [
            (call_stack, eval_stack),
            (call_stack, push(eval_stack, inputs)),
            (call_stack, pop(eval_stack)),
            (push(call_stack, inputs), eval_stack),
            (pop(call_stack), eval_stack),
            (push(call_stack, top_eval), pop(eval_stack)),
            (pop(call_stack), push(eval_stack, top_call)),
        ]

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 4,
        num_actions = 7,
        dim_inputs = dim_stack,
        dim_readout = 2 * dim_stack,
        state_conditioned = state_conditioned,
        transition = vm_transition,
        init_state = lambda batch, num_heads, device, dtype: (
            torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype),
            torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype)
        ),
        readout = lambda s: cat((s[0][..., 0, :], s[1][..., 0, :]), dim = -1)
    )

    tokens = torch.randn(2, 32, dim, requires_grad = True)

    ret1 = layer(
        tokens,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out1, state, entropies1 = ret1
        assert entropies1.shape == (2, 32, layer.num_heads)
    else:
        out1, state = ret1

    ret2 = layer(
        tokens,
        state = state,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out2, state, entropies2 = ret2
        assert entropies2.shape == (2, 32, layer.num_heads)
        (out2.sum() + entropies2.sum()).backward()
    else:
        out2, state = ret2
        out2.sum().backward()

    assert out1.shape == out2.shape == tokens.shape
    assert len(state) == 2
    assert exists(tokens.grad)
