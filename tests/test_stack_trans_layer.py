import pytest
param = pytest.mark.parametrize

import torch

@param('return_action_entropies', (False, True))
@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_stack_trans_layer(
    return_action_entropies,
    state_conditioned,
    stochastic_action,
    hard_action
):
    from stack_attention.stack_trans_layer import StackTransLayer, exists

    tokens = torch.randn(2, 512, 256, requires_grad = True)

    layer = StackTransLayer(256, state_conditioned = state_conditioned)

    ret1 = layer(
        tokens,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out1, state, entropies1 = ret1
        assert entropies1.shape == (2, 512, layer.num_stacks)
    else:
        out1, state = ret1

    ret2 = layer(
        tokens,
        stack_state = state,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out2, state, entropies2 = ret2
        assert entropies2.shape == (2, 512, layer.num_stacks)
        (out2.sum() + entropies2.sum()).backward()
    else:
        out2, state = ret2
        out2.sum().backward()

    assert out1.shape == out2.shape == tokens.shape
    assert exists(tokens.grad)
    assert exists(layer.to_action_logits.weight.grad)
    assert exists(layer.combine.weight.grad)

@param('return_action_entropies', (False, True))
@param('state_conditioned', (False, True))
def test_stack_trans_layer_recurrent(
    return_action_entropies,
    state_conditioned
):
    from stack_attention.stack_trans_layer import StackTransLayer, exists

    tokens = torch.randn(2, 16, 64, requires_grad = True)

    layer = StackTransLayer(
        64,
        num_stacks = 2,
        dim_stack = 8,
        stack_size = 12,
        state_conditioned = state_conditioned
    )

    ret1 = layer(
        tokens,
        recurrent = True,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out1, state, entropies1 = ret1
        assert entropies1.shape == (2, 16, layer.num_stacks)
    else:
        out1, state = ret1

    assert out1.shape == (2, 16, 64)
    assert state.shape == (2, layer.num_stacks, layer.stack_size, layer.dim_stack + 1)

    ret2 = layer(
        tokens,
        stack_state = state,
        recurrent = True,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out2, state, entropies2 = ret2
        (out2.sum() + entropies2.sum()).backward()
    else:
        out2, state = ret2
        out2.sum().backward()

    assert out1.shape == out2.shape == tokens.shape
    assert exists(tokens.grad)
    assert exists(layer.to_action_logits.weight.grad)
    assert exists(layer.combine.weight.grad)
    assert exists(layer.to_stack_inputs.weight.grad)

def test_stack_trans_layer_blocks_and_checkpointing():
    from stack_attention.stack_trans_layer import StackTransLayer

    batch = 2
    seq_len = 11
    dim = 64
    block_size = 4

    layer = StackTransLayer(
        dim,
        num_stacks = 2,
        dim_stack = 8,
        stack_size = 12,
        state_conditioned = True,
        return_action_entropies = True
    )

    tokens = torch.randn(batch, seq_len, dim)

    # 1. unblocked
    t1 = tokens.clone().requires_grad_()
    out1, state1, ent1 = layer(t1, recurrent = True)
    (out1.sum() + ent1.sum()).backward()

    # 2. block-recurrent without checkpointing
    t2 = tokens.clone().requires_grad_()
    out2, state2, ent2 = layer(t2, recurrent = True, block_size = block_size, checkpoint_blocks = False)
    (out2.sum() + ent2.sum()).backward()

    # 3. block-recurrent with checkpointing
    t3 = tokens.clone().requires_grad_()
    out3, state3, ent3 = layer(t3, recurrent = True, block_size = block_size, checkpoint_blocks = True)
    (out3.sum() + ent3.sum()).backward()

    # exact match across outputs, states, entropies, and gradients
    assert torch.allclose(out1, out2, atol = 1e-6)
    assert torch.allclose(out1, out3, atol = 1e-6)
    assert torch.allclose(state1, state2, atol = 1e-6)
    assert torch.allclose(state1, state3, atol = 1e-6)
    assert torch.allclose(ent1, ent2, atol = 1e-6)
    assert torch.allclose(ent1, ent3, atol = 1e-6)
    assert torch.allclose(t1.grad, t2.grad, atol = 1e-6)
    assert torch.allclose(t1.grad, t3.grad, atol = 1e-6)

