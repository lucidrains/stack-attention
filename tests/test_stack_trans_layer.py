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
