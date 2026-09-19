import pytest
param = pytest.mark.parametrize

import torch

@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_stack_trans_layer(
    state_conditioned,
    stochastic_action,
    hard_action
):
    from stack_attention.stack_trans_layer import StackTransLayer, exists

    tokens = torch.randn(2, 512, 256, requires_grad = True)

    layer = StackTransLayer(256, state_conditioned = state_conditioned)

    out1, state = layer(
        tokens,
        stochastic_action = stochastic_action,
        hard_action = hard_action
    )

    out2, state = layer(
        tokens,
        stack_states = state,
        stochastic_action = stochastic_action,
        hard_action = hard_action
    )

    assert out1.shape == out2.shape == tokens.shape

    out2.sum().backward()
    assert exists(tokens.grad)
