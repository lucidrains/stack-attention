import pytest
param = pytest.mark.parametrize

import torch

@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_stack_trans_layer(
    stochastic_action,
    hard_action
):
    from stack_attention.stack_trans_layer import StackTransLayer

    tokens = torch.randn(2, 512, 256)

    layer = StackTransLayer(256)

    out = layer(
        tokens,
        stochastic_action = stochastic_action,
        hard_action = hard_action
    )

    assert out.shape == tokens.shape
