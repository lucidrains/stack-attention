import pytest
param = pytest.mark.parametrize

import torch
from torch import cat
from einops import rearrange, einsum

from stack_attention import (
    DataStructureTransLayer,
    combine_superpositions,
    data_structure
)

from stack_attention.data_structure import (
    DataStructure,
    action,
    readout,
    IntrospectionTrajectory,
    Stack,
    Queue,
    TuringTape,
    RegisterFile,
    DualStack
)
from stack_attention.stack_trans_layer import exists

# helpers

def push(stack, item):
    return cat((item.unsqueeze(-2), stack[..., :-1, :]), dim = -2)

def pop(stack):
    return cat((stack[..., 1:, :], torch.zeros_like(stack[..., :1, :])), dim = -2)

# turing machine tape helpers

def tm_readout(state):
    tape, head = state
    return einsum(tape, head, 'b h l d, b h l -> b h d')

def tm_transition(state, inputs):
    tape, head = state

    inputs = rearrange(inputs, 'b h d -> b h 1 d')
    head_expanded = rearrange(head, 'b h l -> b h l 1')

    return [
        (tape, head),                                    # 0: noop
        (tape, head.roll(-1, dims = -1)),                # 1: move left
        (tape, head.roll(1, dims = -1)),                 # 2: move right
        (tape.lerp(inputs, head_expanded), head),        # 3: write under head
    ]

# 5-action stack (push 2, push 1, noop, pop 1, pop 2)

@param('return_action_entropies', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_5_action_stack(return_action_entropies, stochastic_action, hard_action):
    dim = 128
    dim_stack = 16
    stack_size = 16

    def push_n(stack, inputs, n):
        pushed = rearrange(inputs, 'b h (m d) -> b h m d', m = 2)[:, :, :n]
        return cat((pushed, stack), dim = -2)[:, :, :stack_size]

    ds = DataStructure(
        init = lambda batch, num_heads, device, dtype: torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype),
        actions = {
            'push_2': lambda stack, inputs: push_n(stack, inputs, 2),
            'push_1': lambda stack, inputs: push_n(stack, inputs, 1),
            'noop': lambda stack: stack,
            'pop_1': lambda stack: pop(stack),
            'pop_2': lambda stack: pop(pop(stack))
        },
        readout = lambda stack: stack[..., 0, :],
        dim_inputs = 2 * dim_stack,
        dim_readout = dim_stack
    )

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 4,
        data_structure = ds
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
    assert exists(layer.to_action_logits.weight.grad)
    assert exists(layer.combine.weight.grad)

# dual stack vm (call stack + eval stack)

@param('return_action_entropies', (False, True))
@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_dual_stack_vm(return_action_entropies, state_conditioned, stochastic_action, hard_action):
    dim = 128
    dim_stack = 16
    stack_size = 16

    ds = DataStructure(
        init = lambda batch, num_heads, device, dtype: (
            torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype),
            torch.zeros((batch, num_heads, stack_size, dim_stack), device = device, dtype = dtype)
        ),
        actions = {
            'noop': lambda s: s,
            'push_eval': lambda s, x: (s[0], push(s[1], x)),
            'pop_eval': lambda s: (s[0], pop(s[1])),
            'push_call': lambda s, x: (push(s[0], x), s[1]),
            'pop_call': lambda s: (pop(s[0]), s[1]),
            'eval_to_call': lambda s: (push(s[0], s[1][..., 0, :]), pop(s[1])),
            'call_to_eval': lambda s: (pop(s[0]), push(s[1], s[0][..., 0, :]))
        },
        readout = lambda s: cat((s[0][..., 0, :], s[1][..., 0, :]), dim = -1),
        dim_inputs = dim_stack,
        dim_readout = 2 * dim_stack
    )

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 4,
        state_conditioned = state_conditioned,
        data_structure = ds
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
    assert exists(layer.to_action_logits.weight.grad)
    assert exists(layer.combine.weight.grad)

# neural turing machine tape (1d tape with movable read/write head)

@param('return_action_entropies', (False, True))
@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
def test_turing_machine_tape(return_action_entropies, state_conditioned, stochastic_action, hard_action):
    dim = 128
    dim_tape = 16
    tape_len = 16

    def init_state(batch, num_heads, device, dtype):
        tape = torch.zeros((batch, num_heads, tape_len, dim_tape), device = device, dtype = dtype)
        head = torch.zeros((batch, num_heads, tape_len), device = device, dtype = dtype)
        head[..., tape_len // 2] = 1.
        return (tape, head)

    ds = DataStructure(
        init = init_state,
        actions = {
            'noop': lambda s: s,
            'left': lambda s: (s[0], s[1].roll(-1, dims = -1)),
            'right': lambda s: (s[0], s[1].roll(1, dims = -1)),
            'write': lambda s, x: (s[0].lerp(x.unsqueeze(-2), s[1].unsqueeze(-1)), s[1])
        },
        readout = tm_readout,
        dim_inputs = dim_tape,
        dim_readout = dim_tape
    )

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 4,
        state_conditioned = state_conditioned,
        data_structure = ds
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

    tape, head = state

    assert out1.shape == out2.shape == tokens.shape
    assert tape.shape == (2 * 32, 4, tape_len, dim_tape)
    assert head.shape == (2 * 32, 4, tape_len)
    assert exists(tokens.grad)
    assert exists(layer.to_action_logits.weight.grad)
    assert exists(layer.combine.weight.grad)

    assert torch.allclose(head.sum(dim = -1), torch.ones_like(head.sum(dim = -1)), atol = 1e-4)
    assert (head >= -1e-5).all()

def test_turing_machine_behavior():
    batch, num_heads, tape_len, dim_tape = 2, 1, 8, 4

    tape = torch.zeros((batch, num_heads, tape_len, dim_tape))
    head = torch.zeros((batch, num_heads, tape_len))
    head[..., 0] = 1.
    state = (tape, head)

    write_val = torch.ones((batch, num_heads, dim_tape)) * 3.14

    noop_s, left_s, right_s, write_s = tm_transition(state, write_val)

    assert torch.allclose(noop_s[0], tape) and torch.allclose(noop_s[1], head)
    assert torch.allclose(tm_readout(write_s), write_val)
    assert right_s[1][0, 0, 1] == 1. and right_s[1][0, 0, 0] == 0.
    assert left_s[1][0, 0, tape_len - 1] == 1.

def test_combine_superpositions_single_candidate():
    actions = torch.ones(2, 4, 1)
    state = (torch.randn(2, 4, 8), torch.randn(2, 4, 16))

    out = combine_superpositions(state, actions)
    assert torch.allclose(out[0], state[0])
    assert torch.allclose(out[1], state[1])

# declarative data structure subclassing

class Counter(DataStructure):
    def __init__(self, dim = 16):
        super().__init__(dim_inputs = dim, dim_readout = dim)
        self.dim = dim

    def init_state(self):
        return torch.zeros(self.dim)

    @action
    def noop(self, state):
        return state

    @action
    def increment(self, state, item):
        return state + item

    @action
    def reset(self, state):
        return torch.zeros_like(state)

    @readout
    def observe(self, state):
        return state

    def render(self, state):
        return f"norm={state.norm():.2f}"

def test_student_subclass_declaration():
    dim = 64
    dim_inner = 16

    ds = Counter(dim = dim_inner)
    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        data_structure = ds
    )

    assert layer.num_actions == 3
    assert layer.action_names == ['noop', 'increment', 'reset']

    tokens = torch.randn(2, 8, dim, requires_grad = True)
    out, state = layer(tokens)

    assert out.shape == tokens.shape
    assert state.shape == (2 * 8, 2, dim_inner)
    out.sum().backward()
    assert exists(tokens.grad)

# functional / dict declaration

def test_functional_declaration():
    dim = 64
    dim_tape = 8

    tape_ds = DataStructure(
        init = lambda: (torch.zeros(16, dim_tape), torch.zeros(16)),
        actions = {
            'noop': lambda s: s,
            'left': lambda s: (s[0], s[1].roll(-1, -1)),
            'right': lambda s: (s[0], s[1].roll(1, -1)),
            'write': lambda s, x: (s[0].lerp(x.unsqueeze(-2), s[1].unsqueeze(-1)), s[1])
        },
        readout = lambda s: einsum(s[0], s[1], '... l d, ... l -> ... d'),
        dim_inputs = dim_tape,
        dim_readout = dim_tape
    )

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        data_structure = tape_ds
    )

    assert layer.num_actions == 4
    assert layer.action_names == ['noop', 'left', 'right', 'write']

    tokens = torch.randn(2, 4, dim)
    out, state = layer(tokens)
    assert out.shape == tokens.shape

# the simplest declaration - plain functions, no classes

def test_plain_function_declaration():
    dim = 64

    counter = data_structure(
        init = lambda: torch.zeros(1),
        actions = {
            'noop': lambda state: state,
            'increment': lambda state, item: state + item
        },
        readout = lambda state: state,
        render = lambda state: f'count={state.item():.1f}'
    )

    # dims are inferred from the initial state and readout

    compiled = counter.compile()
    assert compiled.dim_inputs == 1
    assert compiled.dim_readout == 1

    layer = DataStructureTransLayer(dim = dim, num_heads = 2, data_structure = counter)
    assert layer.action_names == ['noop', 'increment']

    tokens = torch.randn(2, 4, dim, requires_grad = True)
    out, state = layer(tokens, recurrent = True)

    assert out.shape == (2, 4, dim)
    assert state.shape == (2, 2, 1)

    out.sum().backward()
    assert exists(tokens.grad)

    trajectory = layer.introspect(tokens)
    assert 'count=' in trajectory.summary()

def test_inherited_actions():
    class Base(DataStructure):
        def init_state(self):
            return torch.zeros(4)

        @action
        def noop(self, state):
            return state

        @readout
        def read(self, state):
            return state

    class Child(Base):
        @action
        def double(self, state):
            return state * 2

    layer = DataStructureTransLayer(dim = 32, data_structure = Child())
    assert layer.action_names == ['noop', 'double']

# standard data structures (stack, queue, turing tape, register file, dual stack)

def test_builtin_data_structures():
    dim = 64

    # stack
    stack_layer = DataStructureTransLayer(dim = dim, data_structure = Stack(depth = 8, dim = 16))
    assert stack_layer.num_actions == 3
    assert stack_layer.action_names == ['push', 'pop', 'noop']
    out, state = stack_layer(torch.randn(2, 4, dim))
    assert out.shape == (2, 4, dim)

    # queue
    queue_layer = DataStructureTransLayer(dim = dim, data_structure = Queue(size = 8, dim = 16))
    assert queue_layer.num_actions == 3
    assert queue_layer.action_names == ['noop', 'enqueue', 'dequeue']
    out, state = queue_layer(torch.randn(2, 4, dim))
    assert out.shape == (2, 4, dim)

    # turing tape
    tape_layer = DataStructureTransLayer(dim = dim, data_structure = TuringTape(length = 8, dim = 16))
    assert tape_layer.num_actions == 4
    assert tape_layer.action_names == ['noop', 'left', 'right', 'write']
    out, state = tape_layer(torch.randn(2, 4, dim))
    assert out.shape == (2, 4, dim)

    # register file
    reg_layer = DataStructureTransLayer(dim = dim, data_structure = RegisterFile(num_registers = 4, dim = 16))
    assert reg_layer.num_actions == 4
    assert reg_layer.action_names == ['noop', 'next', 'prev', 'write']
    out, state = reg_layer(torch.randn(2, 4, dim))
    assert out.shape == (2, 4, dim)

    # dual stack
    dual_layer = DataStructureTransLayer(dim = dim, data_structure = DualStack(depth = 8, dim = 16))
    assert dual_layer.num_actions == 7
    out, state = dual_layer(torch.randn(2, 4, dim))
    assert out.shape == (2, 4, dim)

# queue fifo behavior

def test_queue_behavior():
    size = 4
    dim = 2

    q = Queue(size = size, dim = dim)
    buf, head, tail = q.init_state()

    item1 = torch.tensor([1., 2.])
    item2 = torch.tensor([3., 4.])

    s1 = q.enqueue((buf, head, tail), item1)
    assert torch.allclose(q.front(s1), item1)

    s2 = q.enqueue(s1, item2)
    assert torch.allclose(q.front(s2), item1)

    s3 = q.dequeue(s2)
    assert torch.allclose(q.front(s3), item2)

# recurrent sequence execution

@param('state_conditioned', (False, True))
@param('stochastic_action', (False, True))
@param('hard_action', (False, True))
@param('return_action_entropies', (False, True))
def test_recurrent_sequence_execution(state_conditioned, stochastic_action, hard_action, return_action_entropies):
    dim = 64
    depth = 8
    dim_stack = 16
    batch = 2
    seq_len = 4

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        state_conditioned = state_conditioned,
        data_structure = Stack(depth = depth, dim = dim_stack)
    )

    tokens = torch.randn(batch, seq_len, dim, requires_grad = True)

    ret = layer(
        tokens,
        recurrent = True,
        stochastic_action = stochastic_action,
        hard_action = hard_action,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out, final_state, entropies = ret
        assert entropies.shape == (batch, seq_len, layer.num_heads)
        loss = out.sum() + entropies.sum()
    else:
        out, final_state = ret
        loss = out.sum()

    assert out.shape == (batch, seq_len, dim)
    assert final_state.shape == (batch, layer.num_heads, depth, dim_stack)

    loss.backward()
    assert exists(tokens.grad)
    assert exists(layer.to_action_logits.weight.grad)

# latent recurrent steps (thinking loop)

@param('state_conditioned', (False, True))
@param('return_action_entropies', (False, True))
def test_latent_recurrent_steps(state_conditioned, return_action_entropies):
    dim = 64
    depth = 8
    dim_stack = 16
    batch = 3
    num_steps = 4

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        state_conditioned = state_conditioned,
        data_structure = Stack(depth = depth, dim = dim_stack)
    )

    tokens = torch.randn(batch, dim, requires_grad = True)

    ret = layer(
        tokens,
        num_recurrent_steps = num_steps,
        return_action_entropies = return_action_entropies
    )

    if return_action_entropies:
        out, final_state, entropies = ret
        assert entropies.shape == (batch, num_steps, layer.num_heads)
        loss = out.sum() + entropies.sum()
    else:
        out, final_state = ret
        loss = out.sum()

    assert out.shape == (batch, num_steps, dim)
    assert final_state.shape == (batch, layer.num_heads, depth, dim_stack)

    loss.backward()
    assert exists(tokens.grad)
    assert exists(layer.to_action_logits.weight.grad)

# introspection trajectory

def test_introspection_trajectory():
    dim = 64
    depth = 8
    dim_stack = 16
    batch = 2
    seq_len = 5

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        data_structure = Stack(depth = depth, dim = dim_stack)
    )

    tokens = torch.randn(batch, seq_len, dim)

    out, final_state, trajectory = layer(tokens, recurrent = True, introspect = True)

    assert isinstance(trajectory, IntrospectionTrajectory)
    assert trajectory.seq_len == seq_len
    assert trajectory.num_heads == 2
    assert trajectory.num_actions == 3
    assert trajectory.action_probs.shape == (batch, seq_len, 2, 3)
    assert trajectory.entropies.shape == (batch, seq_len, 2)
    assert trajectory.readouts.shape == (batch, seq_len, 2, dim_stack)
    assert trajectory.outputs.shape == (batch, seq_len, dim)
    assert len(trajectory.states) == seq_len

    actions = trajectory.most_likely_actions(batch_idx = 0, head_idx = 0)
    assert len(actions) == seq_len
    assert all(act_name in ['push', 'pop', 'noop'] for _, act_name, _ in actions)

    act_matrix = trajectory.action_matrix(batch_idx = 0, head_idx = 0)
    assert act_matrix.shape == (seq_len, 3)

    summary = trajectory.summary(batch_idx = 0, head_idx = 0)
    assert "Step" in summary
    assert "Action" in summary
    assert "Prob" in summary
    assert "Entropy" in summary
    assert "depth=" in summary

    traj2 = layer.introspect(tokens)
    assert isinstance(traj2, IntrospectionTrajectory)
    assert traj2.seq_len == seq_len

# vmap auto-vectorization

def test_vmap_support():
    dim = 64
    depth = 8
    dim_stack = 8

    class UnbatchedStack(DataStructure):
        def __init__(self):
            super().__init__(dim_inputs = dim_stack, dim_readout = dim_stack, vmap = True)

        def init_state(self):
            return torch.zeros(depth, dim_stack)

        @action
        def push(self, s, x):
            return torch.cat((x.unsqueeze(0), s[:-1]), dim = 0)

        @action
        def pop(self, s):
            return torch.cat((s[1:], torch.zeros_like(s[:1])), dim = 0)

        @action
        def noop(self, s):
            return s

        @readout
        def top(self, s):
            return s[0]

    layer = DataStructureTransLayer(
        dim = dim,
        num_heads = 2,
        data_structure = UnbatchedStack()
    )

    tokens = torch.randn(2, 4, dim, requires_grad = True)
    out, state = layer(tokens)
    assert out.shape == tokens.shape
    out.sum().backward()
    assert exists(tokens.grad)

# researcher error protections

def test_action_and_readout_arity_protection():
    import pytest

    # 0 args (excluding self)
    with pytest.raises(AssertionError, match = 'got 0 positional arguments'):
        class BadAction0(DataStructure):
            @action
            def act(self):
                pass

    # > 2 args (excluding self)
    with pytest.raises(AssertionError, match = 'got 3 positional arguments'):
        class BadAction3(DataStructure):
            @action
            def act(self, state, item, extra):
                pass

    # readout with > 1 args
    with pytest.raises(AssertionError, match = 'must accept only'):
        class BadReadout(DataStructure):
            @readout
            def read(self, state, extra):
                pass

def test_compile_error_protections():
    import pytest

    # in-place mutation protection
    class InPlaceMutator(DataStructure):
        def init_state(self):
            return torch.zeros(4)

        @action
        def bad_inc(self, state):
            state += 1.
            return state

        @readout
        def read(self, state):
            return state

    with pytest.raises(AssertionError, match = 'mutated the input state in-place'):
        InPlaceMutator().compile()

    # non-floating point leaves
    class IntActionState(DataStructure):
        def init_state(self):
            return torch.zeros(4)

        @action
        def act(self, state):
            return torch.zeros_like(state, dtype = torch.long)

        @readout
        def read(self, state):
            return state

    with pytest.raises(AssertionError, match = 'non-floating point'):
        IntActionState().compile()

    # action returns None
    class NoneReturn(DataStructure):
        def init_state(self):
            return torch.zeros(4)

        @action
        def forgot_return(self, state):
            pass

        @readout
        def read(self, state):
            return state

    with pytest.raises(AssertionError, match = 'returned None'):
        NoneReturn().compile()

    # action returns wrong shape
    class BadShape(DataStructure):
        def init_state(self):
            return torch.zeros(4)

        @action
        def wrong_shape(self, state):
            return state[..., :2]

        @readout
        def read(self, state):
            return state

    with pytest.raises(AssertionError, match = 'expected'):
        BadShape().compile()
