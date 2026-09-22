from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
from functools import partial
import inspect

import torch
from torch import cat, Tensor
from torch.nn import Module
from torch_einops_utils import tree_flatten_with_inverse
from einops import einsum, repeat

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# decorators

def action(fn = None, *, takes_item: bool | None = None):
    if not exists(fn):
        return partial(action, takes_item = takes_item)

    assert callable(fn), 'action must be a callable'

    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_self = len(params) > 0 and params[0].name == 'self'
    num_positional = len(params) - (1 if has_self else 0)

    assert 1 <= num_positional <= 2, (
        f"action '{fn.__name__}' must accept (state) or (state, item), got {num_positional} positional arguments"
    )

    fn._is_action = True
    if exists(takes_item):
        fn._takes_item = takes_item

    return fn

def _action_takes_item(fn: Callable) -> bool:
    explicit = getattr(fn, '_takes_item', None)
    if exists(explicit):
        return explicit

    target = fn.forward if isinstance(fn, Module) else fn
    return len(inspect.signature(target).parameters) >= 2

def readout(fn):
    assert callable(fn), 'readout must be a callable'

    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_self = len(params) > 0 and params[0].name == 'self'
    num_positional = len(params) - (1 if has_self else 0)

    assert num_positional == 1, (
        f"readout '{fn.__name__}' must accept only (state), got {num_positional} positional arguments"
    )

    fn._is_readout = True
    return fn

# introspection

@dataclass
class IntrospectionTrajectory:
    """Everything that happened while running a data structure, step by step.

    - action_probs: (batch, seq, heads, actions)
    - entropies: (batch, seq, heads)
    - readouts: (batch, seq, heads, dim_readout)
    - outputs: (batch, seq, dim)
    - states: the state after each step
    """
    action_names: list[str]
    action_probs: Tensor
    entropies: Tensor
    readouts: Tensor
    outputs: Tensor
    states: list[Any]
    data_structure: Any = None

    @property
    def seq_len(self) -> int:
        return self.action_probs.shape[1]

    @property
    def num_heads(self) -> int:
        return self.action_probs.shape[2]

    @property
    def num_actions(self) -> int:
        return self.action_probs.shape[3]

    @property
    def final_state(self) -> Any:
        return self.states[-1] if len(self.states) > 0 else None

    def most_likely_actions(self, batch_idx = 0, head_idx = 0) -> list[tuple[int, str, float]]:
        probs = self.action_probs[batch_idx, :, head_idx]
        indices = probs.argmax(dim = -1).tolist()
        confidences = probs.max(dim = -1).values.tolist()
        return [(t, self.action_names[i], c) for t, (i, c) in enumerate(zip(indices, confidences))]

    def action_matrix(self, batch_idx = 0, head_idx = 0) -> Tensor:
        return self.action_probs[batch_idx, :, head_idx]

    def _render_state(self, state, batch_idx = 0, head_idx = 0) -> str | None:
        if not exists(self.data_structure):
            return None

        leaves, inverse = tree_flatten_with_inverse(state)
        state_slice = inverse([leaf[batch_idx, head_idx] for leaf in leaves])

        try:
            return self.data_structure.render(state_slice)
        except Exception:
            return None

    def summary(self, batch_idx = 0, head_idx = 0, max_steps = 50) -> str:
        steps = min(self.seq_len, max_steps)
        probs = self.action_probs[batch_idx, :steps, head_idx]
        entropies = self.entropies[batch_idx, :steps, head_idx]
        readouts = self.readouts[batch_idx, :steps, head_idx]
        best_actions = probs.argmax(dim = -1)

        has_render = exists(self.data_structure) and steps > 0 and exists(self._render_state(self.states[0], batch_idx, head_idx))

        header = f"{'Step':>4} | {'Action':<15} | {'Prob':>7} | {'Entropy':>7} | {'Readout Norm':>12}"
        if has_render:
            header += f" | {'State':<20}"

        lines = [header, '-' * len(header)]

        for t in range(steps):
            act_idx = best_actions[t].item()
            line = f"{t:>4} | {self.action_names[act_idx]:<15} | {probs[t, act_idx].item() * 100:>6.1f}% | {entropies[t].item():>7.3f} | {readouts[t].norm().item():>12.3f}"

            if has_render and t < len(self.states):
                line += f" | {self._render_state(self.states[t], batch_idx, head_idx):<20}"

            lines.append(line)

        if self.seq_len > max_steps:
            lines.append(f"... ({self.seq_len - max_steps} more steps truncated)")

        return '\n'.join(lines)

    def print_summary(self, batch_idx = 0, head_idx = 0, max_steps = 50):
        print(self.summary(batch_idx, head_idx, max_steps))

    def __repr__(self) -> str:
        return f"<IntrospectionTrajectory: {self.seq_len} steps, {self.num_heads} heads, actions={self.action_names}>"

# the compiled form consumed by the trans layer

@dataclass(frozen = True)
class CompiledDataStructure:
    action_names: list[str]
    transition: Callable        # (state, inputs) -> list of candidate next states, one per action
    readout: Callable           # state -> (batch, heads, dim_readout)
    init_state: Callable        # (batch, heads, device, dtype) -> state
    dim_inputs: int | None
    dim_readout: int | None

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

# data structure

def _compile_init_state(init_fn: Callable) -> Callable:
    # an init with no arguments returns a prototype that is broadcast over batch and heads

    takes_batch_args = len(inspect.signature(init_fn).parameters) >= 4

    if takes_batch_args:
        return init_fn

    def init_state(batch, num_heads, device, dtype):
        prototype = init_fn()
        leaves, inverse = tree_flatten_with_inverse(prototype)

        expanded = [
            repeat(leaf.to(device = device, dtype = dtype), '... -> b h ...', b = batch, h = num_heads).clone()
            for leaf in leaves
        ]

        return inverse(expanded)

    return init_state

class DataStructure(Module):
    def __init__(
        self,
        *,
        init: Callable | None = None,
        actions: dict[str, Callable] | None = None,
        readout: Callable | None = None,
        readout_transform: Callable | None = None,
        render: Callable | None = None,
        dim_inputs: int | None = None,
        dim_readout: int | None = None,
        vmap: bool = False
    ):
        super().__init__()

        self.dim_inputs = dim_inputs
        self.dim_readout = dim_readout
        self.vmap = vmap

        self._init = init
        self._actions = actions
        self._readout = readout
        self._readout_transform = readout_transform
        self._render = render

        if exists(actions):
            for name, act in actions.items():
                if isinstance(act, Module):
                    self.add_module(f'action_{name}', act)

    def init_state(self):
        raise NotImplementedError(f'{type(self).__name__} must define init_state, or be given init = ...')

    def render(self, state):
        return self._render(state) if exists(self._render) else None

    def compile(self, validate = True) -> CompiledDataStructure:
        actions = self._collect_actions()
        assert len(actions) > 0, f'{type(self).__name__} has no actions. define them with @action, or pass actions = {{name: fn}}'

        readout_fn = self._collect_readout()
        assert exists(readout_fn), f'{type(self).__name__} has no readout. define it with @readout, or pass readout = fn'

        action_fns = list(actions.values())
        takes_item = [_action_takes_item(fn) for fn in action_fns]

        init_fn = self._init if exists(self._init) else self.init_state
        init_state_fn = _compile_init_state(init_fn)

        if self.vmap:
            action_fns = [torch.vmap(torch.vmap(fn)) for fn in action_fns]
            readout_fn = torch.vmap(torch.vmap(readout_fn))

        dim_inputs, dim_readout = self._resolve_dims(init_state_fn, readout_fn)

        if validate:
            self._validate(
                init_state_fn = init_state_fn,
                action_names = list(actions.keys()),
                action_fns = action_fns,
                takes_item = takes_item,
                readout_fn = readout_fn,
                dim_inputs = dim_inputs,
                dim_readout = dim_readout
            )

        def transition(state, inputs):
            return [
                fn(state, inputs) if needs_item else fn(state)
                for fn, needs_item in zip(action_fns, takes_item)
            ]

        return CompiledDataStructure(
            action_names = list(actions.keys()),
            transition = transition,
            readout = readout_fn,
            init_state = init_state_fn,
            dim_inputs = dim_inputs,
            dim_readout = dim_readout
        )

    def _validate(
        self,
        *,
        init_state_fn: Callable,
        action_names: list[str],
        action_fns: list[Callable],
        takes_item: list[bool],
        readout_fn: Callable,
        dim_inputs: int,
        dim_readout: int
    ):
        param = next(self.parameters(), None)
        device = param.device if exists(param) else 'cpu'
        dtype = param.dtype if exists(param) else torch.float32

        sample_state = init_state_fn(1, 1, device, dtype)
        sample_inputs = torch.randn(1, 1, dim_inputs, device = device, dtype = dtype)

        # 1. validate initial state leaves are floating point

        state_leaves, inverse = tree_flatten_with_inverse(sample_state)
        assert all(isinstance(l, Tensor) and l.is_floating_point() for l in state_leaves), (
            f'{type(self).__name__} initial state leaves must all be floating point tensors'
        )

        state_shapes = [tuple(l.shape) for l in state_leaves]

        # 2. validate readout

        try:
            sample_readout = readout_fn(sample_state)
        except Exception as e:
            raise RuntimeError(f"{type(self).__name__} readout failed on initial state: {e}") from e

        assert isinstance(sample_readout, Tensor) and sample_readout.is_floating_point(), (
            f"{type(self).__name__} readout must return a floating point tensor"
        )

        assert sample_readout.shape == (1, 1, dim_readout), (
            f"{type(self).__name__} readout returned shape {sample_readout.shape}, expected (1, 1, {dim_readout})"
        )

        # 3. validate each action

        for name, fn, needs_item in zip(action_names, action_fns, takes_item):
            original_leaves = [l.clone() for l in state_leaves]
            cloned_state = inverse([l.clone() for l in state_leaves])
            args = (cloned_state, sample_inputs) if needs_item else (cloned_state,)

            try:
                next_state = fn(*args)
            except Exception as e:
                raise RuntimeError(f"action '{name}' failed during compile dry-run: {e}") from e

            assert exists(next_state), (
                f"action '{name}' returned None. Did you forget to return the next state?"
            )

            # check for in-place mutation

            current_input_leaves, _ = tree_flatten_with_inverse(cloned_state)
            input_mutated = any(not torch.equal(orig, curr) for orig, curr in zip(original_leaves, current_input_leaves))

            assert not input_mutated, (
                f"action '{name}' mutated the input state in-place. "
                f"In-place modifications corrupt candidate states in superposition."
            )

            # check PyTree structure and shapes

            next_leaves, _ = tree_flatten_with_inverse(next_state)

            assert len(next_leaves) == len(state_leaves), (
                f"action '{name}' returned {len(next_leaves)} tensor leaves, expected {len(state_leaves)}"
            )

            next_shapes = [tuple(l.shape) for l in next_leaves]

            assert next_shapes == state_shapes, (
                f"action '{name}' returned shapes {next_shapes}, expected {state_shapes}"
            )

            assert all(isinstance(l, Tensor) and l.is_floating_point() for l in next_leaves), (
                f"action '{name}' returned non-floating point tensor leaves"
            )

    def _collect_actions(self) -> dict[str, Callable]:
        if exists(self._actions):
            return dict(self._actions)

        actions = {}
        for klass in reversed(type(self).__mro__):
            for name, fn in vars(klass).items():
                if getattr(fn, '_is_action', False):
                    actions[name] = getattr(self, name)

        return actions

    def _collect_readout(self) -> Callable | None:
        readout = self._readout

        if not exists(readout):
            readout = next((
                getattr(self, name)
                for klass in type(self).__mro__
                for name, fn in vars(klass).items()
                if getattr(fn, '_is_readout', False)
            ), None)

        if not exists(readout):
            return None

        # maybe compose with transform on readout of state

        if not exists(self._readout_transform):
            return readout

        transform = self._readout_transform
        return lambda state: transform(readout(state))

    def _resolve_dims(self, init_state_fn: Callable, readout_fn: Callable) -> tuple[int | None, int | None]:
        dim_inputs, dim_readout = self.dim_inputs, self.dim_readout

        # early return if dimensions already explicitly set

        if exists(dim_inputs) and exists(dim_readout):
            return dim_inputs, dim_readout

        # infer from sample state and readout

        try:
            param = next(self.parameters(), None)
            device = param.device if exists(param) else 'cpu'
            dtype = param.dtype if exists(param) else torch.float32

            sample_state = init_state_fn(1, 1, device, dtype)
            sample_readout = readout_fn(sample_state)
            assert isinstance(sample_readout, Tensor)

            dim_readout = default(dim_readout, sample_readout.shape[-1])

            if isinstance(sample_state, Tensor):
                dim_inputs = default(dim_inputs, sample_state.shape[-1])

        except Exception:
            pass

        dim_inputs = default(dim_inputs, dim_readout)
        return dim_inputs, dim_readout

def data_structure(
    init: Callable,
    actions: dict[str, Callable],
    readout: Callable,
    *,
    readout_transform: Callable | None = None,
    render: Callable | None = None,
    dim: int | None = None,
    dim_inputs: int | None = None,
    dim_readout: int | None = None,
    vmap: bool = False
) -> DataStructure:
    return DataStructure(
        init = init,
        actions = actions,
        readout = readout,
        readout_transform = readout_transform,
        render = render,
        dim_inputs = default(dim_inputs, dim),
        dim_readout = default(dim_readout, dim),
        vmap = vmap
    )

# standard data structures

def push(stack, item):
    return cat((item.unsqueeze(-2), stack[..., :-1, :]), dim = -2)

def pop(stack):
    return cat((stack[..., 1:, :], torch.zeros_like(stack[..., :1, :])), dim = -2)

class Stack(DataStructure):
    def __init__(self, depth = 16, dim = 16, **kwargs):
        super().__init__(dim_inputs = dim, dim_readout = dim, **kwargs)
        self.depth = depth
        self.dim = dim

    def init_state(self):
        return torch.zeros(self.depth, self.dim)

    @action
    def push(self, stack, item):
        return push(stack, item)

    @action
    def pop(self, stack):
        return pop(stack)

    @action
    def noop(self, stack):
        return stack

    @readout
    def top(self, stack):
        return stack[..., 0, :]

    def render(self, stack):
        norm = stack.norm(dim = -1)
        depth = min((norm > 1e-3).sum().item(), self.depth)
        bar = '#' * depth + '.' * (self.depth - depth)
        return f"[{bar}] depth={depth}"

class Queue(DataStructure):
    def __init__(self, size = 16, dim = 16, **kwargs):
        super().__init__(dim_inputs = dim, dim_readout = dim, **kwargs)
        self.size = size
        self.dim = dim

    def init_state(self):
        buf = torch.zeros(self.size, self.dim)
        head = torch.zeros(self.size)
        tail = torch.zeros(self.size)
        head[0] = 1.
        tail[0] = 1.
        return buf, head, tail

    @action
    def noop(self, state):
        return state

    @action
    def enqueue(self, state, item):
        buf, head, tail = state
        buf_next = buf.lerp(item.unsqueeze(-2), tail.unsqueeze(-1))
        return buf_next, head, tail.roll(1, dims = -1)

    @action
    def dequeue(self, state):
        buf, head, tail = state
        return buf, head.roll(1, dims = -1), tail

    @readout
    def front(self, state):
        buf, head, tail = state
        return einsum(buf, head, '... s d, ... s -> ... d')

    def render(self, state):
        buf, head, tail = state
        h = head.argmax(dim = -1).item()
        t = tail.argmax(dim = -1).item()
        return f"head@{h}, tail@{t}"

class TuringTape(DataStructure):
    def __init__(self, length = 16, dim = 16, **kwargs):
        super().__init__(dim_inputs = dim, dim_readout = dim, **kwargs)
        self.length = length
        self.dim = dim

    def init_state(self):
        tape = torch.zeros(self.length, self.dim)
        head = torch.zeros(self.length)
        head[self.length // 2] = 1.
        return tape, head

    @action
    def noop(self, state):
        return state

    @action
    def left(self, state):
        tape, head = state
        return tape, head.roll(-1, dims = -1)

    @action
    def right(self, state):
        tape, head = state
        return tape, head.roll(1, dims = -1)

    @action
    def write(self, state, item):
        tape, head = state
        return tape.lerp(item.unsqueeze(-2), head.unsqueeze(-1)), head

    @readout
    def read(self, state):
        tape, head = state
        return einsum(tape, head, '... l d, ... l -> ... d')

    def render(self, state):
        tape, head = state
        pos = head.argmax(dim = -1).item()
        tape_vis = ''.join(['^' if i == pos else '.' for i in range(self.length)])
        return f"[{tape_vis}] pos={pos}"

class DualStack(DataStructure):
    def __init__(self, depth = 16, dim = 16, **kwargs):
        super().__init__(dim_inputs = dim, dim_readout = 2 * dim, **kwargs)
        self.depth = depth
        self.dim = dim

    def init_state(self):
        call_stack = torch.zeros(self.depth, self.dim)
        eval_stack = torch.zeros(self.depth, self.dim)
        return call_stack, eval_stack

    @action
    def noop(self, state):
        return state

    @action
    def push_eval(self, state, item):
        call_stack, eval_stack = state
        return call_stack, push(eval_stack, item)

    @action
    def pop_eval(self, state):
        call_stack, eval_stack = state
        return call_stack, pop(eval_stack)

    @action
    def push_call(self, state, item):
        call_stack, eval_stack = state
        return push(call_stack, item), eval_stack

    @action
    def pop_call(self, state):
        call_stack, eval_stack = state
        return pop(call_stack), eval_stack

    @action
    def eval_to_call(self, state):
        call_stack, eval_stack = state
        return push(call_stack, eval_stack[..., 0, :]), pop(eval_stack)

    @action
    def call_to_eval(self, state):
        call_stack, eval_stack = state
        return pop(call_stack), push(eval_stack, call_stack[..., 0, :])

    @readout
    def top(self, state):
        call_stack, eval_stack = state
        return cat((call_stack[..., 0, :], eval_stack[..., 0, :]), dim = -1)

    def render(self, state):
        call_stack, eval_stack = state
        c_depth = (call_stack.norm(dim = -1) > 1e-3).sum().item()
        e_depth = (eval_stack.norm(dim = -1) > 1e-3).sum().item()
        return f"call={c_depth}, eval={e_depth}"

class RegisterFile(DataStructure):
    def __init__(self, num_registers = 8, dim = 16, **kwargs):
        super().__init__(dim_inputs = dim, dim_readout = dim, **kwargs)
        self.num_registers = num_registers
        self.dim = dim

    def init_state(self):
        regs = torch.zeros(self.num_registers, self.dim)
        ptr = torch.zeros(self.num_registers)
        ptr[0] = 1.
        return regs, ptr

    @action
    def noop(self, state):
        return state

    @action
    def next(self, state):
        regs, ptr = state
        return regs, ptr.roll(1, dims = -1)

    @action
    def prev(self, state):
        regs, ptr = state
        return regs, ptr.roll(-1, dims = -1)

    @action
    def write(self, state, item):
        regs, ptr = state
        return regs.lerp(item.unsqueeze(-2), ptr.unsqueeze(-1)), ptr

    @readout
    def read(self, state):
        regs, ptr = state
        return einsum(regs, ptr, '... r d, ... r -> ... d')

    def render(self, state):
        regs, ptr = state
        pos = ptr.argmax(dim = -1).item()
        return f"reg@{pos}/{self.num_registers}"
