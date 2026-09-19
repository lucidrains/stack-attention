## Stack Attention (wip)

For following a line of research that augments attention with a differentiable stack, beginning with DuSell et al. at ETH Zurich

## Install

```bash
$ pip install stack-attention-pytorch
```

## Usage

### StackTransLayer

```python
import torch
from stack_attention import StackTransLayer

tokens = torch.randn(2, 512, 256)

layer = StackTransLayer(256)

out1, state = layer(tokens)

out2, state = layer(
    out1,
    stack_state = state
)

assert out1.shape == out2.shape == tokens.shape
```

## Citations

```bibtex
@misc{dusell2024stackattentionimprovingability,
    title   = {Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns}, 
    author  = {Brian DuSell and David Chiang},
    year    = {2024},
    eprint  = {2310.01749},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2310.01749}, 
}
```

```bibtex
@misc{zhang2025stacktranslargelanguagemodel,
    title    = {StackTrans: From Large Language Model to Large Pushdown Automata Model}, 
    author   = {Kechi Zhang and Ge Li and Jia Li and Huangzhao Zhang and Yihong Dong and Jia Li and Jingjing Xu and Zhi Jin},
    year     = {2025},
    eprint   = {2507.15343},
    archivePrefix = {arXiv},
    primaryClass = {cs.SE},
    url      = {https://arxiv.org/abs/2507.15343}, 
}
```

```bibtex
@inproceedings{joulin2015inferring,
    author    = {Armand Joulin and Tom{\'a}{\v{s}} Mikolov},
    title     = {Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets},
    booktitle = {Advances in Neural Information Processing Systems 28 (NIPS 2015)},
    year      = {2015}
}
```
