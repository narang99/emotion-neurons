"""Model families: how to load a model and its tokenizer, how to wrap an instruction in its chat format, and the
checkpoint names of the tensors read one at a time (`models.weights`).

- `ModelAdapter` is the protocol a family implements. Each adapter holds its own config (`config.ModelConfig`).
  - `hf` (`models.hf.HFAdapter`): Qwen3 and other native transformers models.
  - `qwen1` (`models.qwen1.Qwen1Adapter`): Qwen/Qwen-7B-Chat converted to a native Qwen2 model, ChatML by hand.
- A new family is a config class in the `ModelConfig` union, an adapter class, and one case in `adapter_for`.
- Both families end up with Qwen3-style layer names (model.layers, mlp.down_proj, post_attention_layernorm), so the
  hooks in `models.qwen3` read every family unchanged.
"""

from typing import Literal, Protocol

from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..config import HFModel, ModelConfig, Qwen1Model
from .hf import HFAdapter
from .qwen1 import Qwen1Adapter

TensorKind = Literal["down_proj", "lm_head", "final_norm"]
"""down_proj needs a layer."""


class ModelAdapter(Protocol):
    def load(self, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]: ...

    def load_tokenizer(self) -> PreTrainedTokenizerBase: ...

    def chat_prompt(self, tokenizer: PreTrainedTokenizerBase, instruction: str) -> str: ...

    def tensor_name(self, kind: TensorKind, layer: int | None = None) -> str: ...

    def mlp(self, model: PreTrainedModel, layer: int) -> nn.Module:
        """The layer's MLP. The input of its `down_proj` is the neuron activations, `silu(gate) * up`."""
        ...


def adapter_for(model: ModelConfig) -> ModelAdapter:
    match model:
        case HFModel():
            return HFAdapter(model)
        case Qwen1Model():
            return Qwen1Adapter(model)
