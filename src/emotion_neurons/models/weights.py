"""Single weight matrices from the Hub, without downloading whole shards.

- A safetensors shard starts with an 8-byte header length and a JSON header of byte offsets.
  - So one tensor is a ranged read of its own bytes.
- Tensors are cached under `$HF_HOME/emotion-neurons` (default `~/.cache/huggingface`).
"""

import json
import os
import struct
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from jaxtyping import Float
from torch import Tensor

from .adapters import adapter_for

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "emotion-neurons"


def shard_of(model: str, name: str) -> str:
    revision = adapter_for(model).revision(model)
    try:
        index = json.loads(Path(hf_hub_download(model, "model.safetensors.index.json", revision=revision)).read_text())
    except EntryNotFoundError:  # a single-file model has no index
        return "model.safetensors"
    return index["weight_map"][name]


def load_tensor(model: str, name: str) -> Float[Tensor, "..."]:
    """fp32, read straight from the shard on the Hub (at the model family's pinned revision, if any)."""
    revision = adapter_for(model).revision(model)
    repo = f"{model}@{revision}" if revision else model
    with HfFileSystem().open(f"{repo}/{shard_of(model, name)}", "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + header_len + start)
        buf = bytearray(f.read(end - start))
    return torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).float()


def load_w_down(model: str, layer: int) -> Float[Tensor, "d_model d_mlp"]:
    """Column n is neuron n's write direction `w_n`. Cached as fp32 (the file name earlier runs used)."""
    path = cache_dir() / f"{model.replace('/', '--')}.l{layer}.down_proj.pt"
    if path.exists():
        return torch.load(path)
    W = load_tensor(model, adapter_for(model).tensor_name("down_proj", layer))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(W, path)
    return W
