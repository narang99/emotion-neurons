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

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "emotion-neurons"


def shard_of(model: str, name: str) -> str:
    try:
        index = json.loads(Path(hf_hub_download(model, "model.safetensors.index.json")).read_text())
        return index["weight_map"][name]
    except EntryNotFoundError:  # a single-file model has no index
        return "model.safetensors"


def load_tensor(model: str, name: str) -> Float[Tensor, "..."]:
    with HfFileSystem().open(f"{model}/{shard_of(model, name)}", "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + header_len + start)
        buf = bytearray(f.read(end - start))
    return torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).float()


def load_cached_tensor(model: str, name: str) -> Float[Tensor, "..."]:
    """fp32 in memory, bf16 on disk (the checkpoint's own precision)."""
    path = cache_dir() / f"{model.replace('/', '--')}.{name}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(load_tensor(model, name).bfloat16(), path)
    return torch.load(path).float()


def load_cached_bf16(model: str, name: str) -> Float[Tensor, "..."]:
    """bf16 in memory too: two vocabulary matrices in fp32 need 5 GB."""
    path = cache_dir() / f"{model.replace('/', '--')}.{name}.pt"
    if not path.exists():
        load_cached_tensor(model, name)
    return torch.load(path)


def load_w_down(model: str, layer: int) -> Float[Tensor, "d_model d_mlp"]:
    """Kept as its own fp32 cache file, which earlier runs already wrote."""
    path = cache_dir() / f"{model.replace('/', '--')}.l{layer}.down_proj.pt"
    if path.exists():
        return torch.load(path)
    W = load_tensor(model, f"model.layers.{layer}.mlp.down_proj.weight")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(W, path)
    return W
