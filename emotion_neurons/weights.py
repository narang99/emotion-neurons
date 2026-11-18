"""One weight matrix from the Hub without downloading the whole shard: a ranged read of its bytes."""

import json
import struct

import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def load_tensor(model: str, name: str) -> torch.Tensor:
    try:
        index = json.loads(open(hf_hub_download(model, "model.safetensors.index.json")).read())
        shard = index["weight_map"][name]
    except EntryNotFoundError:  # a single-file model has no index
        shard = "model.safetensors"
    with HfFileSystem().open(f"{model}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        buf = bytearray(f.read(b - a))
    return torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).float()


def load_w_down(model: str, layer: int) -> torch.Tensor:
    """[d_model, d_mlp], float32."""
    return load_tensor(model, f"model.layers.{layer}.mlp.down_proj.weight")
