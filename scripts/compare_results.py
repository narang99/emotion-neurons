"""Compare a run's output files against a reference run, with a float tolerance. CPU.

    python scripts/compare_results.py NEW_DIR REFERENCE_DIR [FILE ...]

- JSON: structure equal, floats within 1e-7 (relative or absolute).
- safetensors: same keys, max abs diff under 1e-5 (or `--tolerance`).
- anything else: byte-equal.
- Prints one line per file, OK or DIFF with the first differences.
"""

import argparse
import json
import math
from pathlib import Path

from safetensors.torch import load_file


def json_differences(a, b, path: str, out: list[str], tol: float = 1e-7) -> None:
    if isinstance(a, float) or isinstance(b, float):
        numbers = isinstance(a, (int, float)) and isinstance(b, (int, float))
        if not numbers or not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            out.append(f"{path}: {a} vs {b}")
    elif type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}")
        for k in set(a) & set(b):
            json_differences(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            json_differences(x, y, f"{path}[{i}]", out, tol)
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} vs {str(b)[:60]}")


def tensor_differences(a: Path, b: Path, tolerance: float) -> list[str]:
    x, y = load_file(str(a)), load_file(str(b))
    if x.keys() != y.keys():
        return [f"keys differ {sorted(set(x) ^ set(y))[:5]}"]
    worst = max(float((x[k].float() - y[k].float()).abs().max()) for k in x)
    return [] if worst < tolerance else [f"max abs diff {worst}"]


def differences(a: Path, b: Path, tolerance: float) -> list[str]:
    if a.suffix == ".json":
        out: list[str] = []
        json_differences(json.loads(a.read_text()), json.loads(b.read_text()), "", out)
        return out
    if a.suffix == ".safetensors":
        return tensor_differences(a, b, tolerance)
    return [] if a.read_bytes() == b.read_bytes() else ["bytes differ"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    files = args.files or sorted(str(p.relative_to(args.reference)) for p in args.reference.rglob("*") if p.is_file())
    for f in files:
        if not (args.new / f).exists():
            print(f"MISSING {f}")
            continue
        found = differences(args.new / f, args.reference / f, args.tolerance)
        print(f"OK      {f}" if not found else f"DIFF    {f}  ({len(found)}: {' | '.join(found[:2])})")


if __name__ == "__main__":
    main()
