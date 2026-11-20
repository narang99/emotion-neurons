"""Can neurons at their poles alone build the emotion direction? Per-token greedy with restricted candidates. GPU.

    python scripts/raw_pools.py --config configs/raw_pools.yaml

- Same tokens and terms as `raw_sets.py`: each emotion's highest-scoring held-out tokens, raw `a_n(t) w_n`.
- Candidate pools per token:
  - `all`: every neuron;
  - `poles`: neurons at a pole at that token (<= p3 or >= p97 of general text);
  - `middle`: every other neuron;
  - `middle_sized`: a random subset of `middle` with as many neurons as `poles` (a fair size control).
- Per pool: the cosine of the whole pool's sum, the best greedy cosine, and the set size at each threshold.
  Toward the emotion vector and toward a random direction.
- The corpus quantiles and the token activations are cached in runs/<run_id>/cache.safetensors.
- Writes results/<run_id>/raw_pools.json.
"""

import json
import statistics
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from emotion_neurons.config import load_run_config
from emotion_neurons.decomposition.raw_greedy import first_reaching, raw_greedy
from emotion_neurons.emotions import EMOTIONS
from emotion_neurons.models import decoder_layers, load_model
from emotion_neurons.texts import load_stories

torch.set_grad_enabled(False)
sys.path.insert(0, "scripts")
from raw_sets import story_reads, top_tokens  # noqa: E402
from tail_check import corpus_quantiles  # noqa: E402

POOLS = ("all", "poles", "middle", "middle_sized")


def build_cache(model, tokenizer, cfg) -> dict[str, torch.Tensor]:
    """p3 / p97 per neuron, and per emotion the activations at its top tokens."""
    rs = cfg.raw_sets
    Q, _ = corpus_quantiles(model, tokenizer, cfg, [0.03, 0.97])
    units = load_file(str(rs.layer_vectors_file))["unit"][cfg.layer].float()
    stories = [s for s in load_stories(cfg.data) if s.heldout and s.emotion in EMOTIONS]
    reads = story_reads(model, tokenizer, stories, cfg.layer, cfg.data.split_token)
    centre = torch.cat([rd["r"].float() for rd in reads]).mean(0)
    cache = {"p3": Q[0], "p97": Q[1]}
    for e, name in enumerate(EMOTIONS):
        own = [rd for rd in reads if rd["story"].emotion == name]
        picks = top_tokens(own, centre, units[e], rs.tokens_per_emotion, rs.per_story)
        cache[f"acts/{name}"] = torch.stack([own[i]["a"][p] for _, i, p in picks])
    return cache


def pools(a, p3, p97, generator) -> dict[str, torch.Tensor]:
    poles = (a <= p3) | (a >= p97)
    middle = ~poles
    sized = torch.zeros_like(poles)
    for i in range(len(a)):
        idx = middle[i].nonzero().squeeze(1)
        keep = idx[torch.randperm(len(idx), generator=generator)[: int(poles[i].sum())].to(idx.device)]
        sized[i, keep] = True
    return {"all": torch.ones_like(poles), "poles": poles, "middle": middle, "middle_sized": sized}


def pool_row(W, WtW, a, d, allowed, steps: int, thresholds: list[float]) -> dict:
    d_hat = d / d.norm()
    pool_sum = (a * allowed) @ W.T
    pool_cos = pool_sum @ d_hat / pool_sum.norm(dim=1)
    run = raw_greedy(W, WtW, a, d_hat.expand(len(a), -1), steps, allowed)
    best, k_best = run.cos.max(1)
    row = {
        "pool_size": float(allowed.sum(1).float().median()),
        "pool_sum_cos": float(pool_cos.mean()),
        "best_cos": float(best.mean()),
        "best_cos_q25_q75": [float(x) for x in torch.quantile(best, torch.tensor([0.25, 0.75], device=best.device))],
        "k_at_best": float((k_best + 1).float().median()),
        "at": {},
    }
    for thr in thresholds:
        k = first_reaching(run.cos, thr)
        reached = k[k > 0].float()
        row["at"][str(thr)] = {
            "reached": float((k > 0).float().mean()),
            "k_median": float(reached.median()) if len(reached) else None,
        }
    return row


def main():
    cfg = load_run_config(sys.argv[1:])
    rs = cfg.raw_sets
    t0 = time.time()
    model, tokenizer = load_model(cfg.model, cfg.device)
    cache_file = cfg.runs_dir / cfg.run_id / "cache.safetensors"
    if cache_file.exists():
        cache = load_file(str(cache_file))
    else:
        cache = build_cache(model, tokenizer, cfg)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in cache.items()}, str(cache_file))
    print(f"[cache] {time.time() - t0:.0f}s", flush=True)

    W = decoder_layers(model)[cfg.layer].mlp.down_proj.weight.float()
    WtW = W.T @ W
    units = load_file(str(rs.layer_vectors_file))["unit"][cfg.layer].float().to(cfg.device)
    p3, p97 = cache["p3"].to(cfg.device), cache["p97"].to(cfg.device)
    generator = torch.Generator().manual_seed(rs.seed)
    out = {"pools": POOLS, "emotions": {}}
    for e, name in enumerate(EMOTIONS):
        a = cache[f"acts/{name}"].float().to(cfg.device)
        candidates = pools(a, p3, p97, generator)
        random_dir = torch.randn(W.shape[0], generator=generator).to(cfg.device)
        row = {}
        for target, d in (("emotion", units[e]), ("random", random_dir)):
            row[target] = {p: pool_row(W, WtW, a, d, candidates[p], rs.steps, rs.thresholds) for p in POOLS}
        out["emotions"][name] = row
        em, rn = row["emotion"], row["random"]
        print(f"[{name}] " + " | ".join(
            f"{p} (n {em[p]['pool_size']:.0f}): sum {em[p]['pool_sum_cos']:+.2f}, best {em[p]['best_cos']:.2f} at k {em[p]['k_at_best']:.0f}"
            f" (random dir best {rn[p]['best_cos']:.2f})" for p in POOLS) + f" | {time.time() - t0:.0f}s", flush=True)  # fmt: skip

    def mean_over(target, pool, key):
        return statistics.mean(out["emotions"][e][target][pool][key] for e in EMOTIONS)

    out["mean"] = {
        t: {p: {k: mean_over(t, p, k) for k in ("pool_size", "pool_sum_cos", "best_cos", "k_at_best")} for p in POOLS}
        for t in ("emotion", "random")
    }
    for t in ("emotion", "random"):
        print(
            f"[mean {t}] "
            + " | ".join(
                f"{p}: sum {v['pool_sum_cos']:+.2f}, best {v['best_cos']:.2f} at k {v['k_at_best']:.0f}"
                for p, v in out["mean"][t].items()
            )
        )
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "raw_pools.json").write_text(json.dumps(out, indent=1))
    print(f"[saved] {cfg.result_dir}/raw_pools.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
