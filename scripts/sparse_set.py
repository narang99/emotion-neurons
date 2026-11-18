"""Sparse neuron sets per emotion write: hill-climb the size / direction trade-off. CPU, no forward pass.

    python scripts/sparse_set.py --config configs/sparse_l25.yaml

Setting. delta_e = W_down @ dA_e = sum_n u_n, u_n = W_down[:, n] * dA_e[n] (dA = mean activation on emotion e's
train stories minus the mean over the 12 emotions). Keeping a set S holds every other neuron at its 12-emotion mean,
so the write is s_S = sum_{n in S} u_n. Binary: a neuron is kept as it is or not at all.

Metrics for a set S:
- cos:      cos(s_S, delta_e) on the train stories (the reconstruction);
- along:    s_S . delta_hat / |delta_e|, the share of the write's length that S carries along it;
- rel_err:  |delta_e - s_S| / |delta_e| (>= sin of the angle);
- transfer: the same S on the 20 held-out stories per emotion against the train write, cos(sum_S W[:, n] dA_ho[n],
  delta_e). Noise in the two is independent, so a set that fits story noise gains nothing. The full set gives
  `ceiling` = cos(delta_ho, delta_e); a sparse set can beat it by leaving out noisy neurons.
- target: `raw` (delta_e) or `denoised` (delta_e and every u_n with the neutral PCs of the write projected out).

Search (all moves are exact, from the Gram G = U^T U, so a step is O(|S| * N)):
1. greedy: forward selection, each step adds the neuron that raises cos most.
   crossfit: the same, but the alignment term s . d is taken across halves of the train stories (A/B, random per
   emotion): dA_A[n] (W^T W dA_B)[n] and the mirror, so a neuron's own story noise cannot raise its score; the
   norm is the mean of the halves' own |s|^2.
   shrunkK: greedy against a shrunk target W (dA * max(0, 1 - K/t^2)), t = dA / its standard error over the train
   stories; the set's own write is unchanged, only the selection leans on reproducible shifts.
   Baselines: order by projection c_n, random.
2. swap: at fixed size, best-improvement 1-swaps (one out, one in) until none raises cos.
3. shrink: for a target cos tau, start at the smallest greedy prefix reaching tau; repeatedly drop the neuron whose
   removal costs least, and when cos falls under tau, swap until it is back above tau or no swap helps (then the
   last set above tau is the result).

Writes results/<run_id>/sparse_set.json: per emotion and method, the order and its cos / transfer curves, the
smallest k whose transfer matches the full set, and per tau the smallest greedy and shrunk sets.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import EMOTIONS, load_run_config
from emotion_neurons.weights import load_w_down

torch.set_grad_enabled(False)


class State:
    """A set S with t = U^T s, sd = s . d_hat, ss = |s|^2, kept exact under add / remove."""

    def __init__(self, G, c, S):
        self.G, self.c, self.diag = G, c, G.diagonal()
        self.mask = torch.zeros(G.shape[0], dtype=torch.bool)
        self.mask[S] = True
        idx = torch.tensor(S, dtype=torch.long)
        self.t = G[idx].sum(0) if len(S) else torch.zeros(G.shape[0])
        self.sd = float(c[idx].sum()) if len(S) else 0.0
        self.ss = float(self.t[idx].sum()) if len(S) else 0.0

    @property
    def cos(self):
        return self.sd / max(self.ss, 1e-30) ** 0.5

    def add(self, j):
        self.ss += 2 * float(self.t[j]) + float(self.diag[j])
        self.t += self.G[j]
        self.sd += float(self.c[j])
        self.mask[j] = True

    def remove(self, i):
        self.ss += -2 * float(self.t[i]) + float(self.diag[i])
        self.t -= self.G[i]
        self.sd -= float(self.c[i])
        self.mask[i] = False

    def members(self):
        return self.mask.nonzero().squeeze(1)

    def add_scores(self):
        den = self.ss + 2 * self.t + self.diag
        sc = (self.sd + self.c) / den.clamp_min(1e-30).sqrt()
        return sc.masked_fill(self.mask | (den <= 0), -2)

    def remove_scores(self):
        S = self.members()
        return S, (self.sd - self.c[S]) / (self.ss - 2 * self.t[S] + self.diag[S]).clamp_min(1e-30).sqrt()

    def best_swap(self):
        S = self.members()
        num = self.sd - self.c[S][:, None] + self.c[None, :]
        nrm = (self.ss - 2 * self.t[S] + self.diag[S])[:, None] + (2 * self.t + self.diag)[None, :] - 2 * self.G[S]
        sc = (num / nrm.clamp_min(1e-30).sqrt()).masked_fill_(self.mask[None, :], -2)
        k = int(sc.argmax())
        i, j = int(S[k // sc.shape[1]]), k % sc.shape[1]
        return i, j, float(sc.view(-1)[k])


def greedy(G, c, steps):
    st = State(G, c, [])
    order, cos = [], []
    for _ in range(steps):
        j = int(st.add_scores().argmax())
        st.add(j)
        order.append(j)
        cos.append(st.cos)
    return order, cos


def swap(st: State, max_iter: int, stop_at: float | None = None):
    n = 0
    while n < max_iter and (stop_at is None or st.cos < stop_at):
        i, j, v = st.best_swap()
        if v <= st.cos + 1e-7:
            break
        st.remove(i)
        st.add(j)
        n += 1
    return n


def shrink(G, c, start, tau, max_swaps):
    st = State(G, c, start)
    best = st.members().tolist()
    while True:
        S, sc = st.remove_scores()
        st.remove(int(S[sc.argmax()]))
        if st.cos < tau:
            swap(st, max_swaps, stop_at=tau)
        if st.cos < tau:
            return best
        best = st.members().tolist()


def metrics(W, a, d, a_ho, S):
    idx = torch.tensor(S, dtype=torch.long)
    s, s_ho = W[:, idx] @ a[idx], W[:, idx] @ a_ho[idx]
    cs = torch.nn.functional.cosine_similarity
    return {"k": len(S), "cos": float(cs(s, d, dim=0)), "along": float(s @ d / (d @ d)),
            "rel_err": float((d - s).norm() / d.norm()), "transfer": float(cs(s_ho, d, dim=0))}  # fmt: skip


def curves(W, a, d, a_ho, order):
    """Train cos and transfer of every prefix of `order`."""
    o = torch.as_tensor(order)
    cs = torch.nn.functional.cosine_similarity
    return cs((W[:, o] * a[o]).cumsum(1), d[:, None], dim=0), cs((W[:, o] * a_ho[o]).cumsum(1), d[:, None], dim=0)


def activation_means(cfg):
    """Per-story mean activation (tokens split_token+) and the split: train halves A/B (random, per emotion), held-out."""
    rd = cfg.runs_dir / cfg.neuron_set.vectors_run
    stories = json.loads((rd / "stories.json").read_text())["stories"]
    sums = load_file(str(rd / "sums.safetensors"))
    X = (sums["mlp_act/late"].double() / sums["counts"][:, 1:2].double()).float()
    lab = torch.tensor([EMOTIONS.index(s["emotion"]) if s["emotion"] in EMOTIONS else -1 for s in stories])
    ho = torch.tensor([s["heldout"] for s in stories])
    half = torch.zeros(len(stories), dtype=torch.long)
    gen = torch.Generator().manual_seed(cfg.sparse_set.seed)
    for e in range(len(EMOTIONS)):
        idx = (~ho & (lab == e)).nonzero().squeeze(1)
        half[idx[torch.randperm(len(idx), generator=gen)[len(idx) // 2 :]]] = 1

    def dA(mask):
        mu = torch.stack([X[mask & (lab == e)].mean(0) for e in range(len(EMOTIONS))])
        return mu - mu.mean(0)

    # Standard error of each emotion mean over its train stories (the grand mean's own error is ~1/12 of it).
    se = torch.stack([X[~ho & (lab == e)].std(0) / (~ho & (lab == e)).sum() ** 0.5 for e in range(len(EMOTIONS))])
    return {"train": dA(~ho), "A": dA(~ho & (half == 0)), "B": dA(~ho & (half == 1)), "heldout": dA(ho), "se": se}


def main():
    cfg = load_run_config(sys.argv[1:])
    sp = cfg.sparse_set
    t0 = time.time()
    W = load_w_down(cfg.model, cfg.layer)
    vec = load_file(str(cfg.results_dir / cfg.neuron_set.vectors_run / "vectors.safetensors"))
    acts = activation_means(cfg)
    gap = float((acts["train"] - vec["mlp_act/mean/raw"].float()).abs().max())
    assert gap < 1e-3, f"train dA from the pooled sums does not match Stage 2's vectors ({gap})"
    if sp.target == "denoised":  # remove the neutral PCs of the write from both sides: W -> P W
        V = vec["mlp_out/neutral_pcs"].float()
        W = W - V.T @ (V @ W)
    WtW = W.T @ W
    print(f"[load] W {tuple(W.shape)}, target {sp.target}, {time.time() - t0:.0f}s")
    out = {"config": sp.model_dump(), "emotions": {}}
    emos = [e for e in EMOTIONS if not sp.emotions or e in sp.emotions]
    for emo in emos:
        e = EMOTIONS.index(emo)
        te = time.time()
        a, a_ho, aA, aB = (acts[k][e] for k in ("train", "heldout", "A", "B"))
        d = W @ a
        r = {"ceiling": float(torch.nn.functional.cosine_similarity(W @ a_ho, d, dim=0))}
        orders = {}
        # Plain greedy on the train write.
        G = WtW * a[:, None] * a[None, :]
        c = a * (W.T @ d) / d.norm()
        orders["greedy"], _ = greedy(G, c, sp.greedy_steps)
        # Cross-fit greedy: the alignment s . d taken across halves (dA_A[n] (W^T W dA_B)[n] and the mirror), so a
        # neuron's own story noise cannot raise it; the norm |s|^2 is the mean of the two halves' own norms. That
        # pairing keeps a neuron's score bounded: with the full-train norm, a neuron near zero on the full set but
        # of opposite sign in the halves got a tiny norm and a huge score (and a cross-half Gram is indefinite).
        cx = (aA * (WtW @ aB) + aB * (WtW @ aA)) / 2 / d.norm()
        Gh = WtW * aA[:, None] * aA[None, :]
        Gh += WtW * aB[:, None] * aB[None, :]
        Gh /= 2
        orders["crossfit"], _ = greedy(Gh, cx, sp.greedy_steps)
        del Gh
        # Shrunk-target greedy: select against W (dA * r), r = max(0, 1 - k/t^2) with t = dA / se (James-Stein style),
        # so the target leans on the neurons whose shift is reproducible across stories. The set's write is unchanged.
        t2 = (a / acts["se"][e]) ** 2
        for kappa in sp.shrink_kappas:
            d_s = W @ (a * (1 - kappa / t2).clamp(min=0))
            orders[f"shrunk{kappa:g}"], _ = greedy(G, a * (W.T @ d_s) / d_s.norm(), sp.greedy_steps)
        # Baselines: projection c_n, and significance of dA (|t| over train stories, from the two halves' spread).
        orders["by_c"] = c.argsort(descending=True)[: sp.greedy_steps].tolist()
        orders["random"] = torch.randperm(len(a), generator=torch.Generator().manual_seed(e))[: sp.greedy_steps].tolist()
        r["methods"] = {}
        for m, order in orders.items():
            cos, tr = curves(W, a, d, a_ho, order)
            k_ceiling = next((i + 1 for i, v in enumerate(tr.tolist()) if v >= r["ceiling"]), None)
            kbest = int(tr.argmax()) + 1
            r["methods"][m] = {"order": order, "cos": cos.tolist(), "transfer": tr.tolist(),
                               "k_matches_full": k_ceiling, "best": {"k": kbest, "transfer": float(tr[kbest - 1]), "cos": float(cos[kbest - 1])},
                               "at": {k: {"cos": float(cos[k - 1]), "transfer": float(tr[k - 1])} for k in sp.sizes if k <= len(order)}}  # fmt: skip
        # Smallest set per target train cos (plain greedy, then shrink with swaps).
        r["tau"] = {}
        gcos = r["methods"]["greedy"]["cos"]
        for tau in sp.taus:
            k0 = next((i + 1 for i, v in enumerate(gcos) if v >= tau), None)
            if k0 is not None:
                S = shrink(G, c, orders["greedy"][:k0], tau, sp.max_swaps)
                r["tau"][tau] = {"greedy": metrics(W, a, d, a_ho, orders["greedy"][:k0]),
                                 "shrink": metrics(W, a, d, a_ho, S) | {"neurons": sorted(S)}}  # fmt: skip
        del G
        out["emotions"][emo] = r
        ms = r["methods"]
        print(f"[{emo}] full-set transfer {r['ceiling']:.2f} | k to match it: " + ", ".join(f"{m} {v['k_matches_full']}" for m, v in ms.items())
              + " | best: " + ", ".join(f"{m} {v['best']['transfer']:.2f}@{v['best']['k']}" for m, v in ms.items())
              + " | tau: " + ", ".join(f"{t} {v['greedy']['k']}->{v['shrink']['k']}" for t, v in r["tau"].items())
              + f" | {time.time() - te:.0f}s", flush=True)  # fmt: skip
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "sparse_set.json").write_text(json.dumps(out))
    print(f"[saved] {cfg.result_dir}/sparse_set.json")
    print(f"[total] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
