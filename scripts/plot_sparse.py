"""Size / direction trade-off per emotion from sparse_set.json, and the chosen sets. CPU.

    python scripts/plot_sparse.py --config configs/sparse_l25.yaml --run-id sparse_l25_raw

Writes results/<run_id>/tradeoff.png (train cos and held-out transfer against set size, per method) and
results/<run_id>/sets.json: per emotion, the greedy set at two operating points:
- match: the smallest prefix whose held-out transfer reaches the full set's (generalises as well as all 12288);
- tau0.9: the smallest set found with train cos >= 0.9 (greedy, then shrink with swaps).
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emotion_neurons.config import load_run_config  # noqa: E402

STYLE = {"greedy": ("#1f77b4", "-"), "crossfit": ("#2ca02c", "-"), "shrunk1": ("#9467bd", "-"),
         "by_c": ("#ff7f0e", "--"), "random": ("#7f7f7f", ":")}  # fmt: skip


def main():
    cfg = load_run_config(sys.argv[1:])
    r = json.loads((cfg.result_dir / "sparse_set.json").read_text())
    emos = list(r["emotions"])
    fig, axs = plt.subplots(3, 4, figsize=(18, 12), sharex=True, sharey=True)
    sets = {}
    for ax, emo in zip(axs.flat, emos):
        v = r["emotions"][emo]
        for m, (col, ls) in STYLE.items():
            if m not in v["methods"]:
                continue
            mm = v["methods"][m]
            k = range(1, len(mm["cos"]) + 1)
            ax.plot(k, mm["cos"], color=col, ls=ls, lw=1, alpha=0.45)
            ax.plot(k, mm["transfer"], color=col, ls=ls, lw=1.6, label=m)
        ax.axhline(v["ceiling"], color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"{emo} (full-set transfer {v['ceiling']:.2f})", fontsize=10)
        ax.grid(alpha=0.3)
        g = v["methods"]["greedy"]
        km = g["k_matches_full"]
        t9 = v["tau"].get("0.9")
        sets[emo] = {
            "match": {"k": km, "neurons": g["order"][:km] if km else None,
                      "cos": g["cos"][km - 1] if km else None, "transfer": g["transfer"][km - 1] if km else None},
            "tau0.9": t9["shrink"] if t9 else None,
        }  # fmt: skip
    axs.flat[0].legend(fontsize=8, loc="lower right")
    for ax in axs[-1]:
        ax.set_xlabel("neurons kept (log)")
    for ax in axs[:, 0]:
        ax.set_ylabel("cos (thin: train, thick: held-out transfer)")
    fig.suptitle(f"{cfg.run_id}: keeping k neurons of layer 25's MLP, the rest at their 12-emotion mean. "
                 "Dashed black: transfer of the full write.")  # fmt: skip
    fig.tight_layout()
    fig.savefig(cfg.result_dir / "tradeoff.png", dpi=100)
    (cfg.result_dir / "sets.json").write_text(json.dumps(sets))
    print(f"[saved] {cfg.result_dir}/tradeoff.png, sets.json")
    for emo, s in sets.items():
        t = s["tau0.9"]
        print(f"{emo:9} match k={s['match']['k']} (cos {s['match']['cos']:.2f}) | cos>=0.9 k={t['k'] if t else None} "
              f"(transfer {t['transfer']:.2f})" if t else "")  # fmt: skip


if __name__ == "__main__":
    main()
