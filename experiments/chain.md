# A chain from the word "proud" to the " pride" logit

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tags: `exp/chain-round1` (failed), `exp/chain`. Status: kept.

Configs: `configs/chain.yaml`, `configs/chain_l2.yaml`.
Results:
- `results/chain_round2/chain.json` (L8 and L24 middles);
- `results/chain_l2/chain.json` (L2 and L22 middles);
- `results/chain/chain_scan.json` (scan);
- `results/harvest_l2_named/L2.json` (L2 n8663 on general text).

```
python scripts/chain.py --config configs/chain.yaml --run-id chain_round2   # L4, 5.5 min
python scripts/chain_scan.py --config configs/chain.yaml                    # L4, 7 min
python scripts/chain.py --config configs/chain_l2.yaml                      # L4, 4 min
```

## Question

- Can we follow one emotion signal from an input token, through earlier MLP neurons, to the layer-25 "pride" neuron 3068, and then to a logit?
- Is each link stronger than a random baseline?

## Method

- Hits: the top 40 tokens where n3068 is at or above p99.9 of general text (4.25), over the adult texts and the held-out stories.
- For each candidate middle neuron, at each hit t:
  1. **Exact source split** of its input (`circuits/sources.py`, test on a small model):
     - the token's own embedding;
     - each earlier MLP at t;
     - each head from each source token, with attention frozen.
  2. **Necessity**: hold the middle neuron at its mean at t, and measure the change of n3068.
     Baseline: 10 random neurons of the same layer.
  3. **Token swap**: replace the source token (in practice, the hit word itself) by " thing".
     Baseline: 3 random tokens among the 20 before t.
  4. **Sufficiency**: after the swap, patch the middle neuron back to its clean value. Record how much of n3068's drop comes back.
  5. **Last link**: hold n3068 at its mean, and measure the change of log p(" pride") at t.
- **Scan** (`chain_scan.py`), to find middles without guessing:
  - swap the hit word;
  - patch back one layer's MLP or attention output at t, for every layer;
  - in the 3 best MLP layers, patch back single neurons (the 10 with the largest change at each hit), against 5 random neurons.

## Steps

1. **Round 1 (failed):** guessed middles from the earlier circuit, L8 n4826 and L8 n7265. They are lexical neurons for "proud of himself" and "embarrassed".
   - Holding them moved n3068 by 0.04 and 0.09. Random L8 neurons moved it by 0.10.
2. **Scan:** the token's effect flows through:
   - MLP 0 (recovery 0.86) and attention 0 (0.68);
   - then MLP 2 (0.27), MLP 22 (0.22) and MLP 24 (0.16).
   - Best single neurons: L2 n8663 (0.15 over all hits; random L2 neurons 0.001) and L22 n3221.
3. **Round 2:** the chain with L2 n8663 and L22 n3221.
   - I found a bug (see "Mistakes"), fixed it, and ran all middles again.

## Results (round 2, at the hits where the middle neuron is active, |a| > 1)

| middle | its value reads | active hits | hold at mean: n3068 change | random neurons | word swapped: n3068 | ... with middle patched back |
|---|---|---|---|---|---|---|
| L2 n8663 | 高档 upscale, 富裕 wealthy, " richest" | 9 | +0.17 | 0.05 | -10.29 | **-3.74 (64% back)** |
| L8 n4826 | 自豪 pride, 骄傲 pride | 14 | -0.03 | 0.10 | -9.19 | -9.05 |
| L8 n7265 | 羞 shy, 耻 shame | 12 | -0.24 | 0.10 | -9.48 | -9.30 |
| L22 n3221 | 赢家 winner, 野心 ambition, 红利 bonus | 9 | **-1.06** | 0.04 | -10.29 | -10.13 |
| L24 n2010 | " pride", " proud", 骄傲 | 14 | **-0.96** | 0.03 | -8.19 | -7.93 |

- n3068 at these hits is about 9.2. A random token swap lowers it by 0.3 to 0.8. Swapping the hit word lowers it by 8 to 10.
- **Last link:** holding n3068 at its mean lowers log p(" pride") by 1.01 nats (mean over 40 hits).

### The chain

```
" proud" / " pride"  (input token)
   | embedding -> MLP 0 at the same token (-15.6 into n8663), heads L1h29 and L1h12 (the token attends to itself)
   v
L2 n8663   fires -7 to -12 on " proud" / " pride" (general text: p99.99 = +0.23, so this is a strong negative tail)
   | writes into n3068 (direct effect in the earlier circuit); patched back after the word is removed, it restores
   | 64% of n3068 through all paths (random L2 neuron: 0.1%)
   v
L25 n3068  "pride", about 9 at these hits      <--  L22 n3221 ("winner, ambition"): necessary, -1.06 (26x random)
   |                                           <--  L24 n2010 ("pride", fed by heads L24h15/h14 from context): -0.96 (31x random)
   v
log p(" pride")   -1.01 nats when n3068 is held at its mean
```

## Conclusion

- There is a chain from the input word to the " pride" logit, and every link is much stronger than its random baseline.
- Different links pass different tests:
  - **L2 n8663 is sufficient but not necessary.** Patched back, it restores most of the signal. Removed, it changes little. Other paths carry the same word.
  - **L22 n3221 and L24 n2010 are necessary.** Each one removed costs about 1 of 9.
- The early link is about the word itself (layers 0-2 at the same token). The late links mix in context: n2010 gets most of its input through attention.
- Lexical "pride" and "shame" neurons at L8 fire on the same words but are not on the path.

## Mistakes

- **Round 1 picked middles from the direct writer list.** A large direct write does not mean a causal link. The scan was the right method.
- **Signed source pick.**
  - `top_source` took the source with the largest signed input.
  - L2 n8663 fires negatively, so its own word had a large negative input, and the script swapped the token before it (" so").
  - Round 2 picks by magnitude, and saves the MLP split per layer. All middles were run again.
- **The scan and the chain disagree at first sight.**
  - The scan gives n8663 a mean recovery of 0.15 over 40 hits. The chain gives 64% at its 9 active hits.
  - These agree: 0.64 x 9/40 = 0.14.
- **The log-prob readout under a token swap is not useful.** Replacing the word changes the next-token distribution itself. Only the hold test reads the last link.
