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

## Two more targets: 2862 (laughter) and 11109 (racing heart)

Configs: `configs/chain_scan_2862.yaml`, `configs/chain_scan_11109.yaml`. Results: `results/chain_scan_{2862,11109}/chain_scan.json`.
Tag: `exp/chain_scans`.

- **2862 (laughter): the same shape as 3068.**
  - Hit words: " laughing" (13), " laughed" (9).
  - Layer recovery: MLP 0 0.91, attention 0 0.74, MLP 2 0.38, MLP 22 0.12.
  - Best single neuron: **L2 n2285**, recovery 0.22 over all 40 hits. It is a candidate at 23 hits. Random L2 neurons: 0.00.
    - n2285 is also in happy's sparse set at layer 2.
  - The best 10 neurons together: patched back, they recover 0.53. Held at their mean, they lower 2862 by 1.85 of 23.5 (a random set: +0.06).
- **11109 (racing heart): spread out.**
  - Its hits are context tokens: " my", " body", " shoulders", " had".
  - Layer recovery: MLP 0 0.94, attention 0 0.48, then MLP 23 0.24 and MLP 22 0.18.
  - No single neuron recovers more than 0.04 (L23 n6155). The anxiety neuron L23 n3524 recovers 0.014.
  - The best 10 neurons together recover 0.17. Held at their mean, they lower 11109 by 1.9 of 15.6 (a random set: -0.03).
- **Pattern:**
  - When a target neuron fires on an emotion word, a short lexical path carries it: the token, MLP 0, then one layer-2 word neuron.
  - When it fires on context, many neurons carry it, mostly in layers 22-23.

## n3068 at context hits (no pride word)

Config: `configs/chain_scan_3068_context.yaml` (`exclude_pattern` drops hits on pride words; threshold p97). Results:
`results/chain_scan_3068_context/chain_scan.json`. Tag: `exp/chain_context`.

- Hit tokens: "," (6), " with" (3), " red", " face", " said" (2 each), " chest", " shoulders", " pleased", " up".
  - Example: "puff[[ing]] up his chest".
- Swapping the hit token still lowers n3068 from 8.05 to 0.94. The hit token sets what the position reads through attention.
- Layer recovery: MLP 0 0.84, attention 0 0.56, MLP 22 0.22, MLP 24 0.18, MLP 5 0.12.
- No compact path:
  - The best single neuron (L24 n8868) recovers 0.036. Random neurons: 0.0005.
  - The best 10 together recover 0.10. Held at their mean, they lower n3068 by only 0.08 of 8.05.
- Thus "pride" from context, as for 11109, is spread over many neurons in layers 22-24. Only "pride" from the word has a short path.
