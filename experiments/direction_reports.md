# Three directions of the layer-25 emotion write: main, modes, remaining

Date: 2026-11-20. Branch: `stack/05-causal-sets`. Tag: `exp/direction_reports`. Status: kept.

Config: `configs/direction_reports.yaml`. Results: `results/direction_reports/directions.json`, and the full report with
example sentences: `results/direction_reports/report.md`.

```
python scripts/direction_reports.py --config configs/direction_reports.yaml   # L4, about 8 minutes
```

## Question

- The tail check splits each emotion's write into the sparse set's tail firings and the rest. Do these two parts point in different directions?
- Do they mean different things?

## Method

- Per emotion (happy, sad, afraid, proud), three directions in the space MLP-25 writes into:
  - `main`: the whole emotion write `delta_e = W_down dA_e`;
  - `modes`: the part from the sparse set's tail firings (tokens >= p97 or <= p3);
  - `remaining`: main minus modes.
- Per direction:
  - its norm, and the cosines between the three directions;
  - the logit lens;
  - the tokens where MLP-25's own output projects most on it (240 held-out stories and 60 adult texts);
  - example texts with the top 5% of tokens marked (in `report.md`).

## Results

- **Shape.** `modes` and `remaining` have about the same norm. They point in quite different directions:

  | emotion | norm main / modes / remaining | cos main-modes | cos main-remaining | cos modes-remaining |
  |---|---|---|---|---|
  | happy | 4.41 / 2.82 / 2.77 | 0.79 | 0.79 | 0.25 |
  | sad | 3.44 / 1.72 / 2.38 | 0.77 | 0.89 | 0.39 |
  | afraid | 4.78 / 2.73 / 3.04 | 0.81 | 0.85 | 0.37 |
  | proud | 3.37 / 1.56 / 2.61 | 0.66 | 0.89 | 0.26 |

- **Logit lens: `modes` reads as clear concepts. `remaining` mostly does not.**
  - happy:
    - modes: " hugged", 拥抱 (hug), " handshake", " hugs", 握手 (handshake);
    - remaining: " proudly", 挖掘 (dig), 展开 (unfold), 尽管 (although), mixed.
  - afraid:
    - modes: 心跳 (heartbeat), 怦 (thump), 心动, " nerves", " trem";
    - remaining: " miles", " collapse", " fur", mixed.
  - proud:
    - modes: 自豪 (pride), " pride", " proud", 骄傲 (pride), " proudly", 荣誉 (honour);
    - remaining: " inclusive", " salts", " instant", mixed.
  - sad: no clear read for either. modes: 葡萄 (grape), 送去 (send), 看望 (visit); remaining: 聚集 (gather), 相聚 (reunite).
- **Where MLP-25 writes most along each direction: both are emotional contexts, but of different kinds.**
  - afraid:
    - modes: "My **heart** started pounding", "my stomach **is** in knots": the heart and the stomach;
    - remaining: "Her whole body **was** shaking", "Her pulse **was** roaring", "Her breath **was** coming quick": the body state as a whole.
  - proud:
    - modes: "pulled me into **a** hug", "reached across the desk **and** shook his hand", "squeezed my **arm**": physical gestures;
    - remaining: "and **said** nothing at all", "**shouted**", "he **said** it was the shini...": speech acts.
  - happy:
    - both peak on laughing ("cack**ling**", "**laughed**").
    - modes adds hugs ("pulled **me** into a hug"). remaining adds "I **burst** out laughing".
  - sad: both peak on "**sat** at the table", "**looked** at the empty chair".

## Conclusion

- The tail firings of the sparse set (`modes`) write a direction that the vocabulary can read: hugs, heartbeat, pride words.
- The rest of the write (`remaining`) is as large, but points elsewhere (cosine 0.25-0.39). Its vocabulary read is not clear.
  - It still peaks in emotional contexts, often on other cues of the same scene (speech, whole-body states).
- Thus the feature-like part and the diffuse part of the write carry different information. Neither one alone is "the" emotion write.

## TODO

- Find the modes of the neurons behind `remaining`, to see what is happening in it.
  - Today `remaining` is read only as a vector (logit lens, top-projecting tokens). No neuron-level read exists.
  - Pick the neurons that contribute most to `remaining` per emotion (outside the sparse set, plus the set's middle tokens).
  - Harvest their p97+ hits, cluster into modes and name them, as in Stage 4 / 4b.
  - Compare with the `modes` neurons: does `remaining` hold a few clear neurons, or many weak ones?

## Limits

- 4 emotions. The logit lens at layer 25 is only a rough read.
- `remaining` mixes the set's middle tokens with all non-set neurons.
