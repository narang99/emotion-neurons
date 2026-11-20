# Raw activations: what builds an emotion's MLP write (summary of PR #8)

One page. Details: `signal_noise.md`, `residual_sets.md`.

## The question

- Take layer 25's MLP and one emotion, for example happy. Use raw activations, with no mean removed.
- Is the emotion write a sum of a few signal neurons, with all the others cancelling each other out?
- And are the signal neurons firing at their poles (below p3 or above p97 of general text)?

## The answer, in four lines

1. **About 1900 neurons (15%) carry about 85% of the emotion's push.** No small core of a few hundred does it.
2. **The other 10,400 cancel in every direction except the emotion axis.** There they add the last 15%, spread thinly.
3. **About 7000 neurons add a constant offset.** They lean the same way on every text. They are punctuation, function-word and pronoun neurons. The offset is the same on every story, so it does not tell emotions apart.
4. **The emotion part comes about 70% from pole firings.** These are rare tokens (about 8% of tokens) where a neuron fires far outside its usual range.

## The two experiments behind it

| experiment | what it does | main number |
|---|---|---|
| `signal_noise` | per neuron: is its push zero, the same on every story (offset), or emotion-specific? | 31% zero, 57% offset, 13% specific |
| `residual_sets` | fewest neurons whose raw write rebuilds the emotion write in direction and length; then checks the rest on held-out stories | about 1900 neurons; the rest still add 11% to 23% along the emotion axis |

## Superseded

- `exp/raw_sets`, `exp/raw_pools`: the same question on single tokens, with cosine as the target.
  - Cosine on single tokens fits a random direction about as well as the emotion. So those sets told us little.
  - Their code and logs are kept at the tags only.

## Open

- Neurons near p75: part of the feature, or a separate kind of contribution?
- "A feeling was just named" neurons (48, 2066, 3582): are they in the emotion-specific group?
