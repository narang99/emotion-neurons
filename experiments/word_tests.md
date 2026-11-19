# Emotion words or situations, layer by layer

Date: 2026-11-19. Branch: `stack/05-causal-sets`. Tag: `exp/word_tests`. Status: kept.

Config: `configs/word_tests.yaml`. Results: `results/word_tests/word_tests.json`.

```
python scripts/word_tests.py --config configs/word_tests.yaml   # L4, 3 minutes
```

## Question

- In early layers, does the emotion vector follow emotion words? In later layers, does it follow the situation?

## Method

- Text sets, each balanced over the 12 emotions:
  - `heldout`: 240 held-out training stories. Only 14 of them contain one of the 12 label words;
  - `explicit`: 36 adult texts that name the emotion;
  - `implied`: 24 adult texts that do not name it (1 of them has a label word);
  - `mislead`: 24 texts that I wrote (2 per emotion), in `data/word_tests/mislead.jsonl`.
    - The character feels the target emotion. The target word is absent.
    - A different "distractor" emotion word appears 3 or more times: negated, expected, or about another person.
- Story level: the mean stream over tokens 50+, centred on its set's mean, projected on each layer's 12 unit vectors.
- Token level: the stream at each token of the distractor word (75 tokens), centred on the mean over all tokens.

## Results

- **Story level** (accuracy at layers 0, 9, 18, 27, 33):

  | set | L0 | L9 | L18 | L27 | L33 |
  |---|---|---|---|---|---|
  | heldout | 0.64 | 0.77 | 0.84 | 0.78 | 0.75 |
  | explicit | 0.33 | 0.50 | 0.56 | 0.69 | 0.67 |
  | implied | 0.25 | 0.54 | 0.58 | 0.58 | 0.62 |
  | mislead | 0.75 | 0.92 | 0.92 | 1.00 | 0.96 |

  - At story level, the mislead texts are labelled as the distractor in 0 to 4% of cases at every layer.
  - The mean over a whole text dilutes the 3 distractor words. The situation words (cried, grinning, heart hammered) dominate even at layer 0.
  - Thus the story-level test cannot separate words from situations.
- **Token level, at the distractor word:**
  - Labelled as the distractor: 0.24 at layer 0, about 0.50 at layers 9-30, 0.39 at layer 33.
  - Labelled as the true emotion: 0.04 to 0.09 at all layers.
  - At "was not **afraid**", the vector reads "afraid", more in the middle layers than at layer 0. The negation does not move it to the true emotion.
- The model's own answer is correct for 0.92 of the mislead texts. It never gives the distractor.

## Conclusion

- At the word itself, the emotion vector follows the word, not the situation, at all depths. The negation is not resolved along the vector.
- The model still answers correctly. Thus the context is used somewhere else (other directions, or later tokens).

## Limits

- The mislead texts are short and written by me, and their situations are strong. 24 texts only.
- The layer-0 vector comes from children's stories. It is weaker on adult texts.
