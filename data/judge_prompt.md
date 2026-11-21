# Judging a neuron from its firing contexts

Each file describes one neuron of a language model. It lists the text where the neuron fires most strongly.
The token where it fires is marked like this: `[[word]]`. Read only what the file shows.

For each neuron, answer:

1. **theme**: what the firing contexts have in common, in at most 8 words. Look at the marked token and the words just before it.
2. **emotion_named**: is the neuron related to the emotion named at the top of the file?
   - 0: no.
   - 1: weakly, or only some contexts.
   - 2: clearly, most contexts.
   - "Related" includes the emotion word itself, its synonyms, and situations, body signs or actions typical of that emotion.
3. **any_emotion**: is it related to any emotion or feeling at all (same 0 / 1 / 2 scale)?
4. **kind**: one of
   - `word`: one word or a few word forms (for example "proud", "pride");
   - `situation`: a kind of scene or event (for example hugging, a loss, a test);
   - `body`: body parts, body signs or posture (for example heartbeat, tears, shoulders);
   - `syntax`: punctuation, function words, pronouns, grammar or formatting;
   - `topic`: a subject area (for example cooking, code, sport);
   - `unclear`: no shared pattern.

Output one JSON object per line, nothing else:

```
{"id": "N000", "theme": "...", "emotion_named": 0, "any_emotion": 0, "kind": "syntax"}
```

Rules:

- Judge each neuron on its own. Do not compare neurons.
- Do not guess beyond the contexts. If there is no shared pattern, say `unclear` with emotion scores 0.
- Do not open any file other than the snippet files you are given.
