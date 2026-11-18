# Top-64 neurons on non-children texts, and their values (key-value read)

Run 2026-11-18. Configs `configs/adult_probe_l25.yaml` (GPU, L4, 2 min) and `configs/kv_values_l25.yaml` (CPU).
Outputs: `results/adult_probe_l25/summary.md` (hits.json, 31 MB, at
`https://narang-public-s3.s3.amazonaws.com/narang99-pages/emotion-neurons/results/adult_probe_l25/hits.json`),
`results/kv_values_l25/values.{json,md}`, `results/mode_emotion_share.json`.

## Setting

- **Why:** 92–96% of the hits in emotion-labelled modes came from TinyStories; so do 92% of the harvest's feeling
  words. Are these neurons about emotion, or about children's stories?
- **Texts:** 60 new ones by Haiku (`data/adult_stories/`), 5 per emotion: fiction, forum post, letter, journal,
  fiction. Idx 0–2 name the emotion, 3–4 never do. Adult lives (work, money, illness, divorce), no children's openers.
- **Hit:** activation >= the neuron's harvest p97. **Mode:** nearest mode centre, rebuilt from the 5 exemplars.
- **Values:** `W_down[:, n]` through the final RMSNorm gain and the unembedding (logit lens; layer 25 of 36).

## Log

- **Round 1: emotion share of the modes, by neuron.** 207 neurons have Haiku-named modes; only 84 share a prompt.
  - Top 64: 32 have < 10% of hits in emotion modes (23 at 0%). Controls: 14 of 20 at 0%.
  - Stage 3's 123 others used an older prompt with no "emotion-general" label (it called "'and' right after ashamed"
    "none"), so their shares undercount.
- **Round 2: do emotion-mode neurons fire on feeling words outside TinyStories?** Yes, less often.
  - 24 neurons with >= 25% emotion hits, 30 feeling words, word or next token >= p97: TinyStories 20% (any token
    5.4%), FineWeb-edu 9% (1.6%), UltraChat 11% (2.1%). Lift over the corpus rate is as high outside TinyStories.
- **Round 3: the same neurons on the 60 adult texts.** Several carry the emotion in adult register; others are
  generic.
  - 3068: "pride" / "proud" up to 38x p97; 22 of its 30 strongest hits are in proud texts.
  - 11109: bodily anxiety, "my [heart] races", "my hands [get] clammy", up to 50x p97; 22 of 30 in nervous/afraid
    texts. In the harvest its mode was "the token after felt/was naming fear".
  - 2862 (happy #1): "could not stop [laughing]", "we both [laughed]", up to 50x p97.
  - 48: "and" / "then" / "'ve" in guilty texts (13 of 30 strongest), not after feeling words.
  - 3361 fires on 29% of all tokens (pronouns, "that"); 5869 on function words in calm/sad texts: generic here.
  - Hit rate 7.5% of tokens for the 64 (p97 is from general text, so 3% expected there).
  - Mode assignment from 5 exemplars is too strict to use: hits sit at median cosine 0.38 to the nearest centre,
    thresholds median 0.56; 2% of hits pass. Needs full cluster centres (rerun modes.py and save them).
- **Round 4: values (Geva et al. 2021).** Several emotion neurons are key-value memories for their concept; next-token
  agreement is at chance.
  - 3068 promotes " pride", 自豪, " proud", 骄傲; 2862 " laughing", " laughter", 笑; 11109 " cortisol", " adrenal",
    " stress", 心跳 (heartbeat); 48 " avoiding", 回避, " avoidance".
  - Feeling words' mean percentile under the value (0 = top, 0.5 = chance): emotion-mode neurons 0.41 (min 0.14),
    other top-64 0.51, controls 0.50.
  - Geva's next-token test: the token after a hit sits at percentile 0.500 under the neuron's own value vs 0.509
    under other neurons'; controls the same. The values write the concept, not the next token.
  - Whole writes: proud promotes " proud", 骄傲, " pride"; happy " hugged", 拥抱, " handshake"; afraid 心跳,
    " nerves", " breathing"; sad and angry read as noise.

## Surprises

- **Key and value match for the clean neurons:** 3068 reads "pride" and writes pride; 2862 reads laughter and writes
  laughter; 11109 reads a racing heart and writes cortisol / adrenaline / heartbeat.
- **48 writes "avoid"** and fires most in guilt texts.
