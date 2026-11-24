"""General-text corpora, streamed from the Hub and cut into token windows."""

from collections.abc import Iterator

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ..config import CorpusConfig


def document_text(row: dict, field: str) -> str:
    """A list field is a chat: messages of {role, content}."""
    value = row[field]
    if isinstance(value, list):
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in value)
    return value


def corpus_windows(
    corpus: CorpusConfig, tokenizer: PreTrainedTokenizerBase, window: int, min_len: int, seed: int
) -> Iterator[tuple[int, list[int]]]:
    """(document id, token ids) until `corpus.tokens` tokens. A window never crosses documents.

    - The first windows, worth `corpus.skip_tokens`, are passed over. With the same seed and window sizes, these are
      exactly the windows of a run that took `skip_tokens` tokens, so the rest is held out from it.
    """
    stream = load_dataset(corpus.path, corpus.config, split=corpus.split, streaming=True).shuffle(seed=seed, buffer_size=1_000)
    skipped = 0
    total = 0
    for doc_id, row in enumerate(stream):
        ids = tokenizer(document_text(row, corpus.field), add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids), window):
            chunk = ids[start : start + window]
            if len(chunk) < min_len:
                break
            if skipped < corpus.skip_tokens:
                skipped += len(chunk)
                continue
            yield doc_id, chunk
            total += len(chunk)
            if total >= corpus.tokens:
                return
