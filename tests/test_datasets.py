import pytest
import torch

from emotion_neurons.capture import padded, pooled_position_sums, pooled_sums, position_mask
from emotion_neurons.config import HFModel, StoriesData
from emotion_neurons.datasets import StoriesSet
from emotion_neurons.emotions import HOOKED_SITES
from emotion_neurons.models import adapter_for
from emotion_neurons.texts import load_stories

QWEN3 = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN3, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer is not in the local cache")


def test_stories_render_the_tokens_and_positions_of_the_direct_story_path(tokenizer):
    data = StoriesData()
    stories = StoriesSet(data)
    examples = stories.examples()
    assert [e.id for e in examples] == [s.id for s in load_stories(data)]
    adapter = adapter_for(HFModel(name=QWEN3))
    for example in examples:
        rendered = stories.render(example, tokenizer, adapter)
        ids = tokenizer(example.text)["input_ids"]
        assert rendered.input_ids == ids
        assert rendered.positions == list(range(data.split_token, len(ids)))


def test_pooled_sums_over_rendered_positions_match_the_split_token_sums():
    rows = [list(range(7)), list(range(4)), list(range(9))]
    split_token = 3
    ids, mask = padded(rows, pad_id=0)
    late = position_mask([list(range(min(split_token, len(r)), len(r))) for r in rows], ids.shape[1])
    acts = {site: torch.randn(len(rows), ids.shape[1], 5) for site in HOOKED_SITES}
    by_split, counts_split = pooled_sums(acts, mask, split_token)
    by_positions, counts_positions = pooled_position_sums(acts, mask, late)
    assert torch.equal(counts_split, counts_positions)
    for key, v in by_split.items():
        assert torch.equal(v, by_positions[key])
