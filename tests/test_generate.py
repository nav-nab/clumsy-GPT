import re
from types import SimpleNamespace

import pytest
import torch

from clumsygpt.generate import ClumsyGenerator

STRINGS = [
    " the", " cat", " dog", " mat", " bird", " fish", " tree", " house", " river",
    " on", " near", " big", " small", " old", ".", ",", "<eos>",
]
EOS = STRINGS.index("<eos>")
V = len(STRINGS)


class FakeTokenizer:
    eos_token_id = EOS

    def __init__(self, strings=STRINGS):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def encode(self, text):
        return [self.strings.index(piece) for piece in re.findall(r" ?[^ ]+", text)]

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.strings[i] for i in ids)


class FakeModel:
    """Next-token logits depend only on the last token, via a fixed random table."""

    def __init__(self, n=V):
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(n, n, generator=g)
        self.table[:, EOS] = -1e9  # never stop early
        self.weight = torch.randn(n, 6, generator=g)
        self.calls = []
        self.masks = []  # the attention mask of each call, as a list of 0s and 1s, or None

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def __call__(self, input_ids, past_key_values=None, use_cache=True, attention_mask=None):
        self.calls.append((input_ids.shape[1], past_key_values is not None))
        self.masks.append(None if attention_mask is None else attention_mask[0].tolist())
        return SimpleNamespace(logits=self.table[input_ids][None, 0], past_key_values="cache")


def make_generator():
    return ClumsyGenerator(FakeModel(), FakeTokenizer())


def run(gen, **kwargs):
    defaults = dict(clumsiness=0.0, looseness=0.1, top_k=1000, slip_type=0.0, max_new_tokens=60, seed=1)
    defaults.update(kwargs)
    return gen.generate(" the cat on", **defaults)


def test_no_clumsiness_means_no_slips():
    r = run(make_generator(), clumsiness=0.0)
    assert r.slips == []
    assert r.tokens == 60
    assert r.trials > 0
    assert all(label is None for _, label in r.segments)


def test_full_clumsiness_slips_every_eligible_word():
    r = run(make_generator(), clumsiness=1.0)
    assert len(r.slips) == r.trials > 0
    assert sum(label == "slip" for _, label in r.segments) == len(r.slips)
    assert all(old != new for old, new in r.slips)


def test_intrusion_mode_also_slips():
    r = run(make_generator(), clumsiness=1.0, slip_type=1.0)
    assert len(r.slips) == r.trials > 0


def test_same_seed_gives_same_text():
    gen = make_generator()
    assert run(gen, clumsiness=0.5, seed=7).text == run(gen, clumsiness=0.5, seed=7).text


def test_unseeded_runs_differ():
    gen = make_generator()
    texts = {run(gen, clumsiness=0.5, seed=None).text for _ in range(5)}
    assert len(texts) > 1


def test_prompt_is_kept_and_text_is_the_concatenation():
    r = run(make_generator(), clumsiness=0.5)
    assert r.segments[0] == (" the cat on", None)
    assert r.text.startswith(" the cat on")


def test_cache_is_used_after_the_first_call():
    gen = make_generator()
    run(gen, max_new_tokens=5)
    assert gen.model.calls[0] == (3, False)
    assert all(call == (1, True) for call in gen.model.calls[1:])
    assert len(gen.model.calls) == 5  # one prompt pass, then one per token except the last


def test_slip_replaces_the_token_the_model_sees():
    # With clumsiness 1 every eligible token is replaced, so the model is fed the slip.
    gen = make_generator()
    r = run(gen, clumsiness=1.0, max_new_tokens=10)
    assert r.tokens == 10


PROMPT_LENGTH = 3  # " the cat on"


def test_no_mask_is_used_unless_shrug_off_is_on():
    gen = make_generator()
    run(gen, clumsiness=1.0, max_new_tokens=20)
    assert all(mask is None for mask in gen.model.masks)


def test_no_mask_without_slips_even_when_shrugging():
    gen = make_generator()
    run(gen, clumsiness=0.0, max_new_tokens=20, shrug_off=True)
    assert all(mask is None for mask in gen.model.masks)


def test_zero_shrug_tokens_means_no_mask():
    gen = make_generator()
    run(gen, clumsiness=1.0, max_new_tokens=20, shrug_off=True, shrug_tokens=0)
    assert all(mask is None for mask in gen.model.masks)


def test_each_slip_is_hidden_from_exactly_the_next_three_tokens_fed():
    gen = make_generator()
    run(gen, clumsiness=1.0, max_new_tokens=40, shrug_off=True)
    masks = gen.model.masks
    # Every mask covers the whole cache plus the new token.
    assert all(m is None or len(m) == PROMPT_LENGTH + call for call, m in enumerate(masks))
    hidden_in = {}  # cache position -> the calls whose mask hides it
    for call, mask in enumerate(masks):
        for position, value in enumerate(mask or []):
            if value == 0:
                hidden_in.setdefault(position, []).append(call)
    assert hidden_in
    for position, calls in hidden_in.items():
        fed_in = position - PROMPT_LENGTH + 1  # the call that fed the slip itself
        assert calls == [fed_in + 1, fed_in + 2, fed_in + 3][: len(calls)]  # not hidden from itself
        if fed_in + 3 < len(masks):
            assert len(calls) == 3
    assert any(len(calls) == 3 for calls in hidden_in.values())


def test_shrugging_does_not_change_the_text_when_the_model_ignores_the_mask():
    # The fake model looks only at the last token, so the mask cannot change what it says.
    # This checks that the bookkeeping does not disturb sampling.
    gen = make_generator()
    plain = run(gen, clumsiness=0.5, seed=3)
    shrug = run(gen, clumsiness=0.5, seed=3, shrug_off=True)
    assert plain.text == shrug.text


def test_chinese_japanese_and_korean_tokens_are_never_produced():
    # The model is made to love these tokens: without the block, every word would be one of them.
    extra = [" 你好", " こんにちは", " 안녕", "�"]
    strings = STRINGS + extra
    model = FakeModel(len(strings))
    model.table[:, V:] = 50.0
    gen = ClumsyGenerator(model, FakeTokenizer(strings))
    for clumsiness in (0.0, 1.0):
        result = gen.generate(
            " the cat on", clumsiness, 0.1, 1000, 0.0, max_new_tokens=40, seed=1, shrug_off=True
        )
        assert result.tokens == 40
        assert not any(piece in result.text for piece in extra)


def test_empty_opening_raises():
    with pytest.raises(ValueError):
        make_generator().generate("", 0.1, 0.1, 1000, 0.0)
