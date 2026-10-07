import pytest

from clumsygpt.vocab import LOWER, OTHER, TITLE, build_vocab_info, case_class, lemmas, shares_stem

STRINGS = [" the", " cat", " Cat", " salt", " xqzvt", "ing", ",", " 12", "The", " I", " nurse"]


def test_classification():
    v = build_vocab_info(STRINGS, vocab_size=12)
    # word_start: a space then letters only
    assert v.word_start.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    # pool: word-start tokens that are real words of three or more letters
    # (" xqzvt" is not a word, and " I" is too short)
    assert v.pool.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    # content: pool minus function words (" the")
    assert v.content.tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]


def test_case_variants_share_a_word_index():
    v = build_vocab_info(STRINGS, vocab_size=12)
    assert v.word_index[1] == v.word_index[2]
    assert v.word_index[0] != v.word_index[1]
    assert v.word_index[5] == -1 and v.word_index[6] == -1 and v.word_index[11] == -1


def test_words_and_case_classes():
    v = build_vocab_info(STRINGS, vocab_size=12)
    assert v.words[1] == v.words[2] == "cat"
    assert v.words[5] == "" and v.words[11] == ""
    assert v.case_class.tolist() == [LOWER, LOWER, TITLE, LOWER, LOWER, -1, -1, -1, -1, OTHER, LOWER, -1]


def test_fragments_and_unknown_ids_are_not_words():
    v = build_vocab_info(STRINGS, vocab_size=20)
    assert not v.pool[4] and not v.pool[15]


@pytest.mark.parametrize(
    "word, expected",
    [("salt", LOWER), ("Salt", TITLE), ("SALT", OTHER), ("I", OTHER), ("McDonald", OTHER)],
)
def test_case_class(word, expected):
    assert case_class(word) == expected


@pytest.mark.parametrize(
    "a, b",
    [
        ("salt", "salts"), ("salt", "salty"), ("doctor", "doctors"), ("happy", "happier"),
        ("hot", "hotspot"), ("car", "carry"), ("bread", "breadcrumbs"),
        ("embarrassed", "embarrassment"), ("able", "unable"),
    ],
)
def test_shares_stem(a, b):
    assert shares_stem(a, b) and shares_stem(b, a)


@pytest.mark.parametrize(
    "a, b",
    [
        ("salt", "pepper"), ("cat", "car"), ("dog", "doctor"), ("hot", "heat"),
        ("nurse", "doctor"), ("remember", "memory"), ("boat", "dog"),
    ],
)
def test_does_not_share_stem(a, b):
    assert not shares_stem(a, b)


def test_short_strings_need_a_higher_frequency():
    # Two-letter strings are never words. "ple" is rarer than the three-letter bar; "cat" is not.
    v = build_vocab_info([" ab", " ple", " cat", " nurse"], vocab_size=4)
    assert v.pool.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "a, b",
    [("knew", "know"), ("felt", "feel"), ("told", "tell"), ("said", "says"), ("crying", "cries"), ("went", "go")],
)
def test_irregular_forms_share_a_lemma(a, b):
    assert lemmas(a) & lemmas(b)


@pytest.mark.parametrize("a, b", [("boat", "dog"), ("hot", "cold"), ("remember", "memory")])
def test_different_words_share_no_lemma(a, b):
    assert not lemmas(a) & lemmas(b)
