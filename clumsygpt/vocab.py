"""Classify vocabulary tokens as word-start tokens, whole words and content words."""

import math
import re
from dataclasses import dataclass

import torch
from lemminflect import getAllLemmas
from wordfreq import zipf_frequency

from .config import MIN_WORD_LENGTH, MIN_ZIPF, MIN_ZIPF_SHORT

# Words whose nearest neighbours carry no useful meaning. The list includes the
# stems left behind when a contraction is split ("don" + "'t").
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those my your his her its our their some any no
    every each all both either neither another such much many more most few
    several other own same
    i me we us you he him she it they them myself yourself himself herself
    itself ourselves themselves who whom whose which what whatever whoever one
    of in on at by for with about against between into through during before
    after above below to from up down out off over under onto upon within
    without across along around among behind beside beyond near toward towards
    via per
    and but or nor so yet if then than because while although though unless
    until since whether as when where why how here there once again further
    also just only very too quite rather really even still already always never
    ever often sometimes maybe perhaps well now soon yes no not
    am is are was were be been being have has had having do does did doing done
    will would shall should can could may might must ought let
    don doesn didn isn aren wasn weren haven hasn hadn won wouldn shouldn
    couldn mustn needn ll ve re
    oh uh um er hmm yeah hey okay ok please
    """.split()
)

_WORD_START = re.compile(r" [A-Za-z]+")

# Capitalisation classes of a word.
LOWER, TITLE, OTHER = 0, 1, 2


def case_class(word: str) -> int:
    """LOWER ("salt"), TITLE ("Salt") or OTHER ("SALT", "I", "McDonald")."""
    if word.islower():
        return LOWER
    if len(word) > 1 and word[0].isupper() and word[1:].islower():
        return TITLE
    return OTHER


def shares_stem(a: str, b: str) -> bool:
    """Whether two lowercase words are close surface variants of each other.

    True in either of two cases. They share a prefix of at least three letters
    that covers all but at most the last letter of the shorter word, which
    catches salt/salts/salty. Or they share a long prefix covering three
    quarters of the shorter word, which catches embarrassed/embarrassment. The
    second case is also true when one word is the other with up to three letters
    added at the front, as in able/unable.
    """
    n = min(len(a), len(b))
    common = 0
    while common < n and a[common] == b[common]:
        common += 1
    if common >= max(3, n - 1) or common >= max(4, math.ceil(0.75 * n)):
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and len(longer) - len(shorter) <= 3 and longer.endswith(shorter)


def lemmas(word: str) -> set:
    """The word and its dictionary forms: knew gives {knew, know}, felt gives {felt, feel}."""
    found = {word}
    for options in getAllLemmas(word).values():
        found.update(options)
    return found


@dataclass
class VocabInfo:
    """Per-token information, all of length vocab_size."""

    word_start: torch.Tensor  # bool: a space followed by letters only, e.g. " salt"
    pool: torch.Tensor  # bool: word-start tokens that are whole words
    content: torch.Tensor  # bool: pool tokens that are not function words
    word_index: torch.Tensor  # long: id of the lowercase word, -1 for other tokens
    case_class: torch.Tensor  # long: LOWER, TITLE or OTHER for word-start tokens, -1 otherwise
    words: list  # the lowercase word for word-start tokens, "" otherwise


def token_strings(tokenizer, n):
    """The text of each of the first n token ids."""
    return [tokenizer.decode([i]) for i in range(n)]


# Chinese, Japanese and Korean text, with their punctuation and full-width forms.
_CJK = re.compile(
    "[⺀-⿟　-ヿ㄀-㆏㐀-䶿一-鿿"
    "가-힯豈-﫿＀-￯\U00020000-\U0002fa1f]"
)


def blocked_tokens(strings, vocab_size):
    """Tokens the model must never produce: those containing Chinese, Japanese or Korean text.

    Also blocked are tokens that decode to U+FFFD. They are pieces of a multi-byte character, and
    the pieces of a Chinese character cannot be told apart from the pieces of an emoji, so the few
    emoji built from several pieces are lost along with them.
    """
    blocked = torch.zeros(vocab_size, dtype=torch.bool)
    for i, text in enumerate(strings[:vocab_size]):
        if "�" in text or _CJK.search(text):
            blocked[i] = True
    return blocked


def build_vocab_info(strings, vocab_size, min_zipf=MIN_ZIPF):
    """Classify every token. Ids beyond len(strings) are treated as not words."""
    word_start = torch.zeros(vocab_size, dtype=torch.bool)
    pool = torch.zeros(vocab_size, dtype=torch.bool)
    content = torch.zeros(vocab_size, dtype=torch.bool)
    word_index = torch.full((vocab_size,), -1, dtype=torch.long)
    cases = torch.full((vocab_size,), -1, dtype=torch.long)
    words = [""] * vocab_size

    index_of = {}
    is_whole_word = {}
    for i, text in enumerate(strings[:vocab_size]):
        if not _WORD_START.fullmatch(text):
            continue
        word = text[1:].lower()
        word_start[i] = True
        word_index[i] = index_of.setdefault(word, len(index_of))
        cases[i] = case_class(text[1:])
        words[i] = word
        if word not in is_whole_word:
            threshold = MIN_ZIPF_SHORT.get(len(word), min_zipf)
            is_whole_word[word] = (
                len(word) >= MIN_WORD_LENGTH and zipf_frequency(word, "en") >= threshold
            )
        if is_whole_word[word]:
            pool[i] = True
            content[i] = word not in FUNCTION_WORDS
    return VocabInfo(word_start, pool, content, word_index, cases, words)
