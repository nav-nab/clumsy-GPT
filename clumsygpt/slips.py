"""Decide whether a token gets a slip, and choose the slip word."""

from __future__ import annotations

import torch

from .vocab import LOWER, TITLE, VocabInfo, lemmas, shares_stem


class SlipMaker:
    """Scores candidate words by embedding similarity and samples one.

    Embeddings are mean-centred over the content-word tokens, then normalised, so
    a dot product between two rows is a cosine similarity.
    """

    def __init__(self, embeddings: torch.Tensor, vocab: VocabInfo):
        emb = embeddings.float()
        emb = emb - emb[vocab.content].mean(dim=0)
        self.emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.vocab = vocab
        self._candidate_ids = vocab.content.nonzero().squeeze(1).tolist()
        self._stem_cache = {}
        self._lemma_cache = {}

    def is_eligible(self, discarded: int, previous: int, slip_type: float) -> bool:
        """Whether a Bernoulli trial may run for this sampled token.

        The token must start a word. Each word that is scored against must be a
        content word: the discarded one when slip_type < 1, the previous one
        when slip_type > 0.
        """
        v = self.vocab
        if not v.word_start[discarded]:
            return False
        if slip_type < 1.0 and not v.content[discarded]:
            return False
        if slip_type > 0.0 and not v.content[previous]:
            return False
        return True

    def _lemmas(self, word: str) -> set:
        if word not in self._lemma_cache:
            self._lemma_cache[word] = lemmas(word)
        return self._lemma_cache[word]

    def _variant_mask(self, word: str) -> torch.Tensor:
        """Content-word tokens that are variants of word (cached per word).

        A variant is a word with a similar spelling (salt/salts) or the same
        dictionary form (knew/know). These are inflections of the same word,
        not different words that are associated with it.
        """
        if word not in self._stem_cache:
            mask = torch.zeros(len(self.vocab.content), dtype=torch.bool)
            anchor_lemmas = self._lemmas(word)
            for i in self._candidate_ids:
                other = self.vocab.words[i]
                if shares_stem(word, other) or anchor_lemmas & self._lemmas(other):
                    mask[i] = True
            self._stem_cache[word] = mask
        return self._stem_cache[word]

    def allowed(self, discarded: int, previous: int) -> torch.Tensor:
        """Mask of tokens that may be used as a slip.

        Content words only (whole words that are not function words), with the
        same capitalisation as the discarded token (Title case stays Title
        case, anything else becomes lowercase). The discarded and previous
        words are left out, along with their case variants and inflections.
        """
        v = self.vocab
        wanted = TITLE if int(v.case_class[discarded]) == TITLE else LOWER
        allowed = v.content & (v.case_class == wanted)
        for anchor in (discarded, previous):
            word = int(v.word_index[anchor])
            if word >= 0:
                allowed &= v.word_index != word
                allowed &= ~self._variant_mask(v.words[anchor])
        return allowed

    def choose(
        self,
        logits: torch.Tensor,
        discarded: int,
        previous: int,
        slip_type: float,
        looseness: float,
        top_k: int,
        generator: torch.Generator | None = None,
    ) -> int:
        """Return the id of the slip word that replaces the discarded token.

        The shortlist is the model's top_k candidates among the allowed words.
        Each is scored by (1 - s) * cos(candidate, discarded) + s * cos(candidate, previous),
        where s is slip_type. The scores go through a softmax with temperature
        looseness, and one candidate is sampled.
        """
        allowed = self.allowed(discarded, previous)
        k = min(top_k, int(allowed.sum()))
        if k == 0:
            return discarded
        _, ids = logits.masked_fill(~allowed, float("-inf")).topk(k)

        score = torch.zeros(k)
        if slip_type < 1.0:
            score += (1.0 - slip_type) * (self.emb[ids] @ self.emb[discarded])
        if slip_type > 0.0:
            score += slip_type * (self.emb[ids] @ self.emb[previous])
        probs = torch.softmax(score / looseness, dim=0)
        pick = torch.multinomial(probs, 1, generator=generator)
        return int(ids[pick])
