"""Text generation with word slips."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from . import config
from .slips import SlipMaker
from .vocab import blocked_tokens, build_vocab_info, token_strings


@dataclass
class Result:
    segments: list  # (text, label) pairs; label is "slip" for a slip word, else None
    slips: list = field(default_factory=list)  # (intended word, slip word) pairs
    trials: int = 0  # Bernoulli trials run, one per eligible word
    tokens: int = 0  # tokens generated
    done: bool = False  # False for the partial results of a stream, True for the finished one

    @property
    def text(self) -> str:
        return "".join(text for text, _ in self.segments)


class ClumsyGenerator:
    """A causal language model plus the slip mechanism.

    The model and tokenizer are used through a small interface, so tests can
    pass in fakes: model(input_ids=..., past_key_values=..., use_cache=True)
    returning .logits and .past_key_values, model.get_input_embeddings().weight,
    and tokenizer.encode / decode / eos_token_id / len().
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        weights = model.get_input_embeddings().weight.detach().float().cpu()
        strings = token_strings(tokenizer, min(len(tokenizer), weights.shape[0]))
        self.vocab = build_vocab_info(strings, weights.shape[0])
        self.blocked = blocked_tokens(strings, weights.shape[0])
        self.slipper = SlipMaker(weights, self.vocab)

    @classmethod
    def load(cls, model_name=config.MODEL_NAME):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).float().to(device).eval()
        return cls(model, tokenizer, device)

    def _forward(self, ids, past, length=0, hidden=()):
        """Run the model on new token ids; return next-token logits (on the CPU) and the cache.

        length is the number of tokens already in the cache. hidden lists cache positions that
        the new tokens must not attend to. Positions are unchanged: a hidden token still counts
        as part of the sequence, but the new tokens cannot read it.
        """
        input_ids = torch.tensor([ids], device=self.device)
        kwargs = {}
        if hidden:
            mask = torch.ones(1, length + len(ids), dtype=torch.long, device=self.device)
            mask[0, list(hidden)] = 0
            kwargs["attention_mask"] = mask
        out = self.model(input_ids=input_ids, past_key_values=past, use_cache=True, **kwargs)
        logits = out.logits[0, -1].float().cpu()
        # Blocked tokens (Chinese, Japanese and Korean text) can never be chosen, as the model's own
        # word or as a slip: both are drawn from these logits.
        return logits.masked_fill(self.blocked, float("-inf")), out.past_key_values

    def _decode(self, ids):
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def generate(self, *args, **kwargs) -> Result:
        """Continue the prompt and return the finished Result. Arguments are those of stream."""
        result = None
        for result in self.stream(*args, **kwargs):
            pass
        return result

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        clumsiness: float,
        looseness: float,
        top_k: int,
        slip_type: float,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.SAMPLING_TEMPERATURE,
        seed: int | None = None,
        shrug_off: bool = False,
        shrug_tokens: int = config.SHRUG_TOKENS,
    ) -> Iterator[Result]:
        """Continue the prompt, yielding the Result so far after each token. The last one is the finished
        text. A fresh random seed is used unless seed is given.

        With shrug_off, each slip is hidden from the next shrug_tokens tokens that are fed to
        the model. They cannot attend to it, so the slip does not steer them. The token after
        the slip is still chosen from the slip's own output, which cannot be hidden from itself.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("The opening is empty.")
        rng = torch.Generator()
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)

        logits, past = self._forward(prompt_ids, None)
        length = len(prompt_ids)  # tokens in the cache
        previous = prompt_ids[-1]
        generated = []  # (token id, is_slip)
        slips = []
        trials = 0
        shrugging = {}  # cache position of a slip -> how many more tokens must not attend to it

        for step in range(max_new_tokens):
            probs = torch.softmax(logits / temperature, dim=0)
            token = int(torch.multinomial(probs, 1, generator=rng))
            if token == self.tokenizer.eos_token_id:
                break

            is_slip = False
            if self.slipper.is_eligible(token, previous, slip_type):
                trials += 1
                if torch.rand((), generator=rng).item() < clumsiness:
                    new = self.slipper.choose(
                        logits, token, previous, slip_type, looseness, top_k, rng
                    )
                    if new != token:
                        slips.append((self._decode([token]).strip(), self._decode([new]).strip()))
                        token, is_slip = new, True

            generated.append((token, is_slip))
            previous = token
            yield Result(self._segments(prompt, generated), list(slips), trials, len(generated))
            if step < max_new_tokens - 1:
                logits, past = self._forward([token], past, length, tuple(shrugging))
                for position in list(shrugging):
                    shrugging[position] -= 1
                    if shrugging[position] == 0:
                        del shrugging[position]
                if is_slip and shrug_off and shrug_tokens > 0:
                    # The slip has just been fed, and it saw itself. The tokens after it must not.
                    shrugging[length] = shrug_tokens
                length += 1

        yield Result(self._segments(prompt, generated), slips, trials, len(generated), done=True)

    def _segments(self, prompt, generated):
        """Group generated tokens into plain runs and single slip words."""
        segments = [(prompt, None)]
        run = []
        for token, is_slip in generated:
            if is_slip:
                if run:
                    segments.append((self._decode(run), None))
                    run = []
                segments.append((self._decode([token]), "slip"))
            else:
                run.append(token)
        if run:
            segments.append((self._decode(run), None))
        return segments
