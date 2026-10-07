# ClumsyGPT

Make a language model slip the way people do: occasionally say a word that is closely associated with a neighbouring word, instead of the one that fits.

**Status:** design only, no code yet. This is a qualitative project. There are no metrics. Success means reading the output and finding it convincingly human.

## The idea

People make two kinds of word slip.

- **Substitution:** they say "pass the pepper" when they meant "salt".
- **Intrusion:** the word they just said pulls in a related one, so "nurse" follows "doctor" where it does not belong.

In both cases the slip is related to a nearby word but wrong for the sentence. LLMs almost never do this. ClumsyGPT injects both kinds on purpose and watches how the model recovers.

## How it works

It is a thin layer around ordinary sampling. The transformer is untouched and the KV cache stays intact.

1. The model samples the next token as usual (temperature 0.7).
2. If the token starts a new word, run a Bernoulli trial with probability **c**. If it fails, keep the token.
3. If it passes, discard the token. This is the word the model meant to say. Take the model's **top K** whole-word candidates at that position. Exclude the discarded word and the previous word.
4. Score each candidate as `(1 − s) · cos(candidate, discarded word) + s · cos(candidate, previous word)`. Here `cos` is cosine similarity of token embeddings, and **s** is the slip type. At s = 0 this is pure substitution. At s = 1 it is pure intrusion.
5. Softmax the scores with temperature **τ** and sample one word. That word is the slip.
6. The slip goes back into the context, and the model carries on from it.

The trial is skipped when a word being scored against is a fragment, a function word or punctuation. The neighbours of "the" or "," are meaningless. Only the discarded word counts at s = 0, only the previous word at s = 1, and both count in between.

## Knobs

| Symbol | GUI name | Meaning | Default |
|---|---|---|---|
| c | Clumsiness | Chance that each new word is replaced by a slip | tune (expect low values) |
| τ | Looseness | Low: nearly always the closest neighbour. High: more distant neighbours | tune |
| K | Top-K | Size of the model's shortlist | 1000 |
| s | Slip type | 0: substitution, scored against the intended word. 1: intrusion, scored against the previous word. In between: a blend | tune |

## Design decisions

- **Base model:** Qwen2.5-0.5B, the base model and not Instruct. Instruct models are tuned for polish and tend to repair slips. Input and output embeddings are tied, so there is one matrix. It runs on a laptop.
- **Two slip types, one slider:** substitution is a well-documented kind of human slip. Intrusion is the word-association slip from the original idea. Mixing the two similarity scores before the softmax makes the endpoints the pure types, with a blend in between.
- **Association:** cosine similarity of token embeddings, mean-centred first. Raw query-key dot products were rejected. Attention scores say where to read from, not which words are related.
- **Shortlist from the model:** the slip is drawn from the model's own top K, then reweighted by association. This keeps the grammar roughly intact, so the result reads as a human slip and not a glitch.
- **Consecutive slips are allowed.** Forbidding them later is a one-line change.
- **Sampling, not greedy,** for the model's own words. The temperature is fixed.

Left out for now: sound-based slips, a random-word control, model comparisons and K sweeps. Add them only if the first outputs raise a question they would answer.

## Interface (planned)

A Gradio app. Type an opening or pick one from a dropdown. Sliders set Clumsiness, Looseness, Top-K and Slip type. Slips are highlighted in the output.

## Openings

`openings.txt` holds 100 text openings, one per line. 85 are dialogue and 15 are narrative or explanatory, for contrast. Each stops mid-sentence with no trailing space, so the model supplies the next token.
