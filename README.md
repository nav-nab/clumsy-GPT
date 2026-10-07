# ClumsyGPT

Make a language model slip the way people do: occasionally swap the word it meant to say for a closely associated one that does not fit.

**Status:** design only, no code yet. This is a qualitative project. There are no metrics. Success means reading the output and finding it convincingly human.

## The idea

People say "pass the pepper" when they meant "salt". The substitute is related to the intended word but wrong for the sentence. LLMs almost never do this. ClumsyGPT injects it on purpose and watches how the model recovers.

## How it works

It is a thin layer around ordinary sampling. The transformer is untouched and the KV cache stays intact.

1. The model samples the next token as usual (temperature 0.7).
2. If the token starts a new word, run a Bernoulli trial with probability **c**. If it fails, keep the token.
3. If it passes, discard the token. This is the word the model meant to say. Take the model's **top K** whole-word candidates at that position, excluding the discarded word itself.
4. Score each candidate by cosine similarity to the discarded token's embedding.
5. Softmax the scores with temperature **τ** and sample one word. That word is the slip.
6. The slip goes back into the context, and the model carries on from it.

The trial is skipped when the sampled token is a word fragment, a function word or punctuation. The neighbours of "the" or "," are meaningless.

## Knobs

| Symbol | GUI name | Meaning | Default |
|---|---|---|---|
| c | Clumsiness | Chance that each new word is replaced by a slip | tune (expect low values) |
| τ | Looseness | Low: nearly always the closest neighbour of the intended word. High: more distant neighbours | tune |
| K | Top-K | Size of the model's shortlist | 1000 |

## Design decisions

- **Base model:** Qwen2.5-0.5B, the base model and not Instruct. Instruct models are tuned for polish and tend to repair slips. Input and output embeddings are tied, so there is one matrix. It runs on a laptop.
- **Substitution, not intrusion:** candidates are scored against the word the model meant to say, not the word before it. This mimics substitution errors such as "pass the pepper" for "salt", a well-documented kind of human slip. Scoring against the preceding word would give intrusions instead ("doctor" followed by "nurse"). Switching to that is a one-line change.
- **Association:** cosine similarity of token embeddings, mean-centred first. Raw query-key dot products were rejected. Attention scores say where to read from, not which words are related.
- **Shortlist from the model:** the slip is drawn from the model's own top K, then reweighted by similarity to the intended word. This keeps the grammar roughly intact, so the result reads as a human slip and not a glitch.
- **Consecutive slips are allowed.** Forbidding them later is a one-line change.
- **Sampling, not greedy,** for the model's own words. The temperature is fixed.

Left out for now: sound-based slips, a random-word control, model comparisons and K sweeps. Add them only if the first outputs raise a question they would answer.

## Interface (planned)

A Gradio app. Type an opening or pick one from a dropdown. Sliders set Clumsiness, Looseness and Top-K. Slips are highlighted in the output.

## Openings

`openings.txt` holds 100 text openings, one per line. 85 are dialogue and 15 are narrative or explanatory, for contrast. Each stops mid-sentence with no trailing space, so the model supplies the next token.
