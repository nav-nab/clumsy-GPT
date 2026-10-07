# ClumsyGPT

Make a language model slip the way people do: occasionally say a word that is closely associated with a neighbouring word, instead of the one that fits.

**Status:** working prototype. This is a qualitative project. There are no metrics. Success means reading the output and finding it convincingly human.

## The idea

People make two kinds of word slip.

- **Substitution:** they say "pass the pepper" when they meant "salt".
- **Intrusion:** the word they just said pulls in a related one, so "nurse" follows "doctor" where it does not belong.

In both cases the slip is related to a nearby word but wrong for the sentence. LLMs almost never do this. ClumsyGPT injects both kinds on purpose and watches how the model recovers.

## How it works

It is a thin layer around ordinary sampling. The transformer is untouched and the KV cache stays intact.

1. The model samples the next token as usual (temperature 0.7).
2. If the token starts a new word, run a Bernoulli trial with probability **c**. If it fails, keep the token.
3. If it passes, discard the token. This is the word the model meant to say. Take the model's **top K** candidates among the allowed words (below) at that position.
4. Score each candidate as `(1 − s) · cos(candidate, discarded word) + s · cos(candidate, previous word)`. Here `cos` is cosine similarity of token embeddings, and **s** is the slip type. At s = 0 this is pure substitution. At s = 1 it is pure intrusion.
5. Softmax the scores with temperature **τ** and sample one word. That word is the slip.
6. The slip goes back into the context, and the model carries on from it.

**Allowed words** are whole content words with the same capitalisation as the discarded word. The discarded word and the previous word are left out, along with their case variants and inflections (salt/salts, know/knew).

The trial is skipped when a word being scored against is a fragment, a function word or punctuation. The neighbours of "the" or "," are meaningless. Only the discarded word counts at s = 0, only the previous word at s = 1, and both count in between. Between the extremes fewer words qualify, so Clumsiness needs to be higher.

## Knobs

| Symbol | GUI name | Meaning | Default (range) |
|---|---|---|---|
| c | Clumsiness | Chance that each eligible word is replaced by a slip | 0.1 (0 to 0.3) |
| τ | Looseness | Low: nearly always the closest neighbour. High: more distant neighbours | 0.05 (0.01 to 0.3) |
| K | Top-K | Size of the model's shortlist | 100 (10 to 5000) |
| s | Slip type | 0: substitution, scored against the intended word. 1: intrusion, scored against the previous word. In between: a blend | 0 (0 to 1) |

Looseness above about 0.2 gives mostly noise. About a quarter of generated tokens are eligible for a trial at s = 0, and a fifth at s = 1, so at c = 0.1 roughly 2 to 3% of words slip.

## Design decisions

- **Base model:** Qwen2.5-0.5B, the base model and not Instruct. Instruct models are tuned for polish and tend to repair slips. Input and output embeddings are tied, so there is one matrix. It runs on a laptop.
- **Two slip types, one slider:** substitution is a well-documented kind of human slip. Intrusion is the word-association slip from the original idea. Mixing the two similarity scores before the softmax makes the endpoints the pure types, with a blend in between.
- **Association:** cosine similarity of token embeddings, mean-centred first. Raw query-key dot products were rejected. Attention scores say where to read from, not which words are related.
- **Shortlist from the model:** the slip is drawn from the model's own top K, then reweighted by association. This keeps the grammar roughly intact, so the result reads as a human slip and not a glitch.
- **What counts as a word:** a token that is a space followed by letters. It must be at least three letters and frequent enough in the `wordfreq` list (Zipf 2.5, or 3.5 for three letters and 3.0 for four). Function words are never candidates.
- **No inflections as slips:** without this filter, most nearest neighbours were grammatical variants ("doctor" to "doctors", "know" to "knew"). Variants are removed by spelling and by dictionary form (`lemminflect`). A few derived forms still get through ("sell" to "sale").
- **Consecutive slips are allowed.** Forbidding them later is a one-line change.
- **Sampling, not greedy,** for the model's own words. The temperature is fixed.

Left out for now: sound-based slips, a random-word control, model comparisons and K sweeps. Add them only if the first outputs raise a question they would answer.

## Run it

```
pip install -r requirements.txt
python app.py
```

If you see "PyTorch >= 2.5 is required", `transformers` is too new for your PyTorch. Run `pip install -r requirements.txt` again, since it pins `transformers<5`. The code is tested with PyTorch 2.2.2 and with current PyTorch. On PyTorch 2.2.2 with NumPy 2, a long NumPy warning appears at start-up. It is harmless, and `pip install "numpy<2"` silences it.

The first run downloads the model (about 1 GB). It uses CUDA or Apple Silicon if present, and otherwise the CPU. A 150-token generation takes about 15 seconds on a 2-core CPU. Every run uses a fresh random seed.

The text box at the top is where everything happens. Type an opening or load one from the dropdown, set the four sliders and press Generate. The generated text is added to the box, so pressing Generate again continues it. With Highlight slips on, the text is shown read-only and each slip is coloured, and hovering over one shows a box reading "What I meant to say was: X", where X is the word that was discarded. Untick it to edit the text. A line under the button lists the slips of the latest run and says how many words were eligible.

Settings that are not in the GUI live in `clumsygpt/config.py`: maximum new tokens (150), the sampling temperature, the word-frequency thresholds, and the slider defaults and ranges.

Tests use fake models, so they need no download: `pip install pytest && pytest`.

### Put it online

`modal_app.py` serves the same app on [Modal](https://modal.com), so visitors can use it in a browser without installing anything. It runs on one CPU core and sleeps when nobody is using it. The first visitor after a quiet spell waits a short while for it to wake up.

```
pip install modal
modal setup
modal deploy modal_app.py
```

The last command prints the public URL. Run it again after any change to redeploy. Modal bills for the seconds the container is awake, so set a spending limit in your Modal billing settings before sharing the link.

## Code

- `app.py`: the Gradio GUI.
- `modal_app.py`: serves the GUI on Modal.
- `clumsygpt/generate.py`: the generation loop and the slip trial.
- `clumsygpt/slips.py`: which words are allowed, and scoring and sampling the slip.
- `clumsygpt/vocab.py`: classifies every token as a word, a content word, its capitalisation and its dictionary form.
- `clumsygpt/config.py`: settings.
- `openings.txt`: 100 text openings, one per line. 85 are dialogue and 15 are narrative or explanatory, for contrast. Each stops mid-sentence with no trailing space, so the model supplies the next token.
