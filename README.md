<img src="assets/avatar.svg" alt="ClumsyGPT avatar: a speech bubble with its tongue out" width="150" align="right">

# ClumsyGPT

**A language model that fumbles its words on purpose.**

People say "left" and mean "right". They ask for the pepper when they want the salt. Language models almost never do this. ClumsyGPT makes one slip the way people do, and watches how it recovers.

## The idea

There are two kinds of word slip, and ClumsyGPT makes both.

- **Substitution:** you say "pass the pepper" when you meant "salt".
- **Intrusion:** the word you just said drags in a related one, so "nurse" turns up after "doctor" where it does not belong.

Either way, the slip is related to a nearby word but wrong for the sentence.

## 👉 [Try it in your browser](https://navid-math-ai--clumsygpt-web.modal.run/)

No install and no sign-up. It runs on shared hardware, so the first visit after a quiet spell can take a minute or two to wake up. After that, each generation takes about 15 seconds.

## Try it

### In your browser

1. Open the [demo](https://navid-math-ai--clumsygpt-web.modal.run/).
2. Pick a starting line from **Load an opening**, or type your own in the text box.
3. Press **Generate**. The model continues your text.
4. Slips show up in colour. **Hover over one** to see the word the model meant to say.
5. Press **Generate** again to keep going. Turn up **Clumsiness** for more slips.

### On your own computer

You need Python 3.10 or newer and about 1 GB of disk space for the model. A GPU is optional.

```
git clone https://github.com/nav-nab/clumsy-GPT.git
cd clumsy-GPT
pip install -r requirements.txt
python app.py
```

Open the address it prints (usually http://127.0.0.1:7860). The first run downloads the model, which takes a few minutes. After that it starts quickly.

Running it yourself means no sharing with other visitors, and it uses your GPU or Apple Silicon chip if you have one. On a 2-core CPU, 150 tokens take about 15 seconds.

## Using the app

Everything happens in the text box. Type an opening or load one, set the sliders and press Generate. New text is added to the box, so Generate again to continue.

With **Highlight slips** on, the text is read-only and each slip is coloured. Hover over a slip to see "What I meant to say was: X", where X is the word that was discarded. Untick the box to edit the text. A line under the button lists the latest slips and how many words were eligible. The sliders are explained under [Knobs](#knobs).

## How it works

ClumsyGPT is a thin layer around ordinary sampling. The transformer is untouched and the KV cache stays intact.

1. The model samples the next token as usual (temperature 0.7).
2. If the token starts a new word, run a Bernoulli trial with probability **c**. If it fails, keep the token.
3. If it passes, discard the token. This is the word the model meant to say. Take the model's **top K** candidates among the allowed words (below) at that position.
4. Score each candidate as `(1 − s) · cos(candidate, discarded word) + s · cos(candidate, previous word)`. Here `cos` is the cosine similarity of token embeddings, and **s** is the slip type. At s = 0 this is pure substitution. At s = 1 it is pure intrusion.
5. Softmax the scores with temperature **τ** and sample one word. That word is the slip.
6. The slip goes back into the context, and the model carries on from it.
7. With **Shrug it off** on, the next 3 tokens cannot attend to the slip. Their attention mask hides it, so the slip does not steer them.

**Allowed words** are whole content words with the same capitalisation as the discarded word. The discarded word and the previous word are left out, along with their case variants and inflections (salt/salts, know/knew).

The trial is skipped when a word being scored against is a fragment, a function word or punctuation, because the neighbours of "the" or "," are meaningless. Only the discarded word counts at s = 0, and only the previous word at s = 1. In between, both count. Fewer words qualify there, so Clumsiness needs to be higher.

## Knobs

| Symbol | GUI name | Meaning | Default (range) |
|---|---|---|---|
| c | Clumsiness | Chance that each eligible word is replaced by a slip | 0.1 (0 to 0.3) |
| τ | Looseness | Low: nearly always the closest neighbour. High: more distant neighbours | 0.05 (0.01 to 0.3) |
| K | Top-K | Size of the model's shortlist | 100 (10 to 5000) |
| s | Slip type | 0: substitution, scored against the intended word. 1: intrusion, scored against the previous word. In between: a blend | 0 (0 to 1) |
| – | Shrug it off | A tick box. After a slip, the next 3 tokens cannot attend to it, which helps the sentence keep its direction. It may read less grammatically | off |

**Shrug it off** hides each slip from the next 3 tokens the model reads, by masking its key in attention. Those tokens are computed as if the slip had not been said. Two things are left unchanged. The token right after the slip is still chosen from the slip's own output, so the slip steers that one word. After 3 tokens the slip becomes visible again. The number 3 is `SHRUG_TOKENS` in `clumsygpt/config.py`.

Looseness above about 0.2 gives mostly noise. About a quarter of generated tokens are eligible for a trial at s = 0, and a fifth at s = 1. At c = 0.1, that means roughly 2 to 3% of words slip.

## Observations

Informal notes from reading outputs. They are impressions, not measurements.

- **A slip can derail the whole sentence.** Once a different word is in the context, the model treats it as part of the story and carries on from it. The direction of the text can change completely, and not only for the next few words. **Shrug it off** was added to test one fix: hide the slip from the next 3 tokens, accepting a less grammatical sentence in return for keeping its direction.
- **Raw nearest neighbours are dull.** Without filters, most replacements were grammatical variants of the intended word ("doctor" to "doctors", "know" to "knew"), plus function words and word fragments. Filtering these out is what makes the slips read as slips.
- **Looseness has a narrow useful range.** Above about 0.2 the replacements stop being related to the intended word and read as noise.

## Running locally: notes

If you see "PyTorch >= 2.5 is required", `transformers` is too new for your PyTorch. Run `pip install -r requirements.txt` again, since it pins `transformers<5`. The code is tested with PyTorch 2.2.2 and with current PyTorch. On PyTorch 2.2.2 with NumPy 2, a long NumPy warning appears at start-up. It is harmless, and `pip install "numpy<2"` silences it.

Every run uses a fresh random seed. Settings that are not in the GUI live in `clumsygpt/config.py`: maximum new tokens (150), the sampling temperature, the word-frequency thresholds, and the slider defaults and ranges.

Tests use fake models, so they need no download: `pip install pytest && pytest`.

## Code

- `app.py`: the Gradio GUI.
- `modal_app.py`: the set-up for the hosted demo.
- `clumsygpt/generate.py`: the generation loop and the slip trial.
- `clumsygpt/slips.py`: which words are allowed, and how the slip is scored and sampled.
- `clumsygpt/vocab.py`: classifies every token as a word, a content word, its capitalisation and its dictionary form.
- `clumsygpt/config.py`: settings.
- `openings.txt`: 100 text openings, one per line. 85 are dialogue and 15 are narrative or explanatory, for contrast. Each stops mid-sentence with no trailing space, so the model supplies the next token.

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
