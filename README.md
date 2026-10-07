# ClumsyGPT

Make a language model slip the way people do: occasionally say a word that is closely associated with a neighbouring word, instead of the one that fits.

## The idea

People make two kinds of word slip.

- **Substitution:** they say "pass the pepper" when they meant "salt".
- **Intrusion:** the word they just said pulls in a related one, so "nurse" follows "doctor" where it does not belong.

In both cases the slip is related to a nearby word but wrong for the sentence. LLMs almost never do this. ClumsyGPT injects both kinds on purpose and watches how the model recovers.

## 👉 [Try it in your browser](https://navid-math-ai--clumsygpt-web.modal.run/)

No install, no sign-up. It is a demo on shared hardware, so the first visit after a quiet spell can take a minute or two to wake up, and each generation takes about 15 seconds.

## Try it

### In your browser

1. Open the [demo](https://navid-math-ai--clumsygpt-web.modal.run/).
2. Pick a starting line from the **Load an opening** menu, or type your own in the text box.
3. Press **Generate**. The model continues your text.
4. Slips appear in colour. **Hover over one** to see the word the model meant to say.
5. Press **Generate** again to keep going, or turn up **Clumsiness** for more slips.

### On your own computer

You need Python 3.10 or newer and about 1 GB of free disk space for the model. A GPU is not required.

```
git clone https://github.com/nav-nab/clumsy-GPT.git
cd clumsy-GPT
pip install -r requirements.txt
python app.py
```

Then open the address it prints (usually http://127.0.0.1:7860) in your browser. The first run downloads the model, which takes a few minutes. After that it starts quickly.

Running locally has two advantages: it is not shared with other visitors, and it uses your GPU or Apple Silicon chip if you have one. On a 2-core CPU a 150-token generation takes about 15 seconds.

## Using the app

The text box at the top is where everything happens. Type an opening or load one from the dropdown, set the four sliders and press Generate. The generated text is added to the box, so pressing Generate again continues it. With **Highlight slips** on, the text is shown read-only and each slip is coloured, and hovering over one shows a box reading "What I meant to say was: X", where X is the word that was discarded. Untick it to edit the text. A line under the button lists the slips of the latest run and says how many words were eligible. The sliders are explained in [Knobs](#knobs) below.

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

## Running locally: notes

If you see "PyTorch >= 2.5 is required", `transformers` is too new for your PyTorch. Run `pip install -r requirements.txt` again, since it pins `transformers<5`. The code is tested with PyTorch 2.2.2 and with current PyTorch. On PyTorch 2.2.2 with NumPy 2, a long NumPy warning appears at start-up. It is harmless, and `pip install "numpy<2"` silences it.

Every run uses a fresh random seed. Settings that are not in the GUI live in `clumsygpt/config.py`: maximum new tokens (150), the sampling temperature, the word-frequency thresholds, and the slider defaults and ranges.

Tests use fake models, so they need no download: `pip install pytest && pytest`.

## Code

- `app.py`: the Gradio GUI.
- `modal_app.py`: the set-up for the hosted demo.
- `clumsygpt/generate.py`: the generation loop and the slip trial.
- `clumsygpt/slips.py`: which words are allowed, and scoring and sampling the slip.
- `clumsygpt/vocab.py`: classifies every token as a word, a content word, its capitalisation and its dictionary form.
- `clumsygpt/config.py`: settings.
- `docs/design-decisions.md`: why it is built this way.
- `openings.txt`: 100 text openings, one per line. 85 are dialogue and 15 are narrative or explanatory, for contrast. Each stops mid-sentence with no trailing space, so the model supplies the next token.
