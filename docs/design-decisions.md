# ClumsyGPT: design decisions

Why the project is built the way it is. The [README](../README.md) explains what it does.

- **Base model:** Qwen2.5-0.5B, the base model and not Instruct. Instruct models are tuned for polish and tend to repair slips. Input and output embeddings are tied, so there is one matrix. It runs on a laptop.
- **Two slip types, one slider:** substitution is a well-documented kind of human slip. Intrusion is the word-association slip from the original idea. Mixing the two similarity scores before the softmax makes the endpoints the pure types, with a blend in between.
- **Association:** cosine similarity of token embeddings, mean-centred first. Raw query-key dot products were rejected. Attention scores say where to read from, not which words are related.
- **Shortlist from the model:** the slip is drawn from the model's own top K, then reweighted by association. This keeps the grammar roughly intact, so the result reads as a human slip and not a glitch.
- **What counts as a word:** a token that is a space followed by letters. It must be at least three letters and frequent enough in the `wordfreq` list (Zipf 2.5, or 3.5 for three letters and 3.0 for four). Function words are never candidates.
- **No inflections as slips:** without this filter, most nearest neighbours were grammatical variants ("doctor" to "doctors", "know" to "knew"). Variants are removed by spelling and by dictionary form (`lemminflect`). A few derived forms still get through ("sell" to "sale").
- **Consecutive slips are allowed.** Forbidding them later is a one-line change.
- **Sampling, not greedy,** for the model's own words. The temperature is fixed.

Left out for now: sound-based slips, a random-word control, model comparisons and K sweeps. Add them only if the first outputs raise a question they would answer.
