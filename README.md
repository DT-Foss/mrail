# mrail — maps of exact language-model runs

`.mrail` is an append-only file format for **what a language model has already computed**. For models whose state
after any prefix has a constant size (state-space models, linear attention, gated delta networks, fixed-state
attention such as Ward recall), the state is a pure function of the token prefix. A map of earlier runs therefore lets a
runtime

- **answer known contexts without running the model** — the stored continuation *is* the model's own greedy output;
- **restart exactly** from anchored states at shared prefixes (system prompts, conversation histories), or rebuild any
  state from the stored tokens;
- **draft** continuations for new contexts, verified by the model in one pass, so the map changes speed, never output;
- **grow with use** — every served answer is appended; maps split by topic, load and merge like any file.

Measured with a compiled Qwen2.5-1.5B-Instruct (constant state of 14.4 MiB) on the Apple Neural Engine (see the papers
below): known questions in 3.0–4.3 ms instead of 9.7–10.3 s; 2.54 instead of 1.25 accepted tokens per pass on unseen
coding tasks with a 1.2 MB coding map of 1 641 runs (2.27× greedy decoding); 8.4 MB instead of 893 MB for the anchors of
the same runs through delta coding and stations, 9.6 kB for their tokens alone.

This repository contains the [format specification](SPEC.md) and a dependency-light reference reader and writer
(Python, numpy). It does not contain a model runtime.

```python
import mrail
m = mrail.MapFile("coding.mrail")                  # opens or creates
answer = mrail.exact_answer(m, context_tokens, max_new=256, stop={151645})
m.add_session(prompt_tokens + model_answer_tokens + [next_token], gen_start=len(prompt_tokens))
tree = m.rail.tree(context_tokens, budget=31)       # draft tree [(token, parent)] for verification
```

```
python -m mrail info coding.mrail
python -m mrail merge team.mrail alice.mrail bob.mrail
```

## Share your map

A map is a file of verified output of one exact model. Maps merge by loading their runs (`python -m mrail merge`), so
a map of a repository, a documentation set, a tool protocol or a team's daily questions can be shared with everyone who
runs the same model. Runs from a foreign map are loaded as **drafts only**: they propose continuations, the model
verifies every token, and only runs produced by the local model answer a context directly. A shared map therefore
makes inference faster on its topic and cannot change a single output token. The map is written by the model itself
(the Möbius loop: every verified answer becomes a run, every disagreement becomes a new run), so it grows finer with use
and never turns into a store of canned answers.

Name shared maps by model and topic, e.g. `Qwen2.5-1.5B-Instruct-HCGM.coding.mrail`, and renew them when the model
version changes.

## Papers

- D. T. Foss, *Compute Once: Constant-State Language Models Make Answers Reusable and Agents Cheap* (2026).
- D. T. Foss, *The Holographic Causal Graph Machine: Compiling a Pretrained Transformer into a Constant-State Model
  without Training* (2026).
- D. T. Foss, *No GPU, No KV Cache: A Constant-State Language Model on the Neural Engine of a Mac mini* (2026).

Models and maps: [huggingface.co/tfwnotops](https://huggingface.co/tfwnotops).

License: Apache-2.0.
