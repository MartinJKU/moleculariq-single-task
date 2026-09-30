# Official MolecularIQ semantics this project relies on

Notes taken while reading the official code, so that later choices are traceable
to a source rather than to memory. Everything here is about how the *task* is
defined; none of it is a benchmark score, and none of it was used to tune
training.

Upstream commits: core `a1b8963`, eval `425ecaa`, benchmark `cf08baa`.

---

## Answer extraction

`lm_eval/tasks/moleculariq/extractors.py::extract_moleculariq_answer`, in order:

1. strip a thinking block (`</think>`, `<|think_end|>`);
2. take the content of the **last** `<answer>...</answer>` pair;
3. otherwise the last `<|answer_start|>...<|answer_end|>` pair;
4. otherwise the last balanced JSON/dict/list structure found anywhere;
5. otherwise general fallbacks (boxed, bold, trailing number).

Two consequences the reward code has to respect:

* **A missing `<answer>` block is not a parse failure.** Step 4 happily recovers
  a bare ```json fence, which is what the base policy writes. Calling that
  "malformed" made early rollouts look broken when the answers were in fact being
  read correctly, so `rewards.py` keeps `ParseStatus` (did extraction+verification
  work) separate from `FormatStatus` (did it use the requested envelope).
* **Two conflicting answer blocks are scored on the last one.** We record the
  conflict as a diagnostic but do not penalise it separately, because changing
  that would stop the training reward matching the benchmark.

## Scoring

`moleculariq_core.evaluate_answer` (alias of `chemical_reward`) dispatches on
task type and returns a **binary** 0.0/1.0:

| doc `task_type` | dispatches to | correct when |
|---|---|---|
| `count` / `single_count` / `multi_count` | `multi_count_dict_reward` | *every* requested key matches |
| `index` / `single_index` / `multi_index` | `multi_index_identification_reward` | every key's index **set** matches |
| `generation` / `*constraint_generation*` | `multi_constraint_generation_reward` | SMILES parses **and** satisfies every constraint |

Notes:

* Index comparison is set-based (`sorted(set(...))`), so order and duplicates do
  not matter, but an off-by-one shift is wrong.
* Predicted keys are canonicalised through the natural-language mappings, so
  `ring_count` and its aliases both resolve; duplicate keys score 0.
* A constraint dict is read as `constraint.get('type', constraint.get('property'))`,
  so either spelling works. This project writes `type`.
* Constraint satisfaction is checked against the molecule, never against a
  reference SMILES. Any molecule meeting the constraints is fully correct.

## The system prompt

The benchmark does **not** put the instruction in the task YAML (`description` is
deliberately empty to avoid doubling). It is passed on the command line via
`--system_instruction`, and the canonical string lives in
`task_processor.SYSTEM_PROMPT`.

That string differs from `moleculariq_core.SYSTEM_PROMPTS["concise"]` by a single
trailing space (`"Examples: "` vs `"Examples:"`). The eval repo's copy is
authoritative because it is what actually reaches the model, and that is what is
vendored. `tests/test_prompts.py` pins its hash and asserts whitespace-insensitive
equality with core, so real wording drift still fails.

The user turn is `doc["question"]` verbatim — no wrapper text. Training renders
the same two turns.

## Atom indexing

The README states indices are read "from 0 to N-1, reading the SMILES string left
to right, counting only heavy atoms". The generator shows a transformed SMILES
(randomised and/or kekulised with probability 0.5 each) while the property table
is keyed on the original, which left it ambiguous whether stored indices follow
the *displayed* string or the canonical one.

**Checked directly.** Four official test items with `metadata.is_randomized =
true` were re-solved with the current core, once on the displayed SMILES and once
on the canonical form:

| item | on displayed | on canonical |
|---|---|---|
| `aromatic_ring_index` | match | no match |
| `unspecified_stereocenter_index` | match | no match |
| `heterocycle_index` | match | no match |
| `r_s_stereocenter_r_index` | no match | no match |

So targets follow the **displayed** string, and `generation.py` computes them
there. (The fourth item matched neither: the current core computes `[]` where the
stored target is `[4]`. A small amount of core-version drift exists in the stored
targets; it affects every model identically and nothing was done about it.)

This was a one-off reading of the task definition. No benchmark score was
computed, and nothing downstream was tuned on it.

## Property support is not uniform

`build_property_catalog` probes every candidate property on eight fixed molecules
and keeps only those the official verifier accepts its own ground truth for. At
the pinned commits this keeps 176 count / 174 index / 168 constraint properties
and rejects 67, including:

* all `template_based_reaction_prediction_*_success` constraints —
  `MolecularIQD._get_solver_method` calls `predict_reaction_success`, which
  `TemplateBasedReactionSolver` does not define;
* several `oxidation_state_*` and `brics_decomposition_count` constraints, where
  the constraint path disagrees with the count path about the same property.

The rejections are recorded in the dataset manifest under
`property_catalog.rejected`. Probing rather than hard-coding means a later
upstream fix is picked up automatically.

## The published generation cap cannot run on a 32k-context model

`moleculariq_pass_at_k.yaml` sets `generation_kwargs: max_tokens: 32768`.
Qwen2.5-0.5B-Instruct's context window is also 32768. The harness computes

    max_ctx_len = model.max_length - max_gen_toks

and asserts it is positive, so the task as published fails at request time on
both backends -- vLLM and HF alike. There is no prompt short enough to fix it;
the cap has to be lowered.

`normalize_gen_kwargs` resolves the cap from aliases in the order
`max_gen_toks > max_new_tokens > max_tokens > max_completion_tokens`, so
`--gen_kwargs max_gen_toks=N` cleanly overrides the YAML value (the harness logs
a "multiple max token args provided" warning, which is expected).

### What the official runs actually used

The `with_config` branch carries the eval config for all 34 models in the paper.
**None of them passes a generation override or `max_model_len`** -- including
`qwen2.5-7b.yaml`, the same model family and the same 32768-token window. They
ran the YAML as published, on vLLM, with `batch_size: auto`.

That works on vLLM only by accident. `maybe_truncate` is called with
`shrink_gen_toks=False`, so it takes the "truncate the prompt" branch and
computes a prompt budget of `32768 - 32768 = 0`. `truncate_tokens` then does
`tokens[-0:]` -- and in Python `-0 == 0`, so that slice returns the *entire*
list rather than an empty one. The prompt survives untouched, `max_gen_toks`
stays 32768, and vLLM clamps generation to whatever is left of the window.

So the official effective ceiling was roughly 32,300 tokens: "generate until
EOS". The HF backend reaches the same arithmetic and asserts instead.

### What this project uses

`max_gen_toks=28672`, which is within ~11% of the official effective ceiling and
reserves 4096 tokens for the prompt.

That prompt budget is sized from measurement, not guesswork: over 22,800
rendered training prompts the distribution is median 436 / p99 556 / max 702
tokens, so 4096 is roughly 6x the observed maximum. The margin matters because
an overlong prompt is *left-truncated*, which would silently remove the front of
the system prompt -- a much worse failure than clipping a response.

Neither limit binds in practice: the longest response in any of the seven
evaluations is about 39,000 characters, far below 28,672 tokens. The override
lives in `configs/evaluation.yaml`, is recorded in every evaluation manifest
under `generation_overrides`, and is identical for every model.

### What it means for the numbers

The cap can only ever *cost* accuracy: a truncated response loses its closing
`</answer>`, so extraction fails and it scores 0. There is no mechanism by which
it inflates a score.

* Between the models here: comparable unconditionally -- same cap, backend,
  prompt, task and scoring.
* Against published leaderboard entries: comparable provided the cap never
  binds, which is checkable after the fact from the `--log_samples` output. The
  remaining difference is the backend (HF here, vLLM there): same weights and
  same sampling parameters, different kernels, so outputs are not token
  identical. Worth a sentence in the methods section, not a comparability
  problem.

## Sampling parameters differ by backend, and HF has to be told explicitly

The task sets `do_sample: true` and no temperature. What that ends up meaning
depends entirely on the backend.

**vLLM (what the official runs used).** `normalize_gen_kwargs` leaves
`temperature` unset in the `do_sample=True` branch -- it only warns.
`modify_gen_kwargs` then pops `do_sample` and hands the rest to
`SamplingParams`. With no temperature key, vLLM's own defaults apply:

    temperature 1.0 · top_p 1.0 · top_k off · repetition_penalty 1.0

**HF.** `huggingface.py:996` does
`generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)`,
injecting 0.0. It is only popped again when `do_sample is False`. Our task sets
`do_sample: true`, so both survive into `generate()` and transformers raises:

    ValueError: `temperature` (=0.0) has to be a strictly positive float

Worth being precise about the consequence, because it is easy to get wrong: HF
does **not** fall back to the model's `generation_config.json` here, because 0.0
is explicitly injected. But if that injection were fixed, it *would* -- and for
Qwen2.5-0.5B-Instruct that config is temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.1. None of which the official runs used.

So the HF backend has to be told the sampling parameters explicitly, and the
right values are vLLM's defaults:

    temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0

In transformers, `top_p=1.0` and `top_k=0` disable those warpers and
`repetition_penalty=1.0` is a no-op, so this is plain temperature-1.0 sampling
-- exactly what vLLM did. This is reproduction of the official setting, not a
departure from it, and it is what makes pass@3 meaningful rather than three
copies of one decode.

## Constraint phrasing is sometimes ungrammatical, on purpose

`NaturalLanguageFormatter.format_constraints_list` renders a zero-valued
constraint as a prepositional phrase ("without any hydrogen bond acceptors"),
and `TASKS["constraint_generation"]` templates expect a noun phrase, so some
questions read:

> Assemble a structure that obeys without any hydrogen bond acceptors.

This is not a bug in this project — the official benchmark's own items have the
same artefact, because both are produced by the same formatter. Training
questions are generated the same way deliberately: prompts the policy trains on
should look like the prompts it is tested on, awkward phrasing included.

## Official task configuration

`moleculariq_pass_at_k.yaml`:

```yaml
dataset_path: ml-jku/moleculariq-v0.0     # split: test, 5,111 items
output_type: generate_until
repeats: 3
generation_kwargs: {max_tokens: 32768, until: [], do_sample: true}
metric_list: [pass_at_1, pass_at_3, avg_accuracy, extracted_answers]
```

Beyond the generation settings described above (the cap and the explicit
sampling parameters), `evaluate.py` changes nothing: repeats, prompt, extraction
and scoring all come from the official config. Per-item breakdowns in the report
are computed from `--log_samples` output; the headline numbers always come from
the harness's own results file.

## Evaluation is deterministic

lm_eval seeds its random number generators, so the same weights, backend,
generation settings and batch size on the same GPU class reproduce a result bit
for bit. The base model was evaluated twice as a check; both runs agreed on every
metric and every item. The batch size matters because it changes how the seeded
sampling stream is consumed, so `configs/evaluation.yaml` records the batch size
each reported run used.
