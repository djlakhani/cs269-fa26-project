# Qwen3-0.6B on STALE

This directory contains an independent, resumable pilot evaluator for
`Qwen/Qwen3-0.6B` on ten STALE scenarios (five `T1`, five `T2`) and all three
probing dimensions. The local `STALE/` checkout is reference material only;
this package does not import or modify it.

The evaluator uses the authors' response-generation prompts, disables Qwen
thinking through the chat template, uses greedy decoding, and reserves 512 of
the conservative 32,768-token context window for output. It makes three
independent calls per scenario and performs no LLM judging.

## Colab usage

Select a T4 GPU runtime and mount Drive in a notebook cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

From the terminal, enter this directory and install the package:

```bash
bash scripts/setup_colab.sh
```

Use a directory on the mounted Drive for every important artifact. Run one
query first as a GPU and dependency smoke test:

```bash
bash scripts/run_colab.sh \
  /content/drive/MyDrive/qwen3-06b-stale-pilot \
  smoke
```

Then resume the remaining 29 queries:

```bash
bash scripts/run_colab.sh \
  /content/drive/MyDrive/qwen3-06b-stale-pilot \
  full
```

The wrapper refuses paths outside `/content/drive/MyDrive`. By default the
Hugging Face cache is also kept beneath the artifact directory. Set
`STALE_CACHE_DIR` to another persistent location if desired.

## Portable CLI

The Python pipeline has no Colab-specific paths. A national-compute or Slurm
launcher needs a CUDA-enabled PyTorch installation, then only has to set its
cache/output paths and call the same commands:

```bash
stale-qwen-eval prepare --artifact-root /persistent/stale-pilot --cache-dir /persistent/hf-cache
stale-qwen-eval run --artifact-root /persistent/stale-pilot --cache-dir /persistent/hf-cache --resume
stale-qwen-eval report --artifact-root /persistent/stale-pilot
```

`prepare` will reuse an existing `scenarios.json`; it never resamples or
retruncates it implicitly. `--force` can replace a scenario file only when no
responses exist. Use a new artifact directory for a different sample or token
budget.

The prepared history is shared by all three questions in a scenario. Both
evidence sessions are retained in full. Whole distractor sessions are removed
in a deterministic seed-and-UID order until the Qwen tokenizer verifies that
all three fully templated requests fit within 32,256 input tokens.

## Artifacts and manual scoring

The artifact directory contains:

- `scenarios.json`: the frozen sample and truncated histories.
- `manifest.json`: revisions, checksums, prompts, runtime, GPU, and decoding settings.
- `checkpoints/`: one atomic checkpoint per generated query.
- `responses.jsonl`: ordered machine-readable results rebuilt from checkpoints.
- `results.md`: queries, raw responses, and expected ground-truth criteria.
- `manual_scores.csv`: one row per response for a human `pass` or `fail` label.

Edit only the `score` and `notes` columns in `manual_scores.csv`. Existing
annotations are preserved when reports are rebuilt. Once all 30 rows are
labelled, calculate benchmark numbers with:

```bash
stale-qwen-eval summarize --artifact-root /persistent/stale-pilot
```

The command writes `summary.json` and `summary.md`, with overall, conflict-type,
probing-dimension, and type-by-dimension accuracies. It exits with status 2 and
reports missing rows when scoring is incomplete.

## Reproducibility defaults

- Dataset: `STALEproj/STALE@617c51dc200b5ab09970834144c7e51c77959af0`
- Model: `Qwen/Qwen3-0.6B@167b8104f88905a951069f5f95f9776908da5f68`
- Scenario seed: `42`
- Types: `T1 = co-referential`, `T2 = propagated`
- Dimensions: `dim1 = SR`, `dim2 = PR`, `dim3 = IPA`
- Inference: FP16, SDPA, batch size one, `do_sample=False`, thinking disabled

The manifest records the exact runtime packages and GPU. Greedy decoding makes
runs highly repeatable, but bitwise identity across different GPU architectures
is not guaranteed.

## Tests

Run the offline suite without downloading the model or dataset:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

An optional tokenizer integration check is enabled with:

```bash
RUN_HF_INTEGRATION=1 PYTHONPATH=src python -m unittest tests.test_hf_integration -v
```

## Attribution

STALE is released under CC BY 4.0, with LongMemEval-derived distractors carrying
their accompanying MIT notice. The authors' code is MIT-licensed. Qwen3-0.6B is
Apache 2.0 licensed. Consult the upstream repositories for their full notices.

