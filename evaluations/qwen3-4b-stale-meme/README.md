# Qwen3-4B memory evaluations

This package runs raw, greedy Qwen3-4B evaluations on STALE and MEME from one
shared Transformers inference backend. It is isolated from the existing
Qwen3-0.6B evaluator.

## Evaluation matrix

STALE uses the same pinned dataset, seed 42 sample, and author-derived prompts
as the prior run. It prepares 25 Type I and 25 Type II scenarios and produces
150 independent responses in each view:

- `truncated`: both evidence sessions plus deterministic whole-session distractors.
- `evidence_only`: exactly the two relevant sessions.
- `key_pairs_chronological`: only the M_old and M_new user/assistant pairs.
- `key_pairs_reversed`: the same pairs in reversed presentation order, with
  timestamps and message text unchanged.

MEME uses all 100 episodes but only Cascade and Absence:

- `nofiller`: the official evidence-only `meme_nofiller.json` episodes unchanged.
- `filler_29k`: the official `meme_filler32k.json` episodes, with only complete
  filler sessions removed deterministically until every selected prompt is at
  most 29,000 Qwen tokens. Every evidence session and chronological boundary is
  preserved.

MEME contains 294 selected after-change questions per view. The runner also asks
their 294 matching before-change questions because the official Cascade and
Absence metric applies a trivial-pass filter. Therefore each MEME view makes
588 model calls. The report clearly separates prerequisite and scored calls.

Both benchmarks disable thinking, use FP16, greedy decoding, batch size one,
the Transformers SDPA implementation, and a fail-closed Flash Attention kernel
on the NVIDIA L4.

## AWS setup

Use the repository clone on the `g6.xlarge`, then enter this directory:

```bash
cd ~/cs269-fa26-project/evaluations/qwen3-4b-stale-meme
bash scripts/setup_aws.sh
```

The setup script installs the locked Python 3.12 environment, verifies the L4,
tests Flash SDPA with grouped-query attention, and checks that at least 25 GiB
is free. The 100 GB EBS volume is sufficient for the environment, Qwen3-4B
cache, datasets, and response artifacts.

Run long jobs inside tmux:

```bash
tmux new -s qwen4b-evals
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t qwen4b-evals
```

## STALE smoke and full runs

The wrapper uses one run ID for the truncated/evidence-only pair and another
for the chronological/reversed key-pair experiment:

```bash
bash scripts/run_aws_stale.sh \
  qwen3-4b-stale-seed42-v1 \
  qwen3-4b-stale-key-pairs-v1 \
  smoke \
  "Qwen3-4B STALE four-view evaluation"
```

Smoke mode completes one full scenario (SR, PR, and IPA) in every view: 12
responses total. Inspect all four `results.md` files, then resume:

```bash
bash scripts/run_aws_stale.sh \
  qwen3-4b-stale-seed42-v1 \
  qwen3-4b-stale-key-pairs-v1 \
  full \
  "Qwen3-4B STALE four-view evaluation"
```

## MEME smoke and full runs

```bash
bash scripts/run_aws_meme.sh \
  qwen3-4b-meme-cas-abs-v1 \
  smoke \
  "Qwen3-4B MEME Cascade/Absence, nofiller and filler_29k"
```

Smoke mode completes every selected before/after probe in one episode in each
view. Inspect:

```text
/home/ubuntu/qwen3-4b-artifacts/qwen3-4b-meme-cas-abs-v1/views/nofiller/results.md
/home/ubuntu/qwen3-4b-artifacts/qwen3-4b-meme-cas-abs-v1/views/filler_29k/results.md
```

Then resume the full run:

```bash
bash scripts/run_aws_meme.sh \
  qwen3-4b-meme-cas-abs-v1 \
  full \
  "Qwen3-4B MEME Cascade/Absence, nofiller and filler_29k"
```

The combined workload is 1,776 model calls, many with roughly 29K-token input.
Do not plan on a one-hour end-to-end run on a single L4. Use the smoke runs to
measure actual throughput on the instance before estimating completion time.

## Persistence and resume

Local artifacts live under:

```text
/home/ubuntu/qwen3-4b-artifacts/<run-id>
```

Each wrapper restores that run from and synchronizes it every 60 seconds to:

```text
s3://stale-qwen-evals-589498925064-us-east-1-an/stale-qwen-evals/runs/<run-id>/
```

Every response is atomically checkpointed before the aggregate JSONL and report
are rebuilt. Running the same command with the same run ID and `full` resumes
missing responses without duplication. The Hugging Face cache stays on EBS and
is not uploaded.

Optional deployment overrides:

```bash
export QWEN4B_ARTIFACT_BASE=/home/ubuntu/qwen3-4b-artifacts
export QWEN4B_CACHE_DIR=/home/ubuntu/huggingface-cache
export QWEN4B_S3_BASE_URI=s3://bucket/prefix
export QWEN4B_SYNC_INTERVAL_SECONDS=60
```

Both the pinned model and datasets are public, so a Hugging Face token is not
required.

## Artifact layout

Every context view has its own raw responses and report:

```text
<run-id>/
├── scenarios.json or episodes.json
├── preparation_manifest.json
├── run_metadata.json
└── views/
    └── <view>/
        ├── checkpoints/
        ├── manifest.json
        ├── responses.jsonl
        ├── results.md
        ├── manual_scores.csv
        ├── summary.json
        └── summary.md
```

For MEME, manually enter `pass` or `fail` in both `before_score` and
`after_score`. The summary computes the official pass as their conjunction:

```bash
uv run --locked qwen3-memory-eval meme-summarize \
  --artifact-root /home/ubuntu/qwen3-4b-artifacts/qwen3-4b-meme-cas-abs-v1 \
  --view all
```

STALE retains the prior manual-score and paired-view summary commands:

```bash
uv run --locked qwen3-memory-eval summarize \
  --artifact-root /home/ubuntu/qwen3-4b-artifacts/qwen3-4b-stale-seed42-v1 \
  --view all
```

No LLM judge or automatic semantic grading is run by these scripts.

## Reproducibility

- Model: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
- STALE: `STALEproj/STALE@617c51dc200b5ab09970834144c7e51c77959af0`
- MEME: `meme-benchmark/MEME@e14c4e685772014deb96f91dc7eddfbeb21d90d8`
- Seed: 42
- STALE input/output budget: 32,256/512 tokens
- MEME hard input/output budget: 32,268/500 tokens
- MEME filler target: at most 29,000 input tokens

MEME prompting and transcript serialization reproduce the authors'
`UNIFIED_ANSWER_PROMPT` and `in_context_baseline.py` behavior. The released
v1.0 dataset documents ten potentially incorrect Absence labels; the pipeline
preserves those gold labels and flags the affected rows instead of changing
the benchmark.

## Tests

```bash
uv run --locked python -m unittest discover -s tests -v
```

Optional pinned-tokenizer integration check:

```bash
RUN_HF_INTEGRATION=1 uv run --locked python -m unittest discover \
  -s tests -p 'test_hf_integration.py' -v
```

The AWS smoke runs are the final checks for the model weights, L4 memory,
Flash SDPA, S3 permissions, and live checkpoint recovery.
