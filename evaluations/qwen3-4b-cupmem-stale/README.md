# Qwen3-4B CUP-Mem evaluation on STALE

This package runs an auditable CUP-Mem evaluation over 50 Type I and 50 Type II
STALE scenarios. Each scenario contains its two evidence sessions plus ten
deterministically sampled distractor sessions. CUP-Mem builds one frozen memory
and answers SR, PR, and IPA independently.

## Models

- Target/backbone: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
- Retriever: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Judge and failure diagnosis: `gpt-6-luna`

The target runs locally through vLLM with thinking disabled. The official judge
uses the exact STALE all-in-one prompt. Failure diagnosis is a separate call and
never changes the official score.

## RunPod

The supported pod has one RTX A6000, 50 GB RAM, and at least 50 GB local disk.
Configure the judge key without writing it to disk:

```bash
export JUDGE_OPENAI_API_KEY='...'
```

Set up the locked environment:

```bash
cd evaluations/qwen3-4b-cupmem-stale
bash scripts/setup_runpod.sh
```

Run the two-scenario end-to-end smoke test:

```bash
bash scripts/run_runpod.sh qwen3-4b-cupmem-stale-seed42-v1 smoke
```

Inspect the artifacts, then resume all remaining scenarios with the same ID:

```bash
bash scripts/run_runpod.sh qwen3-4b-cupmem-stale-seed42-v1 full
```

Artifacts are restored from and synchronized every 60 seconds to:

```text
s3://stale-qwen-evals-589498925064-us-east-1-an/cup-mem/runs/<run-id>/
```

Optional environment overrides:

```bash
export CUPMEM_ARTIFACT_BASE=/workspace/cupmem-artifacts
export CUPMEM_CACHE_DIR=/workspace/huggingface-cache
export CUPMEM_S3_BASE_URI=s3://bucket/prefix
export CUPMEM_SYNC_INTERVAL_SECONDS=60
export JUDGE_OPENAI_BASE_URL=https://api.openai.com/v1
```

Model weights, Hugging Face caches, API keys, and raw embedding vectors are not
uploaded. Full Qwen requests/responses and CUP-Mem traces are stored as compressed
Zstandard artifacts; memory summaries, answers, judge labels, and review files
remain directly readable.

## Offline tests

```bash
uv run --locked python -m unittest discover -s tests -v
```
