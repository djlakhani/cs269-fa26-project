# Qwen3-0.6B on STALE

This package performs a paired-context evaluation of `Qwen/Qwen3-0.6B` on
STALE. It samples 25 `T1` and 25 `T2` scenarios with seed 42 and asks all three
STALE probes independently under two context views:

- `truncated`: both evidence sessions plus as many whole distractor sessions as
  fit the 32,256-token input budget.
- `evidence_only`: exactly the two sessions identified by
  `relevant_session_index`.

The resulting evaluation has 50 scenarios, 150 queries per view, and 300 model
responses. Thinking is disabled, decoding is greedy, and scoring is manual.

## AWS deployment

The supported deployment is an EC2 `g6.xlarge` in `us-east-1`, using the AWS
Deep Learning Base OSS Nvidia Driver GPU AMI. The instance provides one NVIDIA
L4, and inference forces PyTorch Flash SDPA. It never permits a silent fallback
to quadratic math attention.

Clone the repository and enter this directory:

```bash
git clone YOUR_REPOSITORY_URL
cd cs269-fa26-project/evaluations/qwen3-06b-stale
```

Set up the locked Python 3.12 environment:

```bash
bash scripts/setup_aws.sh
```

The setup script verifies the L4 and installs the exact dependencies from
`uv.lock`. The AMI supplies the NVIDIA driver; the uv environment supplies the
CUDA-enabled PyTorch wheel and Python packages.

### Smoke run

Choose a stable run ID. The smoke run prepares all 50 frozen scenarios, then
completes one full scenario (SR, PR, and IPA) in each context view:

```bash
bash scripts/run_aws.sh \
  qwen3-06b-stale-seed42-v1 \
  smoke \
  "Qwen3-0.6B, 25 T1 + 25 T2, paired context comparison"
```

### Full run

After inspecting the six smoke responses, use the same run ID. Resume skips
those completed responses:

```bash
bash scripts/run_aws.sh \
  qwen3-06b-stale-seed42-v1 \
  full \
  "Qwen3-0.6B, 25 T1 + 25 T2, paired context comparison"
```

Each view has a 50-scenario progress bar. The wrapper stores live artifacts on
EBS under `/home/ubuntu/stale-artifacts/<run-id>` and synchronizes them to S3
every 60 seconds and on exit:

```text
s3://stale-qwen-evals-589498925064-us-east-1-an/
└── stale-qwen-evals/
    └── runs/
        └── <run-id>/
```

Override deployment paths when necessary:

```bash
export STALE_ARTIFACT_BASE=/home/ubuntu/stale-artifacts
export STALE_CACHE_DIR=/home/ubuntu/huggingface-cache
export STALE_S3_BASE_URI=s3://bucket/prefix
export STALE_SYNC_INTERVAL_SECONDS=60
```

The Hugging Face cache is deliberately outside the artifact root and is not
uploaded to S3. A Hugging Face token is not required for the public pinned model
and dataset.

Run the wrapper inside `tmux` so inference continues if SSH disconnects:

```bash
tmux new -s stale-eval
bash scripts/run_aws.sh qwen3-06b-stale-seed42-v1 smoke "description"
```

## Artifact layout

```text
<run-id>/
├── comparison_summary.json
├── comparison_summary.md
├── run_metadata.json
├── scenarios.json
├── preparation_manifest.json
└── views/
    ├── truncated/
    │   ├── manifest.json
    │   ├── checkpoints/
    │   ├── responses.jsonl
    │   ├── results.md
    │   └── manual_scores.csv
    └── evidence_only/
        └── ...same files...
```

`scenarios.json` contains the single frozen sample and both context views for
every scenario. Each view has independent checkpoints, reports, and manual
scores. Existing manual annotations are preserved when reports are rebuilt.
After both views are fully scored, the root comparison files report accuracy
deltas and paired probe outcomes (improved, regressed, both pass, or both fail).

Once all labels are filled with `pass` or `fail`, summarize both views:

```bash
uv run --locked stale-qwen-eval summarize \
  --artifact-root /home/ubuntu/stale-artifacts/qwen3-06b-stale-seed42-v1 \
  --view all
```

## Portable CLI

The Python pipeline is independent of AWS and S3:

```bash
uv run --locked stale-qwen-eval prepare \
  --artifact-root /persistent/run-id \
  --cache-dir /persistent/hf-cache \
  --samples-per-type 25

uv run --locked stale-qwen-eval run \
  --artifact-root /persistent/run-id \
  --cache-dir /persistent/hf-cache \
  --view all \
  --attention-backend flash_sdpa \
  --resume
```

Execution wrappers for other compute environments can invoke these commands
without changing sampling, prompt construction, truncation, inference, or
reporting code.

## Reproducibility defaults

- Dataset: `STALEproj/STALE@617c51dc200b5ab09970834144c7e51c77959af0`
- Model: `Qwen/Qwen3-0.6B@167b8104f88905a951069f5f95f9776908da5f68`
- Scenario seed: `42`
- Sample: 25 Type I and 25 Type II scenarios
- Input/output budget: 32,256/512 tokens
- Inference: FP16, batch size one, greedy decoding, thinking disabled
- AWS attention: forced PyTorch Flash SDPA

## Tests

Run the offline suite:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Run the opt-in pinned-tokenizer integration test:

```bash
RUN_HF_INTEGRATION=1 uv run --locked python -m unittest discover -s tests -p 'test_hf_integration.py' -v
```

The AWS smoke run is the final integration test for the model, L4, Flash SDPA,
S3 permissions, both context views, checkpointing, and report generation.

## Attribution

STALE is CC BY 4.0, its LongMemEval-derived distractors retain the accompanying
MIT notice, and Qwen3-0.6B is Apache 2.0. The local `STALE/` checkout is reference
material only and is neither imported nor modified.
