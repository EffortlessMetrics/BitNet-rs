# AMD 5700X + Intel Arc A770 Clean Claim Rerun

This runbook is for producing claim-grade BitNet A770 trusted-path evidence.
It must only be used after the claim-boundary specs, model contract rail, route
rail, prompt-suite rail, benchmark rail, LLM-experience rail, history rail, and
claim-ledger rail are merged or deliberately stacked for review.

The target claim remains narrow:

```text
BitNet b1.58 i2_s trusted partial A770 acceleration
```

It is not a selected-attention, resident-KV, full-support-residency, full-device
residency, SLM, or Gemma-class claim.

## Stop Before Running

Do not run a claim-grade sequence if any of these are true:

- `git status --short` is not empty.
- The local branch does not include the reviewed proof rails.
- `claims verify` fails.
- The generated CLI stage would execute on CPU while declaring an A770 route.
- The A770 route is still diagnostic or not claimable for the operation being
  claimed.
- The quality receipt is missing or failed.
- Resource fields needed by the benchmark profile are missing.

Diagnostic runs are allowed, but they must not be published as claim-grade
history or promoted in the claim ledger.

## Preflight

Run from a clean worktree:

```powershell
git status --short

cargo run --locked -p xtask --no-default-features -- model-contract lint --format json

cargo run --locked -p xtask --no-default-features -- hardware a770 kernel-capability-check --format json

cargo run --locked -p xtask --no-default-features -- hardware route resolve --format json

cargo run --locked -p xtask --no-default-features -- prompt-suite verify `
  --suite ci/prompt-suites/seeded-v1.toml `
  --format json

cargo run --locked -p xtask --no-default-features -- prompt-suite render `
  --suite ci/prompt-suites/seeded-v1.toml `
  --model-contract docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml `
  --format json

cargo run --locked -p xtask --no-default-features -- claims verify --format json
```

Expected before any promotion:

```text
claims verify passes
promoted_a770_claims = []
selected attention remains diagnostic/unclaimed
full residency remains unsupported/unclaimed
```

## Generate The Profile CLI Plan

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience profile-cli-plan `
  --profile prefill_512_decode_64 `
  --backend intel-arc-a770-opencl `
  --device-slug amd-5700x-intel-a770 `
  --kernel-route a770.bitnet.i2s.qk256 `
  --output target/llm-experience/profile-cli-stage-plan.json `
  --format json
```

Inspect the plan before executing it:

```powershell
$plan = Get-Content target/llm-experience/profile-cli-stage-plan.json | ConvertFrom-Json
$plan.prompt_identity.prompt_token_count
$plan.profile.max_new_tokens
$plan.backend
$plan.kernel_route.route_id
$plan.cli_command
```

Required values for this profile:

```text
prompt_token_count = 512
max_new_tokens = 64
backend = intel-arc-a770-opencl
kernel_route.route_id = a770.bitnet.i2s.qk256
cli_command contains --proof-model-contract
cli_command contains --proof-kernel-route
```

## Run The CLI Stage

Execute the generated command exactly:

```powershell
$cmd = @($plan.cli_command)
& $cmd[0] @($cmd[1..($cmd.Length - 1)])
```

Then inspect the CLI-stage receipt:

```powershell
$cli = Get-Content target/llm-experience/profile-cli-stage.json | ConvertFrom-Json
$cli.proof_summary.requested_backend
$cli.proof_summary.selected_backend
$cli.proof_summary.execution_backend
$cli.proof_summary.execution_backend_matched
$cli.proof_summary.fallback_used
$cli.proof_summary.kernel_route.route_id
$cli.proof_summary.kernel_route.claimable
```

Hard stop if:

```text
execution_backend != intel-arc-a770-opencl
execution_backend_matched != true
fallback_used != false
kernel_route.claimable != true
```

A CPU CLI-stage receipt that declares an A770 route is diagnostic evidence only.

## Build The Parent Benchmark Receipt

Use the quality receipt from the relevant prompt/quality gate:

```powershell
cargo run --locked -p xtask --no-default-features -- bench from-cli-stage `
  --plan target/llm-experience/profile-cli-stage-plan.json `
  --cli-stage-receipt target/llm-experience/profile-cli-stage.json `
  --model-contract docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml `
  --quality-receipt target/quality/a770-bitnet-quality.json `
  --quality-passed `
  --output target/bench-runs/profile-cli-stage.json `
  --format json

cargo run --locked -p xtask --no-default-features -- bench verify-receipt `
  --receipt target/bench-runs/profile-cli-stage.json `
  --require-claimable `
  --format json
```

Required parent benchmark state:

```text
quality_passed = true
repo.dirty = false
fallback_used = false
profile_matched = true
model_contract_matched = true
route_verified = true
route_claimable = true
resource_envelope_complete = true
benchmark_claim_allowed = true
```

If `--require-claimable` fails, stop. Do not publish claim-grade history.

## Build And Verify The LLM Experience Receipt

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience run `
  --bench-receipt target/bench-runs/profile-cli-stage.json `
  --cli-stage-receipt target/llm-experience/profile-cli-stage.json `
  --output target/llm-experience/a770-bitnet-profile-stage.json `
  --format json

cargo run --locked -p xtask --no-default-features -- llm-experience verify `
  --receipt target/llm-experience/a770-bitnet-profile-stage.json `
  --require-claimable `
  --format json
```

Required experience state:

```text
claim_allowed = true
classification = performance_proven
quality.passed = true
backend.fallback_used = false
kernel_route.route_verified = true
load.complete = true
ttft.complete = true
input_speed.complete = true
output_speed.complete = true
resource_envelope.complete = true
critical not-claims present
```

## Publish Two Distinct Same-Route Receipts

Publish the first clean receipt:

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience publish `
  --receipt target/llm-experience/a770-bitnet-profile-stage.json `
  --history-root target/llm-experience-history `
  --format json
```

Repeat the full clean sequence for a second distinct run. The second receipt
must have a different run ID and a different receipt path.

Then compare:

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience compare `
  --history-root target/llm-experience-history `
  --device amd-5700x-intel-a770 `
  --profile prefill_512_decode_64 `
  --require-same-route `
  --require-claim-ready `
  --format json
```

Required history state:

```text
distinct_run_ids = true
distinct_paths = true
same_device = true
same_backend = true
same_route = true
claim_ready_pair = true
comparison_classification = same_device_same_route_regression
```

## Claim Ledger Check

Only after parent benchmark, experience receipt, and same-route history are
claim-ready:

```powershell
cargo run --locked -p xtask --no-default-features -- claims verify `
  --llm-experience-receipt target/llm-experience/a770-bitnet-profile-stage.json `
  --format json

cargo run --locked -p xtask --no-default-features -- claims docs --check --format json
```

Promotion PRs must keep these not-claims unless separately proved:

```text
selected_attention_residency
resident_kv_decode
attention_scores_residency
softmax_residency
attention_value_mix_residency
full_support_op_residency
full_device_residency
completion
```
