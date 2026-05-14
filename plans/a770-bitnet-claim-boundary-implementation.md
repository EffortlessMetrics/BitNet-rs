# A770 BitNet Productization Plan

## Goal

Make the A770 path usable for real BitNet question-answer workloads without
overclaiming hardware residency.

The target product claim is:

```text
BitNet b1.58 i2_s trusted partial A770 acceleration
```

The final full-residency claim is explicitly out of scope until separate gates
prove selected attention, resident KV, attention scores, softmax, attention
value mix, full support-op residency, and full device residency.

## Required End State

The trusted A770 BitNet lane is product-ready only when the repo can prove:

- official BitNet GGUF weights load through a strict model contract
- tokenizer and prompt template are authoritative
- real prompts produce useful answers
- CPU reference behavior is correct
- A770 route identity is concrete and device-specific
- fallback is false where A770 is claimed
- claimed A770 operations are named
- load, time-to-first-token, input throughput, output throughput, and resources
  are measured
- benchmark claims are quality-gated
- two distinct same-device, same-route history receipts exist
- generated docs and CLI summaries preserve not-claims

## Non-Claims Until Promoted

Every PR in this plan must preserve these not-claims unless it is the explicit
promotion PR for that named capability:

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

## Stop Rules

- Do not add selected-attention kernels before a reviewed selected-attention
  score rule exists.
- Do not treat a CLI smoke as a benchmark claim.
- Do not treat a dirty-worktree run as claim-grade.
- Do not treat a same-receipt comparison as history.
- Do not inherit A770 proof to A750, Arc 140V, OpenVINO GPU, CUDA, Metal, or
  CPU lanes.
- Do not benchmark unsupported model families as A770 support.

## PR 0 - Specs and Plan

Purpose: make the lane followable before implementation.

Files:

```text
docs/specs/a770-bitnet-claim-boundary.md
docs/specs/intel-arc-a770-gpu-roadmap.md
docs/hardware/intel-arc-a770-validation.md
plans/a770-bitnet-claim-boundary-implementation.md
```

Acceptance:

```text
the spec defines claim levels
the spec defines required evidence
the spec defines current not-claims
the plan lists PR-by-PR work
no runtime behavior changes
```

Verification:

```powershell
git diff --check -- docs/specs/a770-bitnet-claim-boundary.md docs/specs/intel-arc-a770-gpu-roadmap.md docs/hardware/intel-arc-a770-validation.md plans/a770-bitnet-claim-boundary-implementation.md
```

## PR 1 - Model Contract and Asset Proof

Purpose: make the official BitNet model identity claimable.

Files:

```text
docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml
xtask/src/model_contract.rs
xtask/src/main.rs
```

Required command:

```powershell
cargo run --locked -p xtask --no-default-features -- model-contract lint --format json
```

Required evidence:

```text
model_id matches the official BitNet GGUF model
local GGUF hash verified
local tokenizer hash verified
chat template recorded
stop-token policy recorded
max context recorded
asset_hashes_verified=true
```

Not enough:

```text
model file exists
tokenizer file exists
download command succeeded
```

## PR 2 - A770 Route and Kernel Capability Matrix

Purpose: prevent broad or inherited A770 claims.

Files:

```text
ci/hardware/device-kernel-routing.toml
ci/hardware/amd-5700x-intel-a770/a770-kernel-capability-matrix.json
xtask/src/hardware.rs or equivalent
```

Required commands:

```powershell
cargo run --locked -p xtask --no-default-features -- hardware a770 kernel-capability-check --format json
cargo run --locked -p xtask --no-default-features -- hardware route resolve --format json
```

Required evidence:

```text
A770 route names a concrete device
route has a concrete kernel variant
fallback_allowed=false for claimable routes
proof receipts listed for claimable kernels
unsupported dense/Gemma routes fail closed
```

Not enough:

```text
runtime says OpenCL exists
device name contains Intel
kernel is compatible with same-family hardware
```

## PR 3 - Seeded Prompt Suite and Anti-Fakery Validation

Purpose: prove broad prompt behavior without fixed-name overfitting.

Files:

```text
ci/prompt-suites/seeded-v1.toml
xtask/src/prompt_suite.rs or equivalent
```

Required commands:

```powershell
cargo run --locked -p xtask --no-default-features -- prompt-suite verify --suite ci/prompt-suites/seeded-v1.toml --format json
cargo run --locked -p xtask --no-default-features -- prompt-suite render --suite ci/prompt-suites/seeded-v1.toml --model-contract docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml --format json
```

Required evidence:

```text
required categories present
deterministic sampling for claimable cases
prompt hashes present
token ID hashes present
seeded surface-name slots present
answer-bound surfaces present
paired context cases check answer changes
manual-review cases cannot promote automated quality claims
```

Not enough:

```text
one real question
one benchmark prompt
prompt strings without token hashes
manual review only
```

## PR 4 - Quality-Gated Benchmark Receipt

Purpose: ensure speed evidence cannot promote without answer quality.

Files:

```text
ci/benchmarks/schemas/bench-run.schema.json
ci/benchmarks/profiles.toml
xtask/src/bench.rs or equivalent
```

Required commands:

```powershell
cargo run --locked -p xtask --no-default-features -- bench verify-receipt --receipt target/bench-runs/profile-cli-stage.json --format json
cargo run --locked -p xtask --no-default-features -- bench compare --require-same-route --require-claim-ready --format json
```

Required evidence:

```text
quality_passed=true before benchmark_claim_allowed=true
repo.dirty=false for claim-grade runs
fallback_used=false for A770 claims
model contract matched
kernel route matched
profile matched
resource fields present
```

Not enough:

```text
tokens per second only
single smoke timing
dirty workspace benchmark
```

## PR 5 - CLI Stage Receipts and Proof Summary

Purpose: let normal user runs expose the claim boundary.

Files:

```text
crates/bitnet-cli/src/main.rs or command modules
```

Required evidence:

```text
backend printed
model contract printed
fallback status printed
claimed A770 ops printed
not-claims printed
prompt/token hash identity available in JSON
load and TTFT stages available in JSON
```

Not enough:

```text
human text says A770
no JSON receipt
no fallback status
```

## PR 6 - LLM Experience Receipt

Purpose: combine model, route, quality, timing, resource, and not-claim evidence.

Files:

```text
ci/benchmarks/schemas/llm-experience-run.schema.json
xtask/src/llm_experience.rs or equivalent
docs/benchmarks/llm-experience.md
```

Required commands:

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience run --format json
cargo run --locked -p xtask --no-default-features -- llm-experience verify --receipt target/llm-experience/a770-bitnet-profile-stage.json --format json
```

Required evidence:

```text
model contract matched
asset hashes verified
route verified
quality passed
fallback=false
load complete
TTFT complete
input speed complete
output speed complete
resource envelope complete
critical not-claims present
```

Not enough:

```text
benchmark receipt alone
CLI smoke alone
diagnostic supplement alone
```

## PR 7 - Clean A770 Parent Benchmark

Purpose: produce the first claim-grade parent benchmark.

Operator runbook:

```text
docs/hardware/amd-5700x-intel-a770-clean-claim-rerun.md
```

Required sequence:

```powershell
git status --short
cargo run --locked -p xtask --no-default-features -- llm-experience profile-cli-plan --format json
# run generated CLI command
cargo run --locked -p xtask --no-default-features -- bench from-cli-stage --format json
cargo run --locked -p xtask --no-default-features -- bench verify-receipt --receipt target/bench-runs/profile-cli-stage.json --format json
```

Required evidence:

```text
git status is empty before run
parent benchmark claim_allowed=true
repo.dirty=false
quality_passed=true
fallback_used=false
route_verified=true
profile matched
```

Not enough:

```text
claim_allowed_if_repo_clean=true
dirty diagnostic run
```

## PR 8 - Same-Device Same-Route History

Purpose: prove stability and prevent self-comparison.

Required sequence:

```powershell
cargo run --locked -p xtask --no-default-features -- llm-experience publish --format json
# repeat full clean run for second distinct receipt
cargo run --locked -p xtask --no-default-features -- llm-experience compare --require-same-route --require-claim-ready --format json
```

Required evidence:

```text
two distinct run IDs
two distinct receipt paths
same device instance
same model contract
same backend
same profile
same kernel route
classification=same_device_regression_signal
history_scope_ready=true
```

Not enough:

```text
same receipt compared to itself
same route but diagnostic parent benchmark
cross-device comparison
```

## PR 9 - Claim Ledger and Generated Dashboard

Purpose: turn verified receipts into public claims.

Files:

```text
ci/claims/claim-ledger.json
ci/claims/correctness-ledger.json
ci/claims/efficiency-ledger.json
docs/claims.md
docs/model-support.md
docs/benchmarks.md
```

Required commands:

```powershell
cargo run --locked -p xtask --no-default-features -- claims verify --format json
cargo run --locked -p xtask --no-default-features -- claims docs --check --format json
```

Required evidence:

```text
trusted partial A770 BitNet claim is ledger-derived
not-claims are present in docs
unsupported model families stay unsupported
benchmark claims point to quality-gated receipts
```

Not enough:

```text
README prose
manual claim table
receipt exists but ledger ignores it
```

## PR 10 - Selected Attention Decision

Purpose: keep selected attention separate from the trusted product path.

Allowed outcomes:

```text
defer selected attention
continue research with a reviewed score-rule design
```

Promotion requires:

```text
production-shaped selected-attention score rule
decode_32 exact
decode_64 exact
decode_128 exact
semantic quality unchanged
stop behavior unchanged
fallback_used=false
selected attention actually used
no diagnostic knobs
```

Not enough:

```text
captured value target without implementation rule
dirty diagnostic decode
another score-row variant
replacement map or output bias
```

## Done Criteria

The A770 trusted BitNet path is done only when:

```text
model contract verified
prompt suite claim-ready
route verified
quality passed
fallback=false
parent benchmark claim_allowed=true
two same-device same-route history receipts exist
claim ledger consumes the evidence
docs and CLI expose claim boundary
selected attention and full residency remain explicit not-claims
```

If any line is missing, the lane is still in progress.
