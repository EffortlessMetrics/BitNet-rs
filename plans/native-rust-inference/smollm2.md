# SmolLM2 360M

SmolLM2 360M is structurally valid. CPU answer readiness is blocked by
same-prompt/reference comparator work.

## Work item: CUDA-MODEL-SMOLLM2-002

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-SMOLLM2-003
Blocked by: native inference plan

### Goal

Capture a same-prompt comparator for SmolLM2.

### Production delta

Classifies mismatch source as prompt policy, tokenizer, shared dense CPU math,
or reference mismatch.

### Non-goals

No CUDA claim.

### Acceptance

Comparator uses same prompt, tokenizer policy, prompt template, and
first-token/top-k evidence.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert comparator artifacts.

## Work item: CUDA-MODEL-SMOLLM2-003

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-SMOLLM2-004
Blocked by: CUDA-MODEL-SMOLLM2-002

### Goal

Retry CPU answer sanity after comparator evidence.

### Production delta

Promote to CPU answer-ready only if the answer gate passes.

### Non-goals

No accelerator claim.

### Acceptance

Model coverage row updates only if CPU proof passes.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

### Rollback

Keep or restore SmolLM2 to structurally valid.

## Work item: CUDA-MODEL-SMOLLM2-004

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-SMOLLM2-005
Blocked by: CUDA-MODEL-SMOLLM2-003

### Goal

Create the SmolLM2 all-layer accelerator plan.

### Production delta

Plan maps model boundaries, unsupported operations, and required dense CUDA
route work.

### Non-goals

No CUDA proof.

### Acceptance

Plan names blockers before one-token CUDA proof.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert the plan.

## Work item: CUDA-MODEL-SMOLLM2-005

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: short decode and warm-session proof
Blocked by: CUDA-MODEL-SMOLLM2-004

### Goal

Capture one-token strict CUDA proof for SmolLM2.

### Production delta

Receipt proves selected backend, selected route, fallback rejection, and
one-token evidence for the exact SmolLM2 artifact.

### Non-goals

No product CLI, speedup, or server claim.

### Acceptance

Receipt explain reports SmolLM2 proof only and keeps inherited Qwen/BitNet
claims false.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <smollm2> "..."
```

### Rollback

Revert route changes and keep the model below accelerator answer-ready.
