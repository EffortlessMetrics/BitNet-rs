# SmolLM2 360M

SmolLM2 360M is structurally valid, but it is not CPU answer-ready and has no
CUDA readiness claim. The current blocker is not another CUDA proof. It is a
fresh same-prompt external reference comparator that can localize the strict CPU
wrong-first-token behavior before any CPU answer, all-layer accelerator, or CUDA
one-token work starts.

## Current Source Of Truth

- `CUDA-MODEL-SMOLLM2-001` is already merged in the NVIDIA campaign. It records
  the exact SmolLM2 360M artifact contract and keeps CPU, CUDA, product, speed,
  server, full-residency, and inherited-proof claims false.
- `CUDA-MODEL-SMOLLM2-002` is already merged in the NVIDIA campaign. It syncs the
  CUDA/model coverage view to the strict CPU preflight blocker and is not a
  pending same-prompt comparator item.
- `SLM-CPU-020` reached strict CPU one-token generation with `fallback_used=false`
  but failed the math quality gate by generating `The`.
- `SLM-CPU-021` diagnosed the wrong-first-token state as unresolved from the
  committed reference-runner, prompt/tokenizer, and strict CPU retry evidence.
- `SLM-CPU-022` added the SmolLM2 first-token/top-k comparator contract and
  `reference-compare` fixture coverage. It is support only: a fresh same-prompt
  external reference comparator artifact remains missing.

The model coverage matrix is authoritative for the claim state. It keeps
`dense_smollm2_360m_candidate` at `structurally_valid` with CPU answer readiness,
accelerator readiness, product CLI readiness, server readiness, speedup, full
residency, dense CUDA proof, and BitNet QK256 proof all false.

## Work item: SMOLLM2-REF-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/slm-cpu/active.toml`
Blocks: SMOLLM2-CPU-001
Blocked by: fresh same-prompt external reference comparator artifact

### Goal

Capture or ingest the SmolLM2 same-prompt first-token/top-k or checkpoint
comparator required by the `SLM-CPU-022` contract.

### Production delta

Classifies the current wrong-first-token blocker as prompt policy, tokenizer,
shared dense CPU math, sampler/logit divergence, or reference mismatch.

### Non-goals

No CPU answer-ready promotion. No all-layer CUDA plan. No CUDA execution. No
speedup, server, product CLI, broad dense GGUF, or BitNet QK256 claim.

### Acceptance

- Comparator uses the exact SmolLM2 360M artifact already covered by the model
  matrix.
- Comparator preserves the same prompt text, tokenizer policy, prompt template,
  BOS/EOS policy, deterministic generation settings, and first-token/top-k or
  checkpoint evidence.
- `reference-compare` validates the artifact shape and records whether the
  mismatch is localized or still unresolved.
- Model coverage remains below `reference_good` unless the comparator evidence
  actually earns that promotion.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- reference-compare --artifact ci\slm-cpu\windows-9950x3d-rtx5070ti\2026-05-16\smollm2-360m-reference-comparator.json --json-out target\slm-cpu\smollm2-360m-reference-comparator-validation.json
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Remove the comparator artifact and validation output, and leave SmolLM2 at the
current `structurally_valid` tier.

## Work item: SMOLLM2-CPU-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/slm-cpu/active.toml`
Blocks: SMOLLM2-CUDA-PLAN-001
Blocked by: SMOLLM2-REF-001

### Goal

Retry SmolLM2 CPU answer sanity only after the same-prompt comparator localizes
or resolves the wrong-first-token blocker.

### Production delta

Promotes SmolLM2 beyond `structurally_valid` only if a bounded CPU quality gate
passes with strict artifact, tokenizer, prompt, backend, and fallback evidence.

### Non-goals

No accelerator, speedup, server, full-residency, broad dense GGUF, or BitNet
QK256 claim.

### Acceptance

- CPU answer retry uses a bounded corpus and the same exact artifact.
- Receipt records selected CPU backend, tokenizer authority, prompt policy,
  generated IDs/text, quality gate, and `fallback_used=false`.
- Model coverage updates only if the CPU proof earns the next tier.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Keep or restore SmolLM2 to `structurally_valid`.

## Work item: SMOLLM2-CUDA-PLAN-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: SMOLLM2-CUDA-001
Blocked by: SMOLLM2-CPU-001

### Goal

Create the SmolLM2 all-layer accelerator plan after CPU answer sanity passes.

### Production delta

Maps the exact SmolLM2 artifact boundaries, dense SLM route requirements,
unsupported operations, and required CUDA fixtures before any strict one-token
CUDA proof.

### Non-goals

No CUDA execution claim. No product CLI, speedup, server, full-residency, broad
dense GGUF, Qwen inheritance, or BitNet QK256 claim.

### Acceptance

- Plan names every layer boundary and unsupported operation before one-token
  CUDA proof.
- Plan keeps dense SLM CUDA separate from BitNet QK256 and from Qwen2.5/Qwen3
  evidence.
- Model coverage remains below accelerator answer-ready.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert the plan and keep SmolLM2 below accelerator answer-ready.

## Work item: SMOLLM2-CUDA-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: short decode and warm-session proof
Blocked by: SMOLLM2-CUDA-PLAN-001

### Goal

Capture one-token strict CUDA proof for SmolLM2 after CPU readiness and the
all-layer plan are complete.

### Production delta

Receipt proves selected backend, selected route, fallback rejection, and
one-token evidence for the exact SmolLM2 artifact.

### Non-goals

No product CLI, speedup, server readiness, full-residency, broad dense GGUF,
Qwen inheritance, or BitNet QK256 claim.

### Acceptance

- Strict CUDA receipt names `nvidia-rtx-5070-ti-cuda` and a dense SLM route, not
  generic CUDA.
- Receipt reports `fallback_used=false`.
- Receipt explain reports SmolLM2-only proof and keeps inherited Qwen and BitNet
  claims false.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <smollm2> "..."
```

### Rollback

Revert route changes and keep the model below accelerator answer-ready.
