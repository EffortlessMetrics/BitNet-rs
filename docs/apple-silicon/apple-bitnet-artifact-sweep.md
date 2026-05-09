# Apple BitNet Artifact Sweep

The Apple BitNet artifact sweep is the control plane for 1-bit / 1.58-bit
candidate validation on Apple Silicon. It uses the MacBook lane first for larger
artifact exploration, then sends accepted candidates back to the M4 Mac mini for
strict Apple CPU/NEON proof.

This is separate from the dense Qwen SLM path. Qwen2.5 0.5B Instruct is a
regular dense SLM and remains the practical Mac local-answer baseline. It does
not prove BitNet model quality, 1-bit layouts, QK256, or full Metal inference.

## Inputs

Primary references:

```text
ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml
docs/apple-silicon/bitnet-candidate-matrix.md
docs/model-artifacts/ANSWER_ARTIFACT_GATE.md
ci/model-artifacts/model-kernel-compatibility.toml
docs/architecture/reference-topology.md
```

## Candidate Order

```text
1. Official Microsoft BitNet b1.58 2B / 2B4T I2_S
2. 0.7B 1bitLLM/bitnet_b1_58-large
3. 3B 1bitLLM TL1/TL2 diagnostic route only
4. Falcon-E 1B/3B as secondary BitNet-like family evidence
```

The official Microsoft I2_S path is first because it is the official target and
the shared model-artifact gate records answer-ready evidence when external
Microsoft tokenizer pre-tokenizer authority is supplied. The 0.7B 1bitLLM model
is a smaller control candidate. The 3B model must not be treated as an I2_S
target unless the compatibility ledger changes.

## Required Record

Every artifact probe must record:

```text
source repo
revision
file
size_bytes
sha256
format
quantization
kernel route
tokenizer authority
pre-tokenizer authority
prompt template
reference runner
reference command
prompt outputs
acceptance or rejection
cleanup status
machine context
```

## Storage Policy

```text
download under cache or target
keep at least 8 GiB free after download when possible
never commit model binaries
delete rejected candidates unless a later item explicitly retains them
record exact SHA256 before accepting or rejecting a candidate
```

## Claim Boundary

An accepted artifact is not a backend success. It only means the candidate is
eligible for a strict backend local-answer proof. M4 BitNet local-answer claims
require a later receipt with:

```text
real model
real tokenizer authority
selected Apple backend
fallback_used=false or explicit fallback reason
generated text
generated token IDs
timing
quality result
```
