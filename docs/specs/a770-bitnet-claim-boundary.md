# A770 BitNet Claim Boundary

## Purpose

This spec defines what BitNet-rs may claim for real BitNet question-answer
usage on an Intel Arc A770. It is narrower than generic "GPU support" and
narrower than full device residency.

The goal is a useful, reproducible product path:

```text
official BitNet model contract
real tokenizer and prompt template
real question-answer prompts
CPU reference behavior
explicit A770 route
fallback_used=false where A770 is claimed
load, TTFT, prefill, decode, and resource receipts
same-device history
clear not-claims
```

## First Claim Target

The first A770 product claim is:

```text
BitNet b1.58 i2_s trusted partial A770 acceleration
```

This means:

- The official BitNet GGUF weights load through the strict model path.
- The tokenizer and prompt template are authoritative and hashable.
- Real prompts produce useful, intelligible answers.
- CPU reference behavior is correct enough to be the comparison baseline.
- The A770 route is concrete: `intel-arc-a770-opencl`.
- Claimed A770 operations run on the A770 with `fallback_used=false`.
- Timings and resources are measured under a named profile.
- The receipt says exactly what is not claimed.

This does not mean:

- All Intel GPUs are supported.
- All OpenCL devices inherit the A770 claim.
- Dense SLM or Gemma-class models are A770-supported.
- Selected attention is resident on A770.
- KV cache decode is resident on A770.
- Attention scores, softmax, or value mix are resident on A770.
- Full support-op residency or full device residency is complete.

## Claim Levels

Use these levels for A770 BitNet support.

| Level | Meaning | Public claim allowed |
|---|---|---|
| `unsupported` | No valid model, route, or kernel evidence. | No |
| `diagnostic` | Local evidence exists but is dirty, incomplete, or not claim-grade. | No |
| `load_proven` | Official weights and tokenizer/template load with hashes. | Limited load claim |
| `quality_proven` | Prompt suite and CPU/A770 behavior gates pass. | Quality claim only |
| `performance_proven` | Quality-gated benchmark and same-route history are claim-grade. | Trusted partial performance claim |
| `resident_proven` | A named resident operation is proved with transfer and fallback receipts. | Resident claim for that operation only |
| `complete` | All required support ops and residency gates are proved. | Full A770 completion claim |

Do not collapse `performance_proven`, `resident_proven`, and `complete`.

## Required Evidence

### Model Contract

Every claimable A770 BitNet run must identify:

- model ID
- official weight file
- weight hash
- tokenizer hash or embedded tokenizer identity
- chat template or prompt template
- quantization format
- max context
- stop-token policy

No model contract means no model support claim.

### Prompt and Answer Quality

Every claimable run must include prompt evidence that is hard to fake by
memorizing fixed names or fixed answers:

- rendered prompt hash
- token ID hash
- deterministic sampling configuration
- seeded prompt identities
- randomized surface names or slots where the prompt suite supports them
- category coverage
- paired context checks where the expected answer must change
- stop and repetition checks

Manual review cases may be stored as diagnostic evidence, but they cannot by
themselves promote an automated quality claim.

### Route Identity

A claimable A770 route must record:

```json
{
  "requested_backend": "intel-arc-a770",
  "selected_backend": "intel-arc-a770-opencl",
  "runtime_api": "opencl",
  "fallback_used": false,
  "resolved_device": {
    "name": "Intel(R) Arc(TM) A770 Graphics",
    "pci_device_id": "0x56A0",
    "vram_bytes": 17179869184
  }
}
```

A route for A750, Arc 140V, OpenVINO GPU, CUDA, Metal, or CPU is a different
route and needs its own evidence.

### Experience Measurements

A claimable A770 BitNet experience receipt must include:

- cold load timing
- warm load timing, when a load-speed claim is made
- time to first token
- prefill/input throughput
- decode/output throughput
- inter-token latency when claiming streaming quality
- peak RSS
- peak VRAM when available
- host/device transfer bytes when available
- kernel invocation counts for claimed A770 operations
- fallback status

Benchmarks may be stored even when incomplete. They are diagnostic until the
quality, route, model, resource, and history gates pass.

## Parent Benchmark and History

A performance claim requires a parent benchmark receipt with:

```text
repo.dirty=false
quality_passed=true
route_verified=true
model_contract_matched=true
fallback_used=false
claim_allowed=true
```

Same-device history requires two distinct receipts:

- distinct non-empty run IDs
- distinct receipt paths
- same device instance
- same model contract
- same backend
- same benchmark profile
- same kernel route

Comparing a receipt to itself is not history. Cross-vendor or cross-device
comparisons are useful diagnostics, not regression evidence.

## Current Not-Claims

Until separate receipts prove them, every A770 trusted-path receipt must keep
these explicit not-claims:

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

## Selected Attention Boundary

Selected attention is a separate research lane. It is not promoted by trusted
partial A770 acceleration.

Before selected attention can be promoted, the repo needs:

- a production-shaped selected-attention score rule
- decode parity at short and long decode lengths
- semantic quality unchanged
- stop behavior unchanged
- fallback_used=false
- selected attention actually used
- no replacement maps, output bias, CPU attention island, or diagnostic knobs

Resident KV, attention scores, softmax, and value mix must wait until selected
attention itself is claim-grade.

## PR Sequence

Build this path in small PRs:

1. Claim-boundary specs and docs.
2. Model contract and local asset hash verification.
3. A770 device route and kernel capability matrix.
4. Seeded prompt suite and anti-fakery validation.
5. Quality-gated benchmark receipt schema.
6. LLM experience receipt generation and verification.
7. Clean A770 parent benchmark rerun.
8. Two distinct same-device, same-route history receipts.
9. Claim ledger and generated dashboard.
10. Selected-attention fork decision, still non-promoting unless its gates pass.

Each PR must preserve the not-claims unless it is specifically the promotion PR
for one named capability.

The detailed implementation plan is:

```text
plans/a770-bitnet-claim-boundary-implementation.md
```

## Related Specs

- `docs/specs/intel-arc-a770-gpu-roadmap.md`
- `docs/hardware/intel-arc-a770-validation.md`
- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
