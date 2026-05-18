# NPU productization implementation plan

## Goal

Lay down the docs, specs, and validation rails for BitNet-rs NPU
productization. NPU support is the governed low-power / warm-resident inference
lane, not a generic accelerator bucket. The first implementation target is Intel
AI Boost NPU on Lunar Lake 258V through OpenVINO NPU.

## Current evidence boundary

The `intel-npu` campaign has merged `NPU-002` through `NPU-011`. Those items
prove only:

- distinct Intel NPU backend identity;
- OpenVINO NPU runtime visibility fields and receipts;
- one tiny static OpenVINO graph smoke path;
- selected static BitNet-shaped RMSNorm, linear-projection, and FFN/ReLU2
  subgraph parity through OpenVINO NPU;
- live 258V OpenVINO 2026.1 visibility and selected static receipts.

They do not prove native bitnet-rs NPU inference, full BitNet inference, packed
QK256 decode, NPU acceleration, speedup, server readiness, broad dense-SLM
quality, CPU fallback as NPU proof, or full residency.

## Rollout sequence

### Phase 0: source-of-truth cleanup

PR: `docs(npu): add NPU source-of-truth map`

Add:

- `docs/npu/README.md`
- `plans/npu/README.md`
- `plans/npu/implementation-plan.md`

Update:

- `docs/specs/INDEX.md`
- `docs/tracking/campaigns/intel-npu/active.toml`
- generated campaign docs when required by `xtask`

Acceptance:

- docs only;
- no runtime claims;
- no route promotion;
- current `NPU-002` through `NPU-011` state is visible;
- Intel Lunar Lake / Intel AI Boost NPU through OpenVINO is the current target;
- Apple Neural Engine, Qualcomm Hexagon, and AMD Ryzen AI are future NPU families
  that do not inherit Intel proof.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-npu
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Phase 1: specs

Add the NPU productization proposal and route contracts:

1. `docs/proposals/BITNET-PROP-0007-npu-productization.md`
2. `docs/specs/BITNET-SPEC-NPU-ROUTE-CONTRACT.md`
3. `docs/specs/BITNET-SPEC-NPU-PROOF-LADDER.md`
4. `docs/specs/BITNET-SPEC-NPU-COLD-WARM-CACHE.md`
5. `docs/specs/BITNET-SPEC-NPU-DENSE-SLM.md`
6. `docs/specs/BITNET-SPEC-NPU-BITNET-SUBGRAPH.md`
7. `docs/specs/BITNET-SPEC-NPU-QUALITY.md`
8. `docs/specs/BITNET-SPEC-NPU-PERFORMANCE.md`
9. `docs/specs/BITNET-SPEC-NPU-RESIDENCY.md`
10. `docs/specs/BITNET-SPEC-NPU-STATUS-SURFACE.md`

Each spec PR must preserve the current not-claims and avoid runtime promotion.

### Phase 2: current-state validation and status UX

Add a current capability matrix, then teach user-facing status surfaces such as
`receipts explain`, `bitnet npu doctor --format json`, and model status to show
NPU visibility, proof family, fallback status, and not-claims.

### Phase 3: dense SLM NPU route

Define and test a Qwen2.5 0.5B Instruct OpenVINO INT4/NF4 symmetric export on
Lunar Lake NPU. Record export manifest, CPU control, GPU comparison, bounded NPU
ask, corpus results, generation-budget sensitivity, and fallback=false receipts.
Do not promote yet.

### Phase 4: cold/cache/warm/resident performance

Add cold, cached, warm-second-ask, and resident session profile runners. Record
pipeline construction, first-ever compile/infer, cache mode and hit/miss,
first-token latency, decode timing, steady tokens/sec, total response timing,
and power/thermal context or explicit unavailable reasons.

### Phase 5: exact-profile route promotion

Promote only an exact model + route + profile when quality passes,
fallback=false is recorded, selected NPU device is recorded, warm/resident timing
is acceptable against CPU/GPU comparators or low-power policy, and cold-start
caveats remain visible.

### Phase 6: BitNet NPU subgraph expansion

Continue static-shape BitNet graph-lowering experiments such as sub-layernorm,
RoPE static slice, embedding/gather, attention score, softmax, A x V, LM-head,
and prefill block candidates. Static subgraph parity remains not full inference.

### Phase 7: future NPU families

Document Apple Neural Engine, Qualcomm Hexagon, and AMD Ryzen AI as separate
research lanes. No proof inheritance from Intel Lunar Lake is allowed.

## CI and validation rails

Docs/spec PRs should run:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-npu
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Probe/status PRs should add package-specific tests for the touched CLI or probe
crates. Live OpenVINO NPU execution must remain opt-in and must not be added to
ordinary generic PR CI.

## Non-goals for docs/spec PRs

- Do not touch QK256 CPU, AVX, CUDA, or OpenCL kernels.
- Do not add live NPU execution to ordinary CI.
- Do not claim speedup, full inference, native NPU kernels, broad quality, or
  full residency.
