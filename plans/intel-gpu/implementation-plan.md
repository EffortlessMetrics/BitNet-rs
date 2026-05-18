# Intel GPU implementation plan

## Purpose

Intel GPU support must become a receipt-backed family of exact routes rather
than a generic GPU label. The plan keeps these lanes separate:

- A770 native OpenCL for BitNet I2_S/QK256 named-operation acceleration.
- A770 OpenVINO GPU as a reference runtime lane only.
- Arc 140V native OpenCL for Lunar Lake selected-device smoke/parity work.
- Arc 140V OpenVINO GPU for dense SLM candidate routing and profile-specific
  promotion.
- Intel NPU as a separate OpenVINO NPU proof family.
- CPU as the reference plate and fallback detector, not GPU proof.

## Global hard rules

- Do not promote runtime claims in documentation/specification PRs.
- Do not claim generic Intel GPU support.
- Do not claim OpenVINO GPU is native OpenCL.
- Do not claim Arc 140V proof is A770 proof.
- Do not claim A770 proof is Arc 140V proof.
- Do not claim dense SLM proof is BitNet QK256/I2_S proof.
- Do not claim full device residency from linears, graph smoke, or LLMPipeline
  output.
- Do not claim speedup without quality-gated, profile-specific benchmark
  receipts and same-device history.
- Do not change QK256 kernels, OpenCL kernels, model coverage, or route
  promotion state in documentation-only PRs.

## PR sequence

### PR 0: `docs(intel-gpu): add Intel GPU source-of-truth map`

Scope:

- Add `docs/intel-gpu/README.md`.
- Add `plans/intel-gpu/README.md`.
- Add `plans/intel-gpu/implementation-plan.md`.
- Update `docs/specs/INDEX.md` with the Intel GPU map.
- Update `docs/tracking/campaigns/intel-a770/active.toml` and
  `docs/tracking/campaigns/intel-258v-platform/active.toml` so their campaign
  constraints point to the shared Intel GPU proof-family boundary.

Acceptance:

- Documentation only.
- No route promotion.
- No receipt changes.
- No kernel, QK256, OpenCL, model-coverage, or CLI behavior changes.
- Campaign checks and generated campaign check pass, or any unavailable proof is
  recorded with a reason.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### PR 1: `docs(proposal): add Intel GPU productization proposal`

Add `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md` explaining why
Intel GPU exists as a product family and why selected-device, selected-model,
receipt-backed local inference is the value rather than generic GPU detection.

### PR 2: `docs(spec): add Intel GPU route contract`

Add:

- `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`.
- `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`.

Define concrete backend labels, runtime APIs, device identity requirements,
OpenCL platform/device recording, OpenVINO `GPU.X` recording, PCI IDs, and
fallback rules.

### PR 3: `docs(spec): add Intel GPU BitNet QK256 contract`

Add `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md` for the native Intel GPU
BitNet route, including official I2_S/QK256 semantics, scalar oracle parity,
OpenCL kernels, tail/stride behavior, tokenizer/template authority, and the rule
that diagnostic toy I2_S kernels cannot satisfy official QK256 proof.

### PR 4: `docs(spec): add Intel GPU dense SLM contract`

Add `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md` for Qwen2.5 0.5B Instruct
OpenVINO INT4 symmetric export on Lunar Lake Arc 140V GPU.0, including the
promotion ladder from export manifest through profile comparison and optional
server exact-profile proof.

### PR 5: `docs(spec): add Intel GPU quality/performance/residency contracts`

Add:

- `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`.
- `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`.
- `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`.

Define route-specific quality gates, failure taxonomy, exact performance
profiles and timing fields, promotion requirements, residency classes, and phase
residency tables.

### PR 6: `docs(spec): add Intel GPU status surface contract`

Add `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md` for status commands,
route matrices, `receipts explain`, and `gpu doctor --vendor intel` output.

### PR 7: `claims(a770): reconcile A770 proof state`

Inspect committed A770 receipts, `verify-receipts.py`, the device-kernel routing
matrix, and the A770 campaign. Keep routes diagnostic unless claim-grade receipts
are committed and the route matrix can point to them.

### PR 8: `receipts(a770): validate route matrix against receipts`

Add a validator that rejects promoted A770 routes without proof receipts,
strict selected backend identity, fallback-free receipts, not-claims, and
accepted speed/residency claim evidence.

### PR 9-14: A770 native OpenCL productization

Refresh selected-device identity, lock official QK256 scalar/oracle parity,
record claim-grade QK256 OpenCL receipts, add deterministic answer behavior
proof, add quality-gated profile timings, and promote only trusted partial
acceleration when all gates pass.

### PR 15-17: Lunar Lake Arc 140V OpenVINO GPU route

Classify corpus-v2 failures, fix profile-specific timing applicability, and
promote only exact profiles that pass quality, fallback, profile-timing,
comparator, telemetry, and claim-boundary gates.

### PR 18-20: Arc 140V native OpenCL BitNet-adjacent lane

Refresh selected-device OpenCL parity, add BitNet QK256 candidate fixtures, and
record whether Arc 140V should pursue native BitNet QK256 before A770. The
expected default is that A770 owns native BitNet OpenCL first.

### PR 21-23: Shared Intel GPU UX

Add route-family explanations to receipts, publish an Intel GPU capability
matrix, and add `bitnet gpu doctor --vendor intel --format json` for route
readiness and device/runtime diagnostics.
