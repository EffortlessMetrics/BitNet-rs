# Intel Core Ultra 7 258V Validation Profile

## Purpose

This file defines the validation bundle for the Core Ultra 7 258V Lunar Lake laptop. The machine is a tri-device validation platform:

| Device | Proof lane |
|---|---|
| CPU | `intel-258v-cpu-avx2` / `cpu-avx2` BitNet CPU lead, strict validation, and fallback |
| Integrated GPU | `intel-arc-140v-opencl` and `intel-arc-140v-openvino-gpu` |
| NPU | `intel-npu-openvino` / `intel_258v_npu_openvino` |

The 258V laptop should not be treated as a single generic Intel accelerator.

The 258V CPU lane is the BitNet CPU lead. It owns strict real-GGUF BitNet CPU validation, scalar-vs-AVX2 answer parity, phase receipts, and same-machine CPU reference artifacts used by the Arc 140V and Intel NPU lanes. The i5-8250U is now the SLM CPU lead and a legacy/low-power BitNet comparison lane; it does not block new BitNet CPU work.

Platform roadmap:

```text
docs/specs/intel-lunar-lake-258v-platform-roadmap.md
```

## Expected Platform Facts

Expected Core Ultra 7 258V profile:

| Component | Expected value |
|---|---|
| Platform | Lunar Lake |
| CPU | 8 cores / 8 threads |
| CPU topology | 4 P-cores + 4 low-power E-cores |
| CPU backend | CPU AVX2 |
| Memory | Up to 32GB LPDDR5X-8533 shared |
| Integrated GPU | Intel Arc 140V |
| GPU peak | 64 INT8 TOPS |
| GPU PCI device ID | 0x64A0 |
| NPU | Intel AI Boost NPU |
| NPU peak | 47 INT8 TOPS |
| Overall platform peak | 115 INT8 TOPS |

The CPU supports AVX2, but this profile should not assume AVX-512.

## Buildout Contract

The detailed buildout plan for backend identity, Arc 140V probing, platform receipts, and 258V CPU validation is maintained in:

```text
docs/specs/intel-lunar-lake-258v-buildout-plan.md
```

Use this validation profile for manual machine-fact collection. Use the buildout plan for implementation scope and acceptance criteria.

## Required Machine Facts

Record these before moving any 258V hardware lane beyond `scaffold`:

| Fact | Why it matters |
|---|---|
| Native Windows, native Linux, or WSL | Do not assume WSL can see the NPU. |
| OpenVINO version | NPU and GPU plugin support is version-sensitive. |
| Intel NPU driver version | Required for NPU receipts. |
| OpenVINO `available_devices` | Should show CPU/GPU/NPU when fully visible. |
| Arc 140V OpenCL visibility | Determines iGPU kernel lane viability. |
| Level Zero visibility | Future lower-level/SYCL path. |
| OpenVINO `GPU.0` full name | Confirms Arc 140V reference target. |
| NPU `compile_model(..., "NPU")` success | Compile path proof. |
| Static-shape tiny graph result | Runtime smoke proof. |
| Shared memory pressure | 32GB LPDDR5X is shared by CPU/GPU/NPU. |
| Power mode / thermal profile | Laptop results depend heavily on power policy. |

## Low-Power Battery Runbook

The current `low_power` route-policy blocker is real battery-mode telemetry and
energy-proxy evidence. The operator checklist for that physical run is:

```text
docs/hardware/intel-258v-low-power-battery-runbook.md
```

Use that runbook before attempting `LNL258V-POWER-006`. It records the strict
AC/charging stop rule, required before/after battery telemetry receipts,
route/profile sample requirements, energy-proxy refresh, artifact refresh, and
promotion gates. AC-only samples remain blocker evidence and must not be
renamed into battery-mode artifacts or used for a power-advantage claim.

## Claim Boundary

- CPU AVX2 correctness does not count as Arc 140V or NPU execution.
- Arc 140V OpenCL execution does not count as NPU execution.
- OpenVINO NPU execution does not count as native OpenCL GPU execution.
- OpenVINO `GPU.0` smoke does not prove BitNet OpenCL kernel acceleration.
- OpenVINO `NPU` smoke does not prove full BitNet inference.
- CPU or GPU fallback cannot count as NPU execution.
- 258V CPU proof is the first priority on this platform; NPU and Arc proofs must compare against the 258V CPU reference when they make BitNet-adjacent parity claims.
- 258V CPU changes may own BitNet CPU sequencing when explicitly scoped; accelerator PRs must not reshape CPU dispatch or QK256 CPU kernels.
- Arc 140V visibility must preserve `requested_backend`, `selected_backend`, runtime API, exact device identity evidence, and `fallback_used=false`; generic Intel GPU visibility is not enough.

## Platform Probe Bundle Artifacts

`LNL258V-002` documents the same-machine probe bundle that later runs should
write under `ci/hardware/intel-258v/<date>/`. These paths are examples and
placeholders for future evidence; adding them to the docs does not commit a
real machine artifact and does not prove runtime execution.

```text
ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/platform-probe-cli.json
ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json
ci/hardware/intel-258v/YYYY-MM-DD/npu-openvino-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/platform-comparison-index.json
```

The bundle must keep each lane independently addressable:

| Artifact | Proof stage | Scope | Claim boundary |
|---|---|---|---|
| `platform-probe.json` | `runtime_detected` | OS, CPU, memory, power, OpenVINO device list, shared platform context | Machine visibility only |
| `platform-probe-cli.json` | `runtime_detected` | CLI-emitted OS, CPU, memory, OpenVINO CPU/GPU/NPU, Arc/NPU identity, and missing-runtime state | Machine visibility only |
| `arc-140v-runtime-probe.json` | `runtime_detected` | Arc 140V OpenCL, Level Zero, OpenVINO `GPU.0`, exact device identity | No OpenCL kernel execution claim |
| `arc-140v-openvino-gpu-smoke.json` | `kernel_smoke_tested` | Tiny static OpenVINO `GPU.0` graph execution with Arc 140V identity and CPU expected-output comparison | No native OpenCL, BitNet, QK256, or acceleration claim |
| `npu-openvino-runtime-probe.json` | `runtime_detected` | OS NPU evidence, OpenVINO `NPU`, driver/compiler/memory properties | No graph execution claim |
| `platform-comparison-index.json` | index only | Links corrected BitNet CPU, dense SLM CPU, Arc 140V, and NPU artifacts from the same machine/date | No independent proof claim |

The comparison index should preserve artifact paths and lane identities so later
CPU, GPU, and NPU receipts can be compared without inferring cross-lane proof:

```json
{
  "machine_id": "intel-258v",
  "date": "YYYY-MM-DD",
  "proof_stage": "runtime_detected",
  "artifacts": {
    "platform": "ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json",
    "arc140v": "ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-runtime-probe.json",
    "arc140v_openvino_gpu": "ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json",
    "npu": "ci/hardware/intel-258v/YYYY-MM-DD/npu-openvino-runtime-probe.json"
  },
  "lanes": {
    "cpu": "intel-258v-cpu-avx2",
    "gpu": "intel-arc-140v-opencl",
    "openvino_gpu": "intel-arc-140v-openvino-gpu",
    "npu": "intel-npu-openvino"
  },
  "fallback_used": false
}
```

The bundle does not prove BitNet inference, Arc 140V execution, OpenVINO NPU
graph execution, parity, or benchmark performance.

### Current Comparison Index

`LNL258V-COMPARE-004` refreshes the same-machine comparison index after the
corrected BitNet CPU reference bundle and dense Qwen SLM CPU receipts landed,
while preserving the selected NPU FFN/ReLU2 subgraph parity and Arc 140V native
OpenCL CPU/iGPU parity receipts:

```text
ci/hardware/intel-258v/2026-05-08/platform-comparison-index.json
```

The index links the corrected BitNet CPU reference bundle, post-fix BitNet
answer/parity/phase receipts, the dense Qwen SLM manifest plus answer/phase
receipts, Arc 140V OpenVINO GPU smoke and native OpenCL CPU-reference parity,
and the NPU OpenVINO runtime/smoke/RMSNorm/linear/FFN selected subgraph parity
receipts. It is not a proof artifact by itself. A lane claim is allowed only
when the cited lane artifact independently proves it.

The comparison index may claim that the repository has a same-machine artifact
map for corrected BitNet CPU, dense SLM CPU, Arc 140V, and Intel NPU receipts.
It must not claim cross-lane performance comparability, Arc 140V BitNet
inference, Intel NPU BitNet inference, dense SLM inference on accelerators,
QK256 accelerator decode, acceleration, or CPU fallback as accelerator proof.

### CPU Phase Receipts After Prompt Fix

`CPU258V-030` refreshes the 258V CPU warm-session phase surface after the
metadata-authoritative BitNet prompt-policy fix and the fixed-corpus
scalar/AVX2 answer pass:

```text
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session-after-prompt-fix.json
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session-after-prompt-fix-profiles/prefill_512.json
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session-after-prompt-fix-profiles/decode_128.json
```

These receipts record a release-built strict CPU warm session with real GGUF
loading, explicit tokenizer resolution, corrected prompt policy,
`i2_s-avx2-reference`, `prefill_512`, `first_token`, `decode_128`, and
`fallback_used=false`. They are CPU phase evidence only. They do not claim
speedup, sustained throughput, broad BitNet answer quality, Arc 140V execution,
Intel NPU execution, QK256 changes, or full model correctness.

### Corrected CPU Reference Bundle

`CPU258V-031` replaces the semantic-debug CPU reference index with a corrected
CPU reference bundle that links the prompt-policy fix, post-fix HF prompt/token
parity, post-fix scalar and AVX2 answer-corpus receipts, post-fix answer
parity, and post-fix phase receipts:

```text
ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-after-semantic-fix.json
```

This bundle is the current 258V CPU reference index for future Arc 140V and
Intel NPU comparison work after it lands. It does not add runtime behavior and
does not prove broad BitNet answer quality, speedup, sustained throughput,
external first-token logits parity, Arc 140V execution, Intel NPU execution,
QK256 changes, or full model correctness.

### Dense SLM Artifact Manifest

`SLM258V-001` adds the first Lunar Lake dense SLM artifact manifest:

```text
ci/hardware/intel-258v/2026-05-08/slm-artifact-manifest.json
```

The manifest selects `qwen2.5-0.5b-instruct-q8_0` as the first 258V dense SLM
CPU smoke candidate and links the CPU answer-smoke receipt below plus the
clean-provenance rerun. It records the exact GGUF repository, revision, file,
SHA256, architecture, quantization, tokenizer metadata, prompt template, context
length, and reference-output expectations inherited from the existing dense Qwen
evidence. It does not commit a model binary, prove broad 258V SLM answer
quality, touch BitNet QK256/I2_S proof, or prove Arc 140V / Intel NPU execution.

### Dense SLM CPU Answer Smoke

`SLM258V-002` runs the pinned Qwen2.5 dense SLM candidate through a strict
three-case CPU answer smoke on the 258V:

```text
ci/hardware/intel-258v/2026-05-08/slm-answer-corpus-qwen25-cpu.json
```

The corpus is defined at:

```text
ci/quality/slm258v-qwen25-answer-corpus.yaml
```

The receipt uses the real `qwen2.5-0.5b-instruct-q8_0.gguf` artifact, GGUF
tokenizer metadata, the `qwen2.5` prompt template, greedy deterministic
generation, `selected_backend=cpu`, and `fallback_used=false`. All three tiny
answer-readiness cases pass. This first receipt is diagnostic evidence because
it records `i2_s-avx2-reference` / `gguf_packed_i2_s` provenance fields.

`SLM258V-003` reruns the same smoke after separating dense SLM provenance from
BitNet packed-kernel receipt fields:

```text
ci/hardware/intel-258v/2026-05-08/slm-answer-corpus-qwen25-cpu-clean-provenance.json
```

The clean-provenance receipt preserves the same model SHA256, tokenizer
metadata, prompt template, generated IDs/text, `selected_backend=cpu`, and
`fallback_used=false`, while its child receipts record
`dense-qwen-cpu-reference`, `gguf_dense_q8_0`, and `dense_slm` provenance
instead of BitNet I2_S/QK256 provenance. It remains a bounded dense SLM CPU
answer-smoke receipt only; it does not prove broad SLM chat quality, speed,
Arc/NPU execution, or BitNet QK256/I2_S proof.

### Dense SLM CPU Phase Receipts

`SLM258V-004` records dense Qwen SLM CPU phase timing receipts on the 258V:

```text
ci/hardware/intel-258v/2026-05-08/slm-phase-warm-session-qwen25-cpu.json
ci/hardware/intel-258v/2026-05-08/slm-phase-warm-session-qwen25-cpu-profiles/prefill_512.json
ci/hardware/intel-258v/2026-05-08/slm-phase-warm-session-qwen25-cpu-profiles/decode_128.json
```

The release-built run uses the pinned Qwen2.5 Q8_0 GGUF artifact, strict loader
and tokenizer resolution, the `qwen2.5` prompt template, `selected_backend=cpu-rust`,
and `fallback_used=false`. The aggregate and per-profile receipts record
`dense-qwen-cpu-reference`, `gguf_dense_q8_0`, and `dense_slm` provenance while
omitting top-level BitNet I2_S/QK256 provenance from the dense profile receipts.
This is dense SLM CPU phase timing evidence only; it does not prove broad SLM
chat quality, speedup, sustained performance, BitNet QK256/I2_S behavior, Arc
140V execution, or Intel NPU execution.

The 2026-05-08 CLI platform probe refresh is:

```text
ci/hardware/intel-258v/2026-05-08/platform-probe-cli.json
```

It records OpenVINO 2026.1 visibility for `CPU`, `GPU`, and `NPU`, identifies
the Arc 140V OpenVINO GPU device as `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)`,
and records Level Zero loader visibility for Arc 140V through `ze_loader.dll`
device ID `0x64A0`. Native OpenCL execution remains proven by the separate
OpenCL smoke/parity receipts.

### CLI Platform Probe

Use the CLI probe command to emit the visibility-only platform receipt from the
current machine without launching kernels or compiling OpenVINO graphs:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- lunar-lake-probe \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/platform-probe-cli.json
```

The command records `proof_stage=runtime_detected`, `runtime_api=platform_probe`,
`fallback_used=false`, and a `must_not_claim` list. It does not replace the
lane-specific Arc 140V, NPU, CPU BitNet, parity, or benchmark artifacts.

### Arc 140V OpenVINO GPU Smoke

Use the Arc 140V OpenVINO GPU smoke command to emit a tiny fixed-shape graph
receipt from the OpenVINO GPU device without loading BitNet models or running
native OpenCL:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- intel-arc-140v-openvino-gpu-smoke \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json
```

The command records `proof_stage=kernel_smoke_tested` only when OpenVINO reports
an Arc 140V GPU device, compiles the tiny static graph to that device, and
matches the CPU expected output. On the 2026-05-08 Windows 258V artifact,
OpenVINO 2026.1 reports the selected runtime device as `GPU` with full device
name `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)`.
It keeps `fallback_used=false`,
`cpu_fallback_allowed=false`, `bitnet_inference=false`, and `qk256_decode=false`.
It does not prove native OpenCL kernels, BitNet inference, or Arc acceleration.

### CPU BitNet Validation Preflight

Use the CPU validation command to emit the Lunar Lake CPU lead artifact without
touching unrelated accelerator surfaces:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- validate cpu-bitnet \
  --machine intel-258v \
  --model /models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer /models/BitNet-b1.58-2B-4T/tokenizer.json \
  --backend cpu \
  --strict \
  --max-tokens 1 \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-bitnet-validation.json
```

This command is validation-only. If the canonical GGUF or tokenizer is absent,
it writes `proof_stage=blocked_preflight` with a structured blocker. It does not
load BitNet tensors, run QK256/TL2 kernels, decode tokens, or make benchmark
claims.

### CPU Phase Benchmark Receipt

Use the CPU phase benchmark receipt emitter to turn strict CPU proof receipts
into phase-aware 258V CPU artifacts:

```bash
cargo run --locked -p bitnet-bench-receipts \
  --bin cpu_phase_benchmark_receipt \
  --no-default-features \
  -- \
  --strict-proof-receipt ci/hardware/intel-258v/YYYY-MM-DD/strict-bitnet-cpu-proof.json \
  --machine-id intel-258v \
  --hardware-lane intel-258v-cpu-avx2 \
  --selected-backend cpu-rust \
  --model-quant-format QK256/I2_S \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --receipt-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-phase-benchmark.json
```

The first 258V phase receipt is:

```text
ci/hardware/intel-258v/2026-05-07/cpu-phase-benchmark.json
```

It records the available first-token strict CPU timing, selected backend/kernel,
fallback status, CPU feature set, 4 P-core / 4 low-power E-core topology,
shared LPDDR memory context, and Balanced power mode. Profiles that are not
backed by a supplied strict CPU proof remain explicit `not_run` gaps. This is a
phase receipt, not a sustained throughput claim and not an Arc 140V or Intel
NPU performance comparison. The CPU258V-003 profile summary records `smoke_1`
and `first_token` from the one-token proof and keeps `decode_128` and
`prefill_512` as explicit `not_run` gaps until matching strict proofs exist.

Follow-up CPU258V-005 evidence attempts are recorded at:

```text
ci/hardware/intel-258v/2026-05-08/cpu-phase-evidence-attempts.json
```

That artifact records timed-out strict CPU attempts for calibrated
`prefill_512` collection and preserves `decode_128` as `not_run`. It is
blocker evidence only; it does not prove prefill, decode, throughput, Arc 140V,
or Intel NPU performance.

CPU258V-006 adds a warm CPU phase runner so the model and tokenizer can be
loaded once before collecting long `prefill_512` and `decode_128` profile
receipts:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- \
  --device cpu \
  cpu-phase-warm-session \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/BitNet-b1.58-2B-4T/tokenizer.json \
  --strict-loader \
  --strict-tokenizer \
  --threads 8 \
  --prefill-prompt-file ci/hardware/intel-258v/YYYY-MM-DD/prefill-512-prompt.txt \
  --decode-tokens 128 \
  --cpu-kernel avx2 \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-phase-warm-session.json
```

The command emits per-profile strict CPU receipts under
`cpu-phase-warm-session-profiles/`. Those receipts are inputs to
`cpu_phase_benchmark_receipt`; they are phase timing evidence only and do not
claim answer quality, sustained throughput, Arc 140V execution, Intel NPU
execution, or acceleration.

CPU258V-013 records the first release-built warm-session receipts emitted after
the BitNet b1.58 RMSNorm/ReLU2 mechanics correction:

```text
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session.json
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session-profiles/prefill_512.json
ci/hardware/intel-258v/2026-05-08/cpu-phase-warm-session-profiles/decode_128.json
```

The run used the real GGUF model, explicit LLaMA 3 tokenizer, strict loader and
tokenizer modes, selected `i2_s-avx2-reference`, and `fallback_used=false`.
`prefill_512` records 513 prompt tokens, 512 prefill tokens, and one generated
token. `decode_128` records 128 generated tokens after a short prompt. These
artifacts fill the prior `prefill_512` and `decode_128` evidence gaps, but they
are still CPU phase timing evidence only: they do not claim answer quality,
sustained throughput, speedup, Arc 140V execution, Intel NPU execution, or
acceleration.

### CPU Answer Template Refresh

CPU258V-007 records the first 258V AVX2 answer-corpus refresh after the CPU
answer lane adopted the BitNet.cpp answer-ready prompt envelope:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template.json
```

The artifact records five timeout rows with `missing_child_receipt` kernels.
It is blocker evidence only: it shows that the newer answer-ready prompt path
did not complete within the bounded local child-run window. It does not prove
answer quality, scalar/AVX2 parity under the new prompt, sustained throughput,
Arc 140V execution, or Intel NPU execution.

CPU258V-008 adds a bounded answer-corpus case filter so the next 258V refresh
can isolate a single prompt before spending a full-corpus local decode window:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- \
  --device cpu \
  answer-corpus \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/BitNet-b1.58-2B-4T/tokenizer.json \
  --cpu-kernel avx2 \
  --case-id arithmetic-single-digit \
  --dump-logit-steps 1 \
  --logits-topk 5 \
  --per-prompt-timeout-seconds 420 \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-answer-corpus-avx2-bitnetcpp-template-case.json
```

The aggregate receipt preserves the full corpus `case_count` and records
`selected_case_count` plus `selected_case_ids`. This is diagnostic scope only:
it narrows timeout evidence collection and does not prove selected-case
completion, answer quality, scalar/AVX2 parity, sustained throughput, Arc 140V
execution, or Intel NPU execution.

CPU258V-009 records the first bounded single-case attempt:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2.json
```

The selected `math_2_plus_2` case timed out within the same 420-second child-run
window and emitted no child receipt. This confirms that at least one individual
BitNet.cpp-template case is blocked independently of full-corpus fanout. It does
not prove answer quality, scalar/AVX2 parity under the new prompt, sustained
throughput, Arc 140V execution, or Intel NPU execution.

CPU258V-010 records the same selected case through a release-built CLI:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2-release.json
```

The release run completes the selected strict CPU AVX2 row with real GGUF
loading, explicit strict tokenizer resolution, `i2_s-avx2-reference`,
`fallback_used=false`, 19 prompt tokens, and 4 generated tokens. It still fails
the exact-answer gate with generated text `'E tradi Paperback mente`, so this is
output-quality evidence, not answer readiness. It does not prove scalar/AVX2
parity, sustained throughput, Arc 140V execution, or Intel NPU execution.

CPU258V-011 records the matching release-built scalar run and scalar-vs-AVX2
parity receipt:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-bitnetcpp-template-math_2_plus_2-release.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-parity-bitnetcpp-template-math_2_plus_2-release.json
```

The selected scalar run uses `i2_s-scalar-reference`; the selected AVX2 run uses
`i2_s-avx2-reference`. Both produce the same prompt token IDs, generated token
IDs `[89048, 124979, 70232, 88162]`, and decoded text
`'E tradi Paperback mente`, with the same `gate_exact_trimmed` quality failure.
This proves selected-case scalar-vs-AVX2 answer parity only. It does not prove
answer correctness, full-corpus parity, sustained throughput, Arc 140V
execution, or Intel NPU execution.

CPU258V-014 records the same selected case after the BitNet b1.58 mechanics
correction from CPU258V-012:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-bitnetcpp-template-math_2_plus_2-post-mechanics.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2-post-mechanics.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-parity-bitnetcpp-template-math_2_plus_2-post-mechanics.json
```

Both the scalar and AVX2 release-built receipts pass the exact answer gate for
the selected `math_2_plus_2` BitNet.cpp-template case, producing decoded text
with a leading space followed by `4` and identical generated token IDs
`[220, 19, 128009]`. The scalar run
selects `i2_s-scalar-reference`, the AVX2 run selects `i2_s-avx2-reference`,
and the parity receipt records `summary.failed=0` with no first divergence.
This is selected-case answer recovery and scalar-vs-AVX2 parity evidence only:
it does not prove general chat quality, full-corpus answer readiness, sustained
throughput, speedup, Arc 140V execution, Intel NPU execution, or acceleration.

CPU258V-015 expands the same post-mechanics run to the full committed
`strict-bitnet-answer-corpus-v1` prompt set:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-bitnetcpp-template-full-post-mechanics.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template-full-post-mechanics.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-parity-bitnetcpp-template-full-post-mechanics.json
```

Both release-built scalar and AVX2 runs pass all five fixed corpus gates:
`math_2_plus_2`, `capital_france`, `repeat_colors`, `say_ok`, and
`yes_no_water`. The parity receipt records `summary.passed=5`,
`summary.failed=0`, and no first divergence. This is 258V CPU answer readiness
for the committed deterministic corpus and scalar-vs-AVX2 full-corpus parity
only. It does not prove general chat quality, sustained throughput, speedup,
Arc 140V execution, Intel NPU execution, or acceleration.

## Current Same-Machine Comparison Evidence

The current Lunar Lake comparison index is:

```text
ci/hardware/intel-258v/2026-05-08/platform-comparison-index.json
```

It points at the corrected 258V BitNet CPU reference bundle, post-fix BitNet
answer/parity/phase receipts, the dense Qwen SLM manifest plus answer/phase
receipts, the Arc 140V native OpenCL CPU/iGPU parity receipt, the live OpenVINO
NPU runtime/smoke/RMSNorm/linear/FFN selected subgraph parity receipts, and the
NPU OpenVINO llama.cpp GGUF reference blocker receipt. This makes the available
BitNet CPU, dense SLM CPU, GPU, and NPU evidence discoverable from one artifact
while preserving separate proof boundaries. It is not a cross-device
performance comparison and does not imply that Arc 140V or Intel NPU run full
BitNet or dense SLM inference.

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, CsSystemType

Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

Write-Host "=== Memory ==="
Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity, Speed, Manufacturer, PartNumber

Write-Host "=== Intel GPU / NPU PnP ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "Arc|140V|NPU|Neural|AI Boost|VPU|Intel.*Graphics"
} | Format-List *

Write-Host "=== OpenCL ==="
where clinfo
clinfo | Select-String -Pattern "Platform Name|Device Name|Device Vendor|Driver Version|OpenCL C"

Write-Host "=== Level Zero / oneAPI ==="
where sycl-ls
sycl-ls
where ze_info
ze_info

Write-Host "=== OpenVINO ==="
python - <<'PY'
import json
import openvino as ov

core = ov.Core()
out = {
    "openvino_version": ov.__version__,
    "available_devices": list(core.available_devices),
    "devices": {}
}
for dev in core.available_devices:
    props = {}
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DEVICE_ALLOC_MEM_SIZE",
        "NPU_MAX_TILES",
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== CPU ==="
lscpu || true

echo "=== Memory ==="
free -h || true

echo "=== GPU / NPU PCI ==="
lspci -nn | grep -Ei 'vga|display|intel|arc|140v|64a0|npu|vpu|neural|accel' || true

echo "=== DRM render nodes ==="
ls -l /dev/dri/renderD* || true
stat -c "%G %n" /dev/dri/renderD* || true
groups "$USER"

echo "=== accel devices ==="
ls -l /dev/accel || true

echo "=== NPU driver logs ==="
dmesg | grep -Ei 'intel_vpu|ivpu|vpu|npu|accel' | tail -200 || true

echo "=== OpenCL ==="
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Version|Driver Version|OpenCL C|Max compute units|Global memory size' || true

echo "=== Level Zero / oneAPI ==="
which sycl-ls || true
sycl-ls || true
which ze_info || true
ze_info || true

echo "=== OpenVINO ==="
python3 - <<'PY'
import json
import openvino as ov

core = ov.Core()
out = {
    "openvino_version": ov.__version__,
    "available_devices": list(core.available_devices),
    "devices": {}
}
for dev in core.available_devices:
    props = {}
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DEVICE_ALLOC_MEM_SIZE",
        "NPU_MAX_TILES",
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

## First Platform Receipt

The first 258V platform receipt should establish visibility only:

```json
{
  "platform": "core-ultra-7-258v",
  "cpu_backend": "intel-258v-cpu-avx2",
  "gpu_backend": "intel-arc-140v-opencl",
  "npu_backend": "intel-npu-openvino",
  "openvino_available_devices": ["CPU", "GPU", "NPU"],
  "openvino_npu_full_name": "...",
  "npu_driver_version": "...",
  "npu_compiler_version": "...",
  "npu_total_mem_size": 0,
  "npu_alloc_mem_size": 0,
  "npu_max_tiles": 1,
  "opencl_arc_140v_visible": true,
  "level_zero_visible": true,
  "npu_visible": true,
  "fallback_used": false,
  "status": "runtime_detected"
}
```

This is not an inference claim. Smoke, parity, and benchmark receipts come later.

The code-facing visibility probe for this first receipt lives in
`bitnet-device-probe` as `probe_lnl258v_platform()`. It emits a JSON-ready
`Lnl258vPlatformProbe` with nested CPU, Arc 140V, NPU, OpenVINO, memory, and
power sections. Unsupported runtime tools must be represented as `false`,
empty lists, or `null` fields rather than panics or fallback claims.

## Ownership

Proof lanes:

- CPU AVX2 remains under CPU runtime proof.
- Arc 140V OpenCL and OpenVINO GPU are owned by the Intel Arc GPU workstream.
- Intel AI Boost NPU and OpenVINO NPU are owned by the Intel NPU workstream.

The platform profile ties the lanes together for comparison, but it does not merge their claims.

## Current NPU Live Evidence

`NPU-010` records live OpenVINO 2026.1 evidence from the 258V laptop:

```text
ci/hardware/intel-258v/2026-05-08/npu-openvino-runtime-probe.json
ci/hardware/intel-258v/2026-05-08/npu-openvino-tiny-graph-smoke.json
ci/hardware/intel-258v/2026-05-08/npu-bitnet-rmsnorm-subgraph-parity.json
ci/hardware/intel-258v/2026-05-08/npu-bitnet-linear-projection-subgraph-parity.json
```

The runtime probe records OpenVINO
`2026.1.0-21367-63e31528c62-releases/2026/1`, available devices `CPU`, `GPU`,
and `NPU`, selected backend `intel-npu-openvino`, runtime device `NPU`, full
device name `Intel(R) AI Boost`, driver version `1004512`, compiler version
`458781`, total device memory `17179869184`, and `fallback_used=false`.

The tiny graph smoke records static OpenVINO NPU execution for
`tiny_matmul_add_f16_1x16` with `graph_execution=true`,
`bitnet_inference=false`, `qk256_decode=false`, and `fallback_used=false`.

The selected BitNet-shaped subgraph receipts record OpenVINO NPU parity for
`bitnet_rmsnorm_f16_1x16` and
`bitnet_linear_projection_f16_1x16x16` against CPU NumPy references. They record
`proof_stage=parity_tested`, `runtime_api=openvino`, `runtime_device=NPU`,
`fallback_used=false`, and CPU-reference error within the declared tolerance.

Allowed claims:

```text
OpenVINO 2026.1 can see and select the Intel AI Boost NPU on this 258V laptop.
The recorded tiny static graph executed on NPU with fallback_used=false.
The recorded RMSNorm and linear-projection static BitNet-shaped subgraphs
matched CPU NumPy references within tolerance on NPU.
```

Not allowed:

```text
Full BitNet inference works on Intel NPU.
Native bitnet-rs NPU inference works.
Intel NPU accelerates BitNet.
Packed BitNet QK256 decode works on Intel NPU.
CPU fallback satisfies NPU proof.
```

## External NPU Reference Evidence

`NPU-009` tracks OpenVINO llama.cpp GGUF as an external Intel NPU reference
lane. The first 258V receipt is:

```text
ci/hardware/intel-258v/2026-05-08/npu-openvino-llamacpp-gguf-reference.json
```

That receipt is intentionally `proof_stage=blocked_reference`: the local
environment did not have the OpenVINO Python runtime available, so the preview
llama.cpp GGUF backend was not invoked and no NPU graph execution occurred.

Allowed claim:

```text
OpenVINO llama.cpp GGUF is tracked as an external reference lane, and the
current local reference attempt is blocked before execution.
```

Not allowed:

```text
Full BitNet inference works on Intel NPU.
Native bitnet-rs NPU inference works.
Intel NPU accelerates BitNet.
Packed BitNet QK256 GGUF decode works on Intel NPU.
CPU fallback satisfies NPU proof.
```

## Current CPU Reference Bundle

`CPU258V-026` refreshes the current 258V CPU reference plate as a single
same-machine evidence index:

```text
ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle.json
```

It supersedes the post-mechanics bundle:

```text
ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json
```

The refreshed bundle links the strict real-GGUF decode smoke, the
post-mechanics full fixed-corpus scalar and AVX2 answer-corpus receipts, the
scalar-vs-AVX2 answer-parity receipt, warm-session `prefill_512` /
`decode_128` phase receipts, and the CPU semantic-debug ladder through
transformer-layer parity. It records the Microsoft BitNet b1.58 I2_S GGUF SHA,
explicit LLaMA 3 tokenizer source, selected scalar and AVX2 CPU kernel IDs, and
`fallback_used=false`.

The CPU semantic-debug ladder links:

```text
ci/hardware/intel-258v/2026-05-08/prompt-authority-audit-math.json
ci/hardware/intel-258v/2026-05-08/hf-prompt-token-reference-parity.json
ci/hardware/intel-258v/2026-05-08/hf-prompt-token-reference-parity-after-prompt-fix.json
ci/hardware/intel-258v/2026-05-08/external-first-token-reference.json
ci/hardware/intel-258v/2026-05-08/first-token-divergence-classification.json
ci/hardware/intel-258v/2026-05-08/external-reference-instrumentation.json
ci/hardware/intel-258v/2026-05-08/cpu-qk256-i8s-semantic-audit.json
ci/hardware/intel-258v/2026-05-08/output-head-logits-index-audit.json
ci/hardware/intel-258v/2026-05-08/transformer-layer-parity.json
```

Allowed claims:

```text
The 258V CPU reference plate is bundled from strict real-GGUF receipts,
full fixed-corpus scalar-vs-AVX2 answer parity, warm-session phase receipts,
and CPU semantic-debug evidence through transformer-layer parity.
The bundle is a CPU reference input for later Arc 140V and Intel NPU parity
comparisons.
```

Not allowed:

```text
General chat quality is proven.
Sustained throughput is proven.
A CPU speedup is proven.
Arc 140V or Intel NPU execution is proven by this CPU bundle.
CPU fallback can satisfy Arc 140V or Intel NPU proof.
External first-token logits parity is proven.
Full model correctness is proven.
```

## Current CPU Semantic Diagnosis

`CPU258V-027` converts the current CPU reference bundle into a
machine-readable diagnosis artifact:

```text
ci/hardware/intel-258v/2026-05-08/cpu-semantic-diagnosis.json
```

The diagnosis keeps two boundaries separate. The first actionable semantic
blocker is prompt policy: the external HF `apply_chat_template` reference omits
BOS and preserves a trailing generation-prompt space after `Assistant:`, while
the current BitNet-rs metadata-authority path prepends BOS and renders
`Assistant:` without that trailing space. The separate evidence blocker is
external reference instrumentation: generated-token IDs and first-token
logits/top-k remain unavailable from the current external runner evidence, so
external logits parity is not proven.

The artifact also summarizes the evidence that is not currently first-failing:
QK256/I2_S/I8_S fixture semantics match the canonical oracle, output-head and
logits-index boundaries report 128,256-token consistency, scalar and AVX2
transformer-layer traces match across 13 recorded boundaries, and the fixed
five-case answer corpus passes scalar-vs-AVX2 parity. Those are narrow
diagnostic claims only; they do not fix prompt policy or prove broad answer
quality.

Recommended next work:

```text
CPU258V-028:
  align metadata-authoritative BitNet prompt rendering with the official
  template boundary, preserving generation-prompt spacing exactly and avoiding
  executor-added BOS/EOS after rendered chat templates.

CPU258V-029:
  rerun scalar and AVX2 answer-corpus receipts after the prompt-policy fix.

CPU258V-031:
  refresh the CPU reference bundle after the semantic fix and parity rerun.
```

Not allowed:

```text
The prompt-policy mismatch is fixed by CPU258V-027.
External generated-token-ID parity is proven.
External first-token logits parity is proven.
New or general BitNet answer quality is proven.
CPU speed, Arc 140V execution, or Intel NPU execution is proven.
```

## Post-Baseline Next Queue

After `CPU258V-016`, the next Lunar Lake work should remain CPU-referenced:

```text
NPU-011:
  selected static BitNet-shaped OpenVINO NPU FFN/ReLU2 subgraph parity
  artifact: ci/hardware/intel-258v/2026-05-08/npu-bitnet-ffn-subgraph-parity.json
  anchor: ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle.json

ARC140V-005:
  native OpenCL CPU/iGPU parity for one isolated kernel or subgraph
  artifact: ci/hardware/intel-258v/2026-05-08/arc-140v-opencl-parity.json
  anchor: ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle.json

ARC140V-003 live artifact:
  OpenVINO GPU tiny static graph smoke on Arc 140V
  artifact: ci/hardware/intel-258v/2026-05-08/arc-140v-openvino-gpu-smoke.json
  claim: OpenVINO GPU graph smoke only; no native OpenCL, BitNet, QK256, or acceleration claim

LNL258V-COMPARE-002:
  refreshed the same-machine comparison index after the new CPU reference
  bundle, Arc OpenVINO GPU smoke, and independent NPU/Arc parity receipts
```

These follow-ups preserve the current priority order:

```text
1. 258V CPU reference
2. Intel NPU selected static subgraph parity
3. Arc 140V native OpenCL parity
```

`NPU-011` extends the selected NPU subgraph ladder with
`bitnet_ffn_relu2_f16_1x16x32`. The live receipt records
`selected_backend=intel-npu-openvino`, `runtime_api=openvino`,
`runtime_device=NPU`, `proof_stage=parity_tested`, `graph_execution=true`, and
`fallback_used=false`. This is selected static subgraph parity only; it is not
full BitNet inference, NPU acceleration, packed QK256 decode, or CPU fallback
proof.

`ARC140V-005` records native OpenCL CPU/iGPU parity for the same-machine 258V
platform. The live receipt records `selected_backend=intel-arc-140v-opencl`,
`runtime_api=opencl`, `proof_stage=parity_tested`, `kernel_execution=true`,
`graph_execution=false`, and `fallback_used=false` for the isolated
`tiny_vector_add` kernel against the current 258V CPU reference bundle.
This is native OpenCL parity only; it is not BitNet inference, Arc acceleration,
packed QK256 decode, OpenVINO GPU proof, or CPU fallback proof.

The 2026-05-08 `ARC140V-003` live OpenVINO GPU smoke artifact records
`selected_backend=intel-arc-140v-openvino-gpu`, `runtime_api=openvino`,
`runtime_device=GPU`, `graph_execution=true`, and `fallback_used=false` for
the tiny `tiny_matmul_add_f16_1x16` graph. It is OpenVINO GPU smoke only and
does not prove native OpenCL, BitNet inference, packed QK256 decode, Arc
acceleration, or CPU fallback proof.

## CPU258V-018 External Prompt/Token Reference Parity

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/hf-prompt-token-reference-parity.json
```

The CPU258V-018 evidence compares official HF
`AutoTokenizer.apply_chat_template` output against BitNet-rs
metadata-authoritative `prompt-authority-audit` output for fixed prompts:
`math_2_plus_2`, `say_ok`, `capital_france`, and `yes_no_water`.

Current result:

```text
first_divergence_stage = prompt
cases_failed = 4
```

The comparison records that HF keeps a trailing generation-prompt space after
`Assistant:` and does not prepend BOS for these `apply_chat_template` prompt
IDs, while the current BitNet-rs metadata-authority path prepends BOS and
renders `Assistant:` without the trailing space.

Allowed claim:

```text
HF rendered prompts and prompt token IDs were compared against BitNet-rs
metadata-authoritative prompt-authority audit output for fixed prompts.
```

Not allowed:

```text
Answer quality is proven.
First-token logits or model inference parity is proven.
CPU speed is proven.
Arc 140V or Intel NPU execution is proven.
Packed QK256 decode semantics are fixed.
```

## CPU258V-028 Prompt Policy Fix

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/hf-prompt-token-reference-parity-after-prompt-fix.json
```

`CPU258V-028` removes the prompt-policy mismatch classified by `CPU258V-027`.
The BitNet answer-ready template now preserves the official HF
`apply_chat_template` generation prompt boundary, including the trailing
space after `Assistant:`, and the metadata-authoritative path no longer prepends an
executor BOS after rendering that chat template.

Current result:

```text
first_divergence_stage = none
cases_passed = 4
cases_failed = 0
```

Allowed claim:

```text
BitNet-rs metadata-authoritative BitNet prompt strings and prompt token IDs
match the external HF apply_chat_template boundary for the fixed prompt corpus.
```

Not allowed:

```text
Answer quality is proven.
External first-token logits parity is proven.
CPU speed is proven.
Arc 140V or Intel NPU execution is proven.
Packed QK256 decode semantics are changed by this prompt-policy fix.
Full model correctness is proven.
```

## CPU258V-029 Answer Corpus After Prompt Fix

Artifacts:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-after-prompt-fix.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-after-prompt-fix.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-parity-after-prompt-fix.json
```

CPU258V-029 reruns the full fixed `strict-bitnet-answer-corpus-v1` prompt set
after the CPU258V-028 prompt-policy fix. The release-built scalar and AVX2 runs
use the same real GGUF model, explicit tokenizer, BitNet.cpp answer-ready prompt
template, greedy settings, one-step top-k capture, and `fallback_used=false`.

Current result:

```text
scalar_quality_failed = 0
avx2_quality_failed = 0
scalar_cases_passed = 5
avx2_cases_passed = 5
answer_parity_failed = 0
first_divergence = null
prompt_template = bitnetcpp-answer
prompt_boundary = trailing Assistant: generation prompt preserved
prompt_add_bos = false
prompt_parse_special = true
```

The tiny corpus answers are:

```text
math_2_plus_2 = 4
capital_france = Paris
repeat_colors = red blue green
say_ok = OK
yes_no_water = No. Water is
```

Allowed claim:

```text
The corrected prompt-policy path was used for the 258V scalar and AVX2
answer-corpus rerun, all five tiny deterministic gates passed in both lanes,
and scalar-vs-AVX2 answer parity has no divergence for those receipts.
```

Not allowed:

```text
General BitNet chat quality is proven.
External first-token logits parity is proven.
CPU speed or sustained throughput is proven.
Arc 140V or Intel NPU execution is proven.
QK256 semantics or transformer math changed in this PR.
Full model correctness is proven.
```

## CPU258V-021 External Reference Instrumentation Boundary

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/external-reference-instrumentation.json
```

The CPU258V-021 instrumentation boundary classifies whether the external
BitNet.cpp reference artifact exposes direct generated-token IDs and first-token
logits/top-k evidence for the fixed 258V prompts. It reads the CPU258V-019
external reference capture and records missing reference fields as explicit
blockers rather than inferring token or logit parity from generated text.

Current result:

```text
cases_total = 4
cases_with_generated_text = 4
cases_with_generated_token_ids = 0
cases_with_first_token_topk_logits = 0
generated_token_ids_available = false
first_token_logits_available = false
classification = reference_runner_requires_instrumentation
```

The result means Microsoft BitNet.cpp generated-text evidence remains useful,
but the current external artifact still cannot prove generated-token-ID parity
or first-token logits parity. The next required evidence is a patched or scripted
reference runner path that exposes direct generated-token IDs and first-token
logits/top-k without text re-tokenization.

Allowed claim:

```text
The external reference evidence boundary is classified, and the missing
generated-token/logit fields are explicit blockers.
```

Not allowed:

```text
Generated-token-ID parity against the external reference is proven.
First-token logits parity is proven.
BitNet answer quality is newly proven.
CPU speed is proven.
Arc 140V or Intel NPU execution is proven.
Packed QK256 decode semantics are fixed.
```

## CPU258V-022 CPU QK256/I2_S/I8_S Semantic Audit

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/cpu-qk256-i8s-semantic-audit.json
```

The CPU258V-022 audit adds CPU scalar fixtures for the QK256/I2_S and BitNet
I8_S semantic boundary that the 258V CPU answer path depends on. It checks the
byte-exact grouped I2_S layout fixture, the canonical code map, the I8_S
activation absmax scale and activation sum, the scaled output formula, and
non-finite inline weight-scale rejection.

Current audited semantics:

```text
I2_S code map = {0: -1.0, 1: 0.0, 2: 1.0, 3: 0.0}
QK256 packed layout = two 128-value chunks, 32 grouped bytes per chunk
byte encoding = lane0<<6 | lane1<<4 | lane2<<2 | lane3
I8_S activation scale = 127.0 / max(1e-5, max(abs(x)))
scaled output = ((int_dot - activation_sum) / activation_scale) * inline_weight_scale
```

Focused validation:

```text
rustfmt --check --config skip_children=true --edition 2024 crates/bitnet-quantization/src/i2s_qk256.rs
cargo test --locked -p bitnet-quantization --no-default-features bitnet_i8s -- --nocapture
cargo test --locked -p bitnet-quantization --no-default-features qk256_bitnet_i2s_grouped_layout_byte_exact_fixture -- --nocapture
cargo test --locked -p bitnet-quantization --no-default-features qk256 -- --nocapture
```

Allowed claim:

```text
The 258V CPU scalar QK256/I2_S/I8_S semantic boundary has focused fixtures for
grouped layout, activation scaling, activation-sum subtraction, inline weight
scale handling, and non-finite scale rejection.
```

Not allowed:

```text
BitNet answer quality is proven.
First-token logits parity is proven.
CPU speed or sustained throughput is proven.
Arc 140V or Intel NPU execution is proven.
QK256 decode on an accelerator is proven.
Full model correctness is proven.
```

## CPU258V-023 Output-Head / Logits-Index Audit

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/output-head-logits-index-audit.json
```

CPU258V-023 records the 258V CPU output-head and logits-index boundary after
the prompt/token, external-reference, and QK256/I2_S/I8_S semantic audits.
CPU258V-024 extends that boundary by recording observed runtime logits vector
length evidence from release-built full-corpus scalar and AVX2 answer-corpus
receipts. It does not run a new broad answer-quality gate. It inspects the GGUF
tensor table, resolves the strict tokenizer, records tied-output-head policy,
records the expected and observed logits vector lengths, and decodes scalar/AVX2
first-step top-k token IDs.

Additional CPU258V-024 input artifacts:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-full-observed-logits.json
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-full-observed-logits.json
```

Current result:

```text
tied_output_policy = tied_token_embeddings
selected_embedding = token_embd.weight
selected_output_head = null
expected_logits_vector_length = 128256
observed_logits_vector_length = 128256
observed_logits_vector_length_source = run_receipt_logits_index_boundary
model_vocab_size_proxy = 128256
metadata_vocab_matches_tokenizer = true
scalar_avx2_first_step_topk_ids_all_match = true
classification = output_head_logits_index_boundary_recorded
first_mismatch_stage = null
notes = []
```

Focused validation:

```text
rustfmt --check --config skip_children=true --edition 2024 crates/bitnet-cli/src/commands/output_head_logits_audit.rs crates/bitnet-cli/src/commands/mod.rs crates/bitnet-cli/src/main.rs crates/bitnet-cli/tests/cli_arg_tests.rs
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli --bin bitnet output_head_logits_audit -- --nocapture
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli --test cli_arg_tests output_head_logits_audit_help_lists_boundary_inputs -- --exact --nocapture
target/release/bitnet.exe answer-corpus --model C:/Code/Models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer C:/Code/Models/BitNet-b1.58-2B-4T/tokenizer.json --device cpu --threads 8 --cpu-kernel scalar --per-prompt-timeout-seconds 300 --dump-logit-steps 1 --logits-topk 5 --json-out ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-full-observed-logits.json
target/release/bitnet.exe answer-corpus --model C:/Code/Models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer C:/Code/Models/BitNet-b1.58-2B-4T/tokenizer.json --device cpu --threads 8 --cpu-kernel avx2 --per-prompt-timeout-seconds 300 --dump-logit-steps 1 --logits-topk 5 --json-out ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-full-observed-logits.json
target/release/bitnet.exe output-head-logits-audit --model C:/Code/Models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer C:/Code/Models/BitNet-b1.58-2B-4T/tokenizer.json --prompt-audit ci/hardware/intel-258v/2026-05-08/prompt-authority-audit-math.json --scalar-answer-corpus ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-scalar-full-observed-logits.json --avx2-answer-corpus ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-full-observed-logits.json --json-out ci/hardware/intel-258v/2026-05-08/output-head-logits-index-audit.json
python -m json.tool ci/hardware/intel-258v/2026-05-08/output-head-logits-index-audit.json
git diff --check
```

Allowed claim:

```text
The 258V CPU output-head/tied-head boundary is recorded for the fixed
post-mechanics BitNet CPU receipts, scalar/AVX2 first step top-k token IDs
decode and match locally, and observed runtime logits vector length is recorded
as 128256 for the full fixed scalar and AVX2 answer-corpus receipts.
```

Not allowed:

```text
BitNet answer quality is newly proven.
First-token logits parity with the external reference is proven.
CPU speed or sustained throughput is proven.
Arc 140V or Intel NPU execution is proven.
Full model correctness is proven.
```

## CPU258V-019 External First-Token Reference Boundary

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/external-first-token-reference.json
```

The CPU258V-019 artifact records the external reference boundary available from
`MODEL-ARTIFACT-007`: Microsoft BitNet.cpp generated text for the fixed prompts
and the exact reference command shape using:

```text
--override-kv tokenizer.ggml.pre=str:llama-bpe
-p "User: <question><|eot_id|>Assistant:"
```

The artifact also records prompt token IDs for that exact prompt string using
HF `AutoTokenizer` with `add_special_tokens=false`. It explicitly marks
generated token IDs and first-token logits as unavailable in the current
reference evidence.

Current result:

```text
reference_generated_text_available = true
reference_generated_token_ids_available = false
reference_logits_available = false
first_token_boundary_classification = generated_text_available_token_id_unavailable
```

Allowed claim:

```text
External BitNet.cpp generated-text boundaries and exact prompt-token policy are
recorded for the fixed CPU258V prompts.
```

Not allowed:

```text
First-token logits parity is proven.
Generated-token-ID parity is proven.
Answer quality beyond the cited MODEL-ARTIFACT-007 reference-runner gate is
proven.
CPU speed is proven.
Arc 140V or Intel NPU execution is proven.
Packed QK256 decode semantics are fixed.
```

## CPU258V-020 First-Token Divergence Classification

Artifact:

```text
ci/hardware/intel-258v/2026-05-08/first-token-divergence-classification.json
```

The CPU258V-020 classifier combines the external BitNet.cpp generated-text
boundary, the prompt-authority audit, scalar and AVX2 answer-corpus receipts,
and the scalar-vs-AVX2 answer-parity receipt. It keeps local CPU parity
separate from the external reference evidence boundary.

Current result:

```text
cases_total = 4
cases_inconclusive = 4
prompt_token_exact_matches = 0
prompt_token_local_bos_prefix_matches = 4
generated_text_trimmed_scalar_matches = 4
generated_text_trimmed_avx2_matches = 4
generated_text_trimmed_scalar_avx2_matches = 4
scalar_avx2_parity_passed = true
reference_generated_token_ids_available = false
reference_logits_available = false
classification = reference_generated_token_ids_and_logits_unavailable
```

The result means the fixed external prompts match the local CPU prompt IDs
after accounting for the local BOS prefix, and the external generated text
matches the local scalar/AVX2 text after trimming. The classifier intentionally
does not claim generated-token-ID parity or first-token logits parity because
the external reference artifact does not expose generated token IDs or logits.

Allowed claim:

```text
The first available CPU258V external-reference boundary is classified, and the
next required evidence is reference generated token IDs plus first-token
logits/top-k.
```

Not allowed:

```text
First-token logits parity is proven.
Generated-token-ID parity against the external reference is proven.
General answer quality is proven.
CPU speed is proven.
Arc 140V or Intel NPU execution is proven.
Packed QK256 decode semantics are fixed.
```

## CPU-BITNET-REF-003 Direct BitNet.cpp Token Boundary

Artifacts:

```text
ci/hardware/intel-258v/2026-05-08/external-first-token-reference-direct.json
ci/hardware/intel-258v/2026-05-08/first-token-divergence-classification-direct.json
```

CPU-BITNET-REF-003 records the direct Microsoft BitNet.cpp token boundary for
the fixed 258V prompts. A local BitNet.cpp `llama-server` build is patched with:

```text
ci/bitnet_cpp_server_token_logits.patch
```

The patch exposes `tok_id` and raw candidate `logit` fields in the server
`completion_probabilities` response. The helper script:

```text
scripts/bitnet_cpp_reference_boundary.py
```

starts that patched local server, uses the post-prompt-fix HF prompt/token
reference as the prompt authority, and records BitNet.cpp prompt token IDs,
generated token IDs, first generated token IDs, decoded first tokens, and
first-token top-k token IDs/probabilities/raw logits.

Current direct reference result:

```text
cases_total = 4
cases_with_reference_generated_token_ids = 4
cases_with_reference_first_token_topk_logits = 4
reference_generated_token_ids_available = true
reference_logits_available = true
boundary_classification = direct_reference_generated_token_ids_and_first_token_topk_logits_recorded
```

The first-token divergence classifier is rerun against that direct reference
artifact and the corrected scalar/AVX2 CPU receipts.

Current classifier result:

```text
validation.passed = true
cases_total = 4
prompt_token_exact_matches = 4
cases_with_reference_generated_token_ids = 4
cases_with_reference_first_token_topk_logits = 4
reference_generated_token_ids_available = true
reference_logits_available = true
scalar_avx2_parity_passed = true
classification = no_divergence_at_first_generated_token
first_divergence_stage = none
```

Allowed claim:

```text
Direct BitNet.cpp generated-token IDs and first-token top-k/logit evidence are
recorded for the fixed 258V prompts, and the first generated token matches the
corrected 258V scalar and AVX2 CPU receipts for all four cases.
```

Not allowed:

```text
Broad BitNet answer quality is proven.
Full generated-token sequence parity is proven beyond the recorded boundary.
BitNet-rs first-token logits parity against BitNet.cpp is proven.
CPU speed or sustained throughput is proven.
Arc 140V or Intel NPU execution is proven.
QK256/I2_S behavior changed.
Full model correctness is proven.
```
