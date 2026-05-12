# Intel 258V Hardware Artifacts

This directory holds machine-local artifacts for the Core Ultra 7 258V Lunar
Lake laptop.

Expected path pattern:

```text
ci/hardware/intel-258v/<date>/platform-probe.json
ci/hardware/intel-258v/<date>/platform-probe-cli.json
ci/hardware/intel-258v/<date>/cpu-bitnet-validation.json
ci/hardware/intel-258v/<date>/cpu-phase-benchmark.json
ci/hardware/intel-258v/<date>/cpu-phase-evidence-attempts.json
ci/hardware/intel-258v/<date>/cpu-phase-warm-session.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-avx2-bitnetcpp-template.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-avx2-bitnetcpp-template-case.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2-release.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-scalar-bitnetcpp-template-math_2_plus_2-release.json
ci/hardware/intel-258v/<date>/cpu-answer-parity-bitnetcpp-template-math_2_plus_2-release.json
ci/hardware/intel-258v/<date>/prompt-authority-audit-math.json
ci/hardware/intel-258v/<date>/hf-prompt-token-reference-parity.json
ci/hardware/intel-258v/<date>/hf-prompt-token-reference-parity-after-prompt-fix.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-scalar-after-prompt-fix.json
ci/hardware/intel-258v/<date>/cpu-answer-corpus-avx2-after-prompt-fix.json
ci/hardware/intel-258v/<date>/cpu-answer-parity-after-prompt-fix.json
ci/hardware/intel-258v/<date>/external-first-token-reference.json
ci/hardware/intel-258v/<date>/first-token-divergence-classification.json
ci/hardware/intel-258v/<date>/cpu-qk256-i8s-semantic-audit.json
ci/hardware/intel-258v/<date>/output-head-logits-index-audit.json
ci/hardware/intel-258v/<date>/transformer-layer-parity.json
ci/hardware/intel-258v/<date>/cpu-reference-bundle.json
ci/hardware/intel-258v/<date>/cpu-semantic-diagnosis.json
ci/hardware/intel-258v/<date>/npu-openvino-runtime-probe.json
ci/hardware/intel-258v/<date>/npu-openvino-tiny-graph-smoke.json
ci/hardware/intel-258v/<date>/npu-bitnet-rmsnorm-subgraph-parity.json
ci/hardware/intel-258v/<date>/npu-bitnet-linear-projection-subgraph-parity.json
ci/hardware/intel-258v/<date>/npu-bitnet-ffn-subgraph-parity.json
ci/hardware/intel-258v/<date>/arc-140v-openvino-gpu-smoke.json
ci/hardware/intel-258v/<date>/arc-140v-opencl-parity.json
ci/hardware/intel-258v/<date>/platform-comparison-index.json
```

`platform-probe.json` is visibility-only. It may record CPU AVX2 facts, Arc
140V OpenCL/Level Zero/OpenVINO GPU visibility, Intel NPU OS/OpenVINO
visibility, memory, power, and OS context. It must not claim BitNet inference,
Arc 140V execution, NPU execution, or acceleration.

`platform-probe-cli.json` is the CLI-emitted form of the same visibility-only
platform probe. The 2026-05-08 refresh records OpenVINO 2026.1 visibility for
CPU, GPU, and NPU on the 258V, identifies the Arc 140V OpenVINO GPU device, and
records Arc 140V Level Zero loader visibility through `ze_loader.dll` device ID
`0x64A0`.

`cpu-phase-benchmark.json` converts strict CPU proof receipts into phase-aware
CPU evidence. It may measure only the phases present in the supplied strict
proof and must keep unavailable profiles as explicit `not_run` gaps.

`cpu-phase-evidence-attempts.json` records strict CPU phase collection attempts
that did not emit proof receipts. It is blocker evidence only and must not be
treated as decode, prefill, throughput, Arc 140V, or Intel NPU proof.

`cpu-phase-warm-session.json` is the aggregate receipt for a one-process CPU
phase collection run. It loads the real GGUF model and tokenizer once, writes
per-profile strict CPU receipts under `cpu-phase-warm-session-profiles/`, and
remains CPU-only phase timing evidence until those profile receipts are
converted by `cpu_phase_benchmark_receipt`.

`cpu-phase-warm-session-after-prompt-fix.json` refreshes that one-process phase
surface after the metadata-authoritative prompt-policy fix and fixed-corpus
answer pass. It writes per-profile receipts under
`cpu-phase-warm-session-after-prompt-fix-profiles/`, records
`i2_s-avx2-reference` with `fallback_used=false`, and remains phase timing
evidence only; it does not claim speedup, sustained throughput, Arc/NPU
execution, QK256 changes, or full model correctness.

`cpu-answer-corpus-avx2-bitnetcpp-template.json` records the first 258V AVX2
attempt to refresh answer-corpus evidence with the BitNet.cpp answer-ready
prompt envelope. Timeout rows and `missing_child_receipt` kernels are blocker
evidence only; they do not prove answer quality or AVX2 correctness.

`cpu-answer-corpus-avx2-bitnetcpp-template-case.json` is the expected shape for
bounded single-case follow-up attempts using `answer-corpus --case-id`. It must
preserve the full corpus identity while recording `selected_case_count` and
`selected_case_ids`, and it remains diagnostic blocker evidence until a real
child receipt is emitted.

`cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2.json` records the
first 258V AVX2 single-case follow-up. The selected `math_2_plus_2` case still
timed out within the bounded child-run window, so it is blocker evidence only.

`cpu-answer-corpus-avx2-bitnetcpp-template-math_2_plus_2-release.json` records
the same selected case through a release-built CLI. The strict CPU run completes
with real GGUF loading, explicit tokenizer resolution, `i2_s-avx2-reference`,
and `fallback_used=false`, but the generated answer fails the exact-answer gate.

`cpu-answer-corpus-scalar-bitnetcpp-template-math_2_plus_2-release.json` and
`cpu-answer-parity-bitnetcpp-template-math_2_plus_2-release.json` record the
matching scalar run and scalar-vs-AVX2 parity result. Scalar and AVX2 generate
the same token IDs and decoded text for the selected case, so the selected bad
answer is not AVX2-specific.

`cpu-answer-corpus-scalar-after-prompt-fix.json`,
`cpu-answer-corpus-avx2-after-prompt-fix.json`, and
`cpu-answer-parity-after-prompt-fix.json` rerun the full fixed
`strict-bitnet-answer-corpus-v1` prompt set after the CPU258V-028 prompt-policy
fix. They record the corrected BitNet answer-ready prompt boundary with the
trailing `Assistant: ` generation prompt, `add_bos=false`, explicit tokenizer
authority, scalar/AVX2 selected kernels, and `fallback_used=false`. The five
tiny deterministic gates pass in both scalar and AVX2 and scalar-vs-AVX2 parity
has no divergence. These receipts still do not claim broad chat quality, CPU
speed, Arc 140V execution, Intel NPU execution, QK256 changes, or full model
correctness.

`platform-comparison-index.json` links independently scoped CPU, Arc 140V, and
Intel NPU artifacts from the same Lunar Lake laptop. It is an index only: it may
record artifact paths, backend identity, runtime API, proof stage, and fallback
status, but it must not merge CPU, GPU, or NPU claims or introduce performance,
BitNet inference, QK256 decode, or acceleration claims.

`cpu-reference-bundle.json` is the current 258V CPU evidence index for
accelerator comparison. It supersedes the post-mechanics bundle by linking the
prompt/token authority audit, external prompt/token parity, external
first-token boundary, first-token divergence classifier, QK256/I8_S semantic
audit, output-head/logits-index audit, observed logits evidence, and
transformer-layer parity ladder. It is a CPU reference index only and must not
claim new answer quality, CPU speed, Arc 140V execution, Intel NPU execution,
external first-token logits parity, or full model correctness.

`cpu-semantic-diagnosis.json` turns the CPU reference bundle into a
machine-readable diagnosis. It records the current prompt-policy mismatch
against the external HF `apply_chat_template` boundary, preserves the separate
external-reference instrumentation gap for generated-token IDs and first-token
logits, and summarizes QK256/I8_S, output-head/logits-index, transformer-layer,
answer-parity, and phase evidence. It is diagnostic only: it does not fix prompt
policy, prove external logits parity, add answer quality or speed claims, or
prove Arc 140V / Intel NPU execution.

`arc-140v-opencl-parity.json` records one isolated native OpenCL vector-add
parity run on Arc 140V against the selected 258V CPU reference bundle. It may
claim only native OpenCL CPU/iGPU parity for that kernel. It must not claim
BitNet inference, Arc acceleration, packed QK256 decode, OpenVINO GPU proof as
native OpenCL proof, or CPU fallback as Arc proof.

`arc-140v-openvino-gpu-smoke.json` records one tiny static OpenVINO GPU graph
smoke on Arc 140V. It may claim only OpenVINO GPU graph execution with
`fallback_used=false` and CPU expected-output agreement. It must not claim
native OpenCL execution, BitNet inference, Arc acceleration, packed QK256
decode, or CPU fallback proof.

`npu-openvino-runtime-probe.json` records live OpenVINO NPU visibility on the
258V. It may claim that OpenVINO selected `intel-npu-openvino` with runtime
device `NPU`; it must not claim graph execution or inference from visibility
alone.

`npu-openvino-tiny-graph-smoke.json` records the live static OpenVINO NPU tiny
graph smoke. It may claim only that the recorded graph executed on NPU with
`fallback_used=false` and matched the CPU expected output.

`npu-bitnet-rmsnorm-subgraph-parity.json` and
`npu-bitnet-linear-projection-subgraph-parity.json` record selected static
BitNet-shaped OpenVINO NPU subgraph parity against CPU NumPy references.
`npu-bitnet-ffn-subgraph-parity.json` adds the selected static FFN/ReLU2
subgraph anchored to the 258V CPU reference bundle. They may claim selected
subgraph parity only. They must not claim full BitNet inference, native
bitnet-rs NPU inference, NPU acceleration, packed QK256 decode, or CPU fallback
proof.
