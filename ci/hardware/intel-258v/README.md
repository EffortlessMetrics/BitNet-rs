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
keeps Level Zero absence explicit.

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

`platform-comparison-index.json` links independently scoped CPU, Arc 140V, and
Intel NPU artifacts from the same Lunar Lake laptop. It is an index only: it may
record artifact paths, backend identity, runtime API, proof stage, and fallback
status, but it must not merge CPU, GPU, or NPU claims or introduce performance,
BitNet inference, QK256 decode, or acceleration claims.

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
