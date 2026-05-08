# Intel 258V Hardware Artifacts

This directory holds machine-local artifacts for the Core Ultra 7 258V Lunar
Lake laptop.

Expected path pattern:

```text
ci/hardware/intel-258v/<date>/platform-probe.json
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
```

`platform-probe.json` is visibility-only. It may record CPU AVX2 facts, Arc
140V OpenCL/Level Zero/OpenVINO GPU visibility, Intel NPU OS/OpenVINO
visibility, memory, power, and OS context. It must not claim BitNet inference,
Arc 140V execution, NPU execution, or acceleration.

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
