# Intel 258V Hardware Artifacts

This directory holds machine-local artifacts for the Core Ultra 7 258V Lunar
Lake laptop.

Expected path pattern:

```text
ci/hardware/intel-258v/<date>/platform-probe.json
ci/hardware/intel-258v/<date>/cpu-bitnet-validation.json
ci/hardware/intel-258v/<date>/cpu-phase-benchmark.json
ci/hardware/intel-258v/<date>/cpu-phase-evidence-attempts.json
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
