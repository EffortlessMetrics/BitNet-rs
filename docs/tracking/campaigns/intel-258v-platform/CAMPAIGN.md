# Intel 258V Platform Campaign

Campaign ID: `intel-258v-platform`

Status: active

## Objective

Validate Core Ultra 7 258V as the BitNet CPU lead and tri-device platform while keeping CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU proof labels separate.

## End State

- Same-machine CPU, GPU, and NPU facts are captured.
- 258V CPU strict real-GGUF validation, scalar/AVX2 answer parity, and phase receipts provide the CPU reference plate.
- Arc 140V OpenCL, OpenVINO GPU, and OpenVINO NPU evidence are not conflated.
- Receipts record OS, drivers, memory, power, thermal, and WSL/native visibility context.

## Hard Constraints

- 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims.
- Arc 140V OpenCL proof is not NPU proof.
- OpenVINO GPU smoke is not packed BitNet kernel proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| LNL258V-RUN-001 | merged | Add JSON-ready Lunar Lake platform probe structs. |
| ARC140V-002 | merged | Add exact Arc 140V runtime identity probe logic. |
| ARC140V-003 | merged | Add Arc 140V OpenVINO GPU.0 tiny graph smoke; merged in #3942. |
| ARC140V-004 | merged | Add Arc 140V native OpenCL tiny kernel smoke; merged in #3953. |
| LNL258V-002 | merged | Add 258V probe bundle and same-machine comparison hooks. |
| LNL258V-003 | merged | Add CLI platform probe emission for the current 258V machine. |
| CPU258V-001 | merged | Add a validation-only CPU BitNet preflight harness for the 258V lane. |
| LNL258V-OWNERSHIP-001 | merged | Made the 258V CPU the BitNet CPU lead and set priority order: CPU, NPU, Arc 140V; merged in #3914. |
| CPU258V-002 | merged | Add scalar-vs-AVX2 strict CPU answer parity on the 258V; merged in #3929. |
| CPU258V-003 | merged | Add 258V CPU phase benchmark receipts for the CPU reference plate; merged in #3938. |
| CPU258V-004 | merged | Require real token-count thresholds before promoting 258V `decode_128` or `prefill_512` phase evidence; merged in #3981. |
| CPU258V-005 | merged | Record local strict CPU phase evidence attempts and keep `prefill_512`/`decode_128` blocked until a receipt-emitting phase runner exists; merged in #3999. |
| CPU258V-006 | merged | Add a strict CPU warm phase runner that emits receipt-converter inputs for `prefill_512` and `decode_128` without speedup, Arc, or NPU claims; merged in #4001. |
| CPU258V-007 | merged | Record the 258V AVX2 answer-corpus refresh under the BitNet.cpp answer-ready prompt envelope as timeout/blocker evidence; merged in #4006. |
| CPU258V-008 | merged | Add bounded `answer-corpus --case-id` diagnostics so the 258V answer-template refresh can run one corpus case at a time without answer-quality, parity, speed, Arc, or NPU claims; merged in #4008. |
| CPU258V-009 | merged | Record a bounded single-case 258V AVX2 answer-corpus attempt for `math_2_plus_2`, preserving timeout/blocker evidence without answer-quality, parity, speed, Arc, or NPU claims; merged in #4010. |
| CPU258V-010 | merged | Record a release-built single-case 258V AVX2 answer-corpus attempt that completes strict CPU execution but fails the answer-quality gate; no parity, speed, Arc, or NPU claims; merged in #4012. |
| CPU258V-011 | ready | Record release-built scalar and scalar-vs-AVX2 parity artifacts for the selected `math_2_plus_2` case, showing the bad answer is shared by scalar and AVX2; no answer-quality, speed, Arc, or NPU claims. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
