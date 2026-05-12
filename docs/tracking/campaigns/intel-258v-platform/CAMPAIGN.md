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
| CPU258V-011 | merged | Record release-built scalar and scalar-vs-AVX2 parity artifacts for the selected `math_2_plus_2` case, showing the bad answer is shared by scalar and AVX2; no answer-quality, speed, Arc, or NPU claims; merged in #4014. |
| CPU258V-012 | merged | Correct BitNet b1.58 CPU model mechanics to use RMSNorm and ReLU2, fix tied-output-head receipt metadata, and record a one-token strict scalar fixture showing the shared answer-quality issue remains after the mechanics correction; merged in #4022. |
| CPU258V-013 | merged | Record release-built warm-session `prefill_512` and `decode_128` strict CPU phase receipts on the 258V after the BitNet b1.58 mechanics correction; phase timing only, no answer-quality, speedup, Arc, or NPU claims; merged in #4036. |
| CPU258V-014 | merged | Record post-mechanics scalar and AVX2 answer-corpus receipts for the selected `math_2_plus_2` BitNet.cpp-template case, showing the corrected CPU path passes the exact answer gate and preserves scalar-vs-AVX2 parity; no general chat, speed, Arc, or NPU claims; merged in #4041. |
| CPU258V-015 | merged | Record post-mechanics scalar and AVX2 answer-corpus receipts for the full committed BitNet.cpp-template corpus on the 258V, showing all five fixed cases pass and scalar-vs-AVX2 full-corpus parity holds; no general chat, speed, Arc, or NPU claims; merged in #4046. |
| LNL258V-COMPARE-001 | merged | Add a same-machine comparison index that links CPU, Arc 140V, and Intel NPU artifacts by path, backend identity, proof stage, and fallback status without merging lane claims; merged in #4076. |
| CPU258V-016 | merged | Record the post-mechanics 258V CPU reference bundle used by accelerator parity receipts; merged in #4087. |
| ARC140V-005 | merged | Add native OpenCL CPU/iGPU parity for one isolated Arc 140V kernel against the 258V CPU reference bundle; merged in #4103. |
| LNL258V-COMPARE-002 | merged | Refresh the same-machine evidence index after the post-mechanics CPU reference bundle, NPU selected subgraph receipts, and Arc 140V native OpenCL parity; merged in #4110. |
| CPU258V-017 | merged | Add a BitNet prompt/token authority audit receipt for the shared 258V bad-answer/input-contract investigation; merged in #4123. |
| CPU258V-018 | merged | Compare official HF `AutoTokenizer.apply_chat_template` rendered prompts and token IDs against BitNet-rs metadata-authoritative prompt-authority audit output for fixed 258V prompts; merged in #4178. |
| CPU258V-019 | merged | Capture the external first-token reference boundary from HF or bitnet.cpp for the fixed 258V prompts, recording generated token/text when available and explicit missing-logits status without claiming logits parity; merged in #4248. |
| CPU258V-020 | merged | Classify first-token divergence using external reference evidence, prompt-authority audit output, and 258V scalar/AVX2 receipts, preserving inconclusive status when reference generated token IDs or logits are unavailable; merged in #4295. |
| CPU258V-021 | merged | Instrument or script the external BitNet reference boundary so generated-token IDs and first-token logits/top-k are captured when available, or blocked with precise evidence when the reference cannot expose them; merged in #4315. |
| CPU258V-022 | merged | Audit 258V CPU scalar QK256/I2_S/I8_S semantics against the canonical BitNet.cpp/CUDA-aligned oracle, covering code mapping, packed bitplane layout, inline scale handling, activation scale use, and accumulator scaling order; merged in #4321. |
| CPU258V-023 | merged | Audit the 258V CPU output-head and logits-index boundary by recording tensor identity, tied/output-head policy, vocab/logit length, EOS/stop IDs, top-k token IDs, and decoded top-k strings without answer-quality, speed, Arc/NPU, or full-model claims; merged in #4329. |
| CPU258V-024 | merged | Capture observed runtime logits vector length evidence from the 258V CPU generation/eval path so the expected tokenizer/output-head boundary can be checked against real logits before deeper transformer layer parity; merged in #4342. |
| CPU258V-025 | merged | Add a 258V CPU transformer-layer parity ladder to classify the first internal divergence after prompt/token, QK256 semantics, output-head, and logits-index boundaries are recorded; merged in #4356. |
| CPU258V-026 | merged | Refresh the 258V CPU reference bundle after the semantic-debug ladder through transformer-layer parity; merged in #4365. |
| CPU258V-027 | ready | Add a 258V CPU semantic diagnosis artifact that classifies the current blocker and recommended next fix from the CPU reference bundle evidence without runtime changes or new answer-quality, speed, Arc, or NPU claims. |
| LNL258V-004 | merged | Add Windows Level Zero loader fallback and refresh the 258V platform probe so Arc 140V records Level Zero identity and PCI ID `0x64A0`; merged in #4148. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
