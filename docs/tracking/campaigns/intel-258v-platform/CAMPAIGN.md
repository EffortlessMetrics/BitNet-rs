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

Current `low_power` battery-mode evidence collection is tracked by
`LNL258V-POWER-006`. The operator runbook is
`docs/hardware/intel-258v-low-power-battery-runbook.md`. It is a checklist and
claim-boundary document, not battery evidence by itself.

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
| CPU258V-027 | merged | Add a 258V CPU semantic diagnosis artifact that classifies the current blocker and recommended next fix from the CPU reference bundle evidence without runtime changes or new answer-quality, speed, Arc, or NPU claims; merged in #4508. |
| CPU258V-028 | merged | Fix the metadata-authoritative BitNet prompt policy mismatch identified by CPU258V-027 so fixed prompt strings and token IDs match the external HF `apply_chat_template` boundary; merged in #4512. |
| CPU258V-029 | merged | Rerun scalar and AVX2 BitNet answer-corpus receipts after the prompt-policy fix and classify any remaining tiny-corpus failures without broad answer-quality, speed, Arc/NPU, or QK256 claims; merged in #4516. |
| CPU258V-030 | merged | Rerun strict CPU warm-session phase receipts after the prompt-policy fix and fixed-corpus pass without speedup, sustained-throughput, Arc/NPU, QK256, or full-model claims; merged in #4520. |
| CPU258V-031 | merged | Refresh the 258V CPU reference bundle after the prompt-policy fix, answer-corpus pass, scalar/AVX2 parity, and corrected warm-session phase receipts without new runtime or accelerator claims; merged in #4523. |
| SLM258V-001 | merged | Add a Lunar Lake dense SLM artifact manifest as a separate dense-model path, not a BitNet QK256/I2_S receipt; merged in #4527. |
| LNL258V-004 | merged | Add Windows Level Zero loader fallback and refresh the 258V platform probe so Arc 140V records Level Zero identity and PCI ID `0x64A0`; merged in #4148. |
| SLM258V-002 | merged | Run the pinned Qwen2.5 dense SLM candidate through a strict 258V CPU answer smoke with `fallback_used=false`; current #4530 evidence is diagnostic until BitNet I2_S provenance is removed from the dense SLM receipt; merged in #4530. |
| SLM258V-003 | merged | Separate dense Qwen SLM receipt provenance from BitNet I2_S/QK256 kernel/layout fields and rerun the 258V Qwen2.5 CPU smoke with clean dense SLM provenance; merged in #4535. |
| SLM258V-004 | merged | Record dense Qwen SLM phase timing receipts on the 258V CPU path, keeping dense SLM phase evidence separate from BitNet QK256/I2_S receipts; merged in #4542. |
| LNL258V-COMPARE-004 | merged | Refresh the same-machine comparison index after the corrected BitNet CPU bundle and dense Qwen SLM CPU answer/phase receipts, preserving independent BitNet CPU, dense SLM CPU, Arc 140V, and NPU claim boundaries; merged in #4545. |
| CPU258V-032 | merged | Harden the 258V post-fix scalar-vs-AVX2 answer-parity receipt so top-level backend/runtime/fallback/kernel identity is explicit; merged in #4550. |
| SLM258V-005 | merged | Harden dense Qwen SLM answer and phase receipts so top-level backend/runtime/fallback/model identity is explicit before OpenVINO CPU/GPU/NPU acceleration work; merged in #4552. |
| SLM-OV258V-001 | merged | Record the Qwen2.5 0.5B Instruct OpenVINO IR INT4 symmetric export manifest for the Lunar Lake dense SLM operating lane, linked to the clean 258V GGUF CPU answer and phase baseline without committing model binaries or claiming OpenVINO CPU/GPU/NPU execution; merged in #4559. |
| SLM-OV258V-002 | merged | Run the Qwen2.5 OpenVINO CPU LLMPipeline answer smoke when the exported INT4 symmetric IR model and openvino_genai runtime are available, or record a blocked-before-execution receipt with the exact missing prerequisites and no execution claim; merged in #4565. |
| SLM-OV258V-002A | merged | Refresh the Qwen2.5 OpenVINO CPU LLMPipeline smoke with live CPU execution evidence now that the local OpenVINO GenAI runtime and INT4 symmetric IR export are available; merged in #4571. |
| SLM-OV258V-003 | merged | Record the Qwen2.5 OpenVINO GPU/Arc 140V LLMPipeline bounded smoke with `fallback_used=false`; merged in #4584. |
| SLM-OV258V-004 | merged | Record the Qwen2.5 OpenVINO NPU / Intel AI Boost LLMPipeline bounded smoke with `fallback_used=false`; merged in #4588. |
| SLM-OV258V-005 | merged | Compare available Qwen2.5 GGUF CPU and OpenVINO CPU/GPU/NPU answer-gate, fallback, and timing fields while recording granular OpenVINO phase gaps; merged in #4591. |
| SLM-OV258V-006 | merged | Add a Qwen2.5 OpenVINO GenAI phase runner for CPU, GPU.0/Arc 140V, and NPU PerfMetrics plus first streamed text chunk timing; merged in #4594. |
| CPU-BITNET-REF-001 | merged | Record the external Microsoft BitNet.cpp generated-text boundary against the corrected 258V CPU reference bundle while preserving generated-token/logit gaps; merged in #4599. |
| CPU-BITNET-REF-002 | merged | Harden the first-token divergence classifier so future direct BitNet.cpp generated-token evidence is consumed as token evidence instead of falling through to an unknown summary; merged in #4710. |
| CPU-BITNET-REF-003 | merged | Record direct Microsoft BitNet.cpp generated-token IDs and first-token top-k/logit evidence for the fixed 258V prompts, then rerun the first-token divergence classifier without broad answer-quality, speed, Arc/NPU, QK256, or full-correctness claims; merged in #4715. |
| CPU-BITNET-PERF-001 | merged | Recorded 258V QK256/I2_S GEMV and GEMM microbench timing receipts with fallback=false and speedup_claim=false in #4607. No answer-quality, sustained-throughput, Arc/NPU, acceleration, QK256 semantic-change, or full-model claims. |
| CPU-BITNET-PERF-002 | merged | #4609 merged with SHA 9bbb45712eff819daffaea61599011626d4f4579, recording a 258V QK256/I2_S tiling/thread candidate matrix with sampled GEMV/GEMM timings, fallback=false, speedup_claim=false, and explicit thread-count-not-applied status. No answer-quality, sustained-throughput, Arc/NPU, acceleration, QK256 semantic-change, or full-model claims. |
| CPU-BITNET-PERF-003 | merged | #4689 merged with SHA f41e923e8c2a4e5da84d0f58f5a117ee0a305f04, recording a 258V QK256/I2_S applied-thread microbench receipt with scoped row-partitioned GEMV and token-partitioned GEMM samples, fallback=false, speedup_claim=false, and no full-runtime worker-policy claim. |
| LNL258V-OP-001 | merged | Add a Lunar Lake operator readiness command that indexes existing CPU, dense SLM OpenVINO, Arc, and NPU receipts into explicit route reasons without running inference or claiming acceleration; merged in #4620. |

| LNL258V-REG-001 | merged | Add a Lunar Lake local regression bundle that checks the operator-readiness receipt for dense CPU default routing, BitNet CPU reference routing, bounded OpenVINO GPU/NPU candidates, Arc/NPU claim boundaries, no hidden fallback, and no acceleration claim without running inference; merged in #4624. |
| LNL258V-COMPARE-005 | merged | Add a Lunar Lake operator comparison command that reads the committed operator-readiness and regression-bundle receipts, indexes route evidence readiness and claim boundaries, and emits a comparison artifact without running inference or claiming acceleration; merged in #4635. |
| LNL258V-ASK-001 | merged | Add a policy-gated `bitnet lunar-lake ask` wrapper for the dense Qwen CPU default route. It enforces the operator-readiness receipt before generation, preserves no-fallback CPU route identity, writes an operator ask receipt with source run evidence, and makes no broad quality, speedup, Arc/NPU execution, acceleration, or BitNet QK256/I2_S claim; merged in #4644. |
| LNL258V-ASK-002 | merged | Restore the live dense Qwen2.5 CPU operator ask route for bounded math prompts, preserve dedicated `output.weight` receipt identity, add an optional `--expect-contains` answer gate, and commit live 258V ask receipts without claiming broad dense SLM quality, speedup, Arc/NPU execution, acceleration, or BitNet QK256/I2_S proof; merged in #4654. |
| LNL258V-ASK-003 | merged | Harden Lunar Lake operator ask receipts so backend/runtime/fallback/model/template identity is explicit at top level while preserving source-run evidence and the no broad quality/speed/Arc/NPU/BitNet claim boundary; merged in #4668. |
| LNL258V-ASK-004 | merged | Added an explicit OpenVINO GenAI operator-ask helper for dense Qwen GPU/NPU candidate routes, recording bounded answer receipts with fallback=false and no acceleration or speedup claim; merged in #4712. |
| CPU-BITNET-EMBD-001 | merged | #4684 recorded 258V BitNet embedding quantization evidence from the committed tensor-boundary audit, including current F16 embedding state and explicit Q6_K-not-active boundary unless a Q6_K embedding artifact is present. |
| LNL258V-OP-002 | merged | Refresh operator readiness, regression, and comparison receipts so the BitNet CPU route indexes direct BitNet.cpp generated-token/logit evidence, direct first-token divergence classification, applied-thread microbench evidence, and embedding-quantization boundary without new inference or acceleration claims; merged in #4718. |
| LNL258V-OP-003 | merged | Refresh operator readiness, regression, and comparison receipts so OpenVINO GPU/NPU dense SLM candidate routes index the bounded operator-ask receipts as answer evidence while preserving smoke receipts as supporting evidence; merged in #4721. |
| LNL258V-ROUTE-001 | merged | Added a policy-only Lunar Lake route promotion ledger that promotes dense Qwen CPU for bounded ask profiles while keeping OpenVINO GPU/NPU as candidates until profile-specific advantage evidence exists; merged in #4771. |
| LNL258V-ROUTE-002 | merged | Added a policy/evidence-only Lunar Lake route profile comparison that indexes promoted and candidate routes across fixed workload profiles, adds the low-power profile gap, and preserves CPU default routing until benchmark-qualified profile evidence exists; merged in #4798. |
| LNL258V-QUAL-001 | merged | Added a bounded Lunar Lake dense Qwen answer corpus v2 covering math, copy, yes/no, factual, instruction, stop/EOS, transcript context, structured output, long-prompt summary, and decode-heavy route checks without running inference or claiming broad quality; merged in #4806. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
