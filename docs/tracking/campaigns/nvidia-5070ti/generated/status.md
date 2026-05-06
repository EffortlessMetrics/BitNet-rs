<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# NVIDIA RTX 5070 Ti validation Campaign Status

- Campaign: `nvidia-5070ti`
- State: `active`
- Objective: Validate RTX 5070 Ti as a CUDA-first BitNet acceleration lane with selected-device receipts and no CPU, OpenCL, WGPU, or generic GPU conflation.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| RTX5070TI-003 | merged | #3679 | `codex/rtx5070ti-003-backend-identity` | Preserve RTX 5070 Ti requested and selected CUDA backend identity without adding kernels or inference claims. |
| RTX5070TI-004 | merged | #3691 | `codex/nvidia-5070ti/RTX5070TI-004-cuda-nvml-probe` | Add RTX 5070 Ti CUDA and NVML runtime probe without claiming kernel execution or BitNet inference. |
| RTX5070TI-005 | merged | #3723 | `codex/nvidia-5070ti/RTX5070TI-005-smoke-receipt` | Compile and run a tiny CUDA kernel on the selected RTX 5070 Ti with a fallback-free smoke receipt and no BitNet inference or speedup claim. |
| RTX5070TI-006 | merged | #3749 | `codex/nvidia-5070ti/RTX5070TI-006-cuda-cpu-parity` | Run one deterministic fixture through 9950X3D CPU reference and RTX 5070 Ti CUDA target, record error metrics and fallback-free selected-device identity, and emit mismatch debug artifacts without full inference claims. |
| RTX5070TI-007 | merged | #3756 | `codex/nvidia-5070ti/RTX5070TI-007-cuda-receipts-counters` | Record CUDA runtime identity and kernel invocation counters in smoke and parity receipts, including fallback-free strict validation, without adding benchmarks or BitNet inference claims. |
| RTX5070TI-008 | merged | #3770 | `codex/nvidia-5070ti/RTX5070TI-008-benchmark-baseline` | Benchmark parity-tested RTX 5070 Ti CUDA kernels/subgraphs against the 9950X3D CPU reference with driver/runtime/VRAM/power/thermal context and no full inference or unproven speedup claim. |
| CUDA-BITNET-001 | merged | #3776 | `codex/cuda-bitnet-001-context-handles` | Add persistent CUDA BitNet context, stream lifetime, reusable workspace, and weight handles without claiming full BitNet inference. |
| CUDA-BITNET-002 | merged | #3782 | `codex/cuda-bitnet-002-i2s-linear` | Add reusable CUDA I2S linear primitive with CPU/CUDA parity, tails and padding support, and kernel stats without full inference claims. |
| CUDA-BITNET-003 | merged | #3786 | `codex/cuda-bitnet-003-qk256-gemv` | Replace scaffold-only QK256 launch with a compiled packed fused dequant GEMV CUDA kernel that passes CPU scalar parity. |
| CUDA-BITNET-004 | merged | #3790 | `codex/cuda-bitnet-004-upload-once-weights` | Prepack and upload BitNet weights once for CUDA with per-layer handles and receipt fields for packed_at_load and uploaded_once. |
| CUDA-BITNET-005 | merged | #3792 | `codex/cuda-bitnet-005-route-linear` | Route actual BitNetLinear transformer dispatch through the selected CUDA backend with coverage counters and strict CPU fallback rejection. |
| CUDA-BITNET-006 | pr_open | #3801 | `codex/cuda-bitnet-006-one-token-proof` | Add strict one-token BitNet CUDA proof with official GGUF, real tokenizer, CUDA kernel invocations greater than zero, zero CPU fallback, CPU/CUDA greedy or top-1 agreement, fallback_used=false, and speedup_claim=false. |
| CUDA-BITNET-007 | proposed | TBD | `codex/cuda-bitnet-007-short-decode-proof` | Add short greedy BitNet CUDA decode proof with timing, kernel invocation growth, memory high-water mark, and fallback_used=false. |
| CUDA-BITNET-008 | proposed | TBD | `codex/cuda-bitnet-008-benchmark-baseline` | Add RTX 5070 Ti full BitNet CUDA benchmark baseline comparing 9950X3D CPU scalar, AVX2, AVX-512, and CUDA with matching model, tokenizer, prompt profile, strict loader mode, fallback_used=false, and runtime context. |
| CUDA-DENSE-001 | proposed | TBD | `codex/cuda-dense-001-reference-lane` | Add regular LLM CUDA dense-kernel reference lane with dense_regular_llm receipt labels that cannot satisfy BitNet packed I2S or QK256 proof acceptance. |

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.
