# BitNet Capability Matrix

## Official Microsoft BitNet 2B

This status page summarizes the official `microsoft/BitNet-b1.58-2B-4T` model
family without widening any support claim. The machine-readable authority
remains `ci/model-artifacts/model-coverage-matrix.toml`; this page is the
human-facing map for the official 2B rows.

| Route | Coverage row | Current tier | Answer status | Speed | Residency | Server |
|---|---|---|---|---|---|---|
| I2_S/QK256 GGUF | `bitnet_official_2b_i2s_qk256` | `product_cli_ready` | CPU and accelerator answer-ready for the bounded official I2_S/QK256 lane | `speedup_claim=false` | `full_residency_claim=false` | Broad `server_ready=false`; exact-profile receipts do not imply broad production readiness |
| TL1 ARM | `bitnet_official_2b_tl1_arm_candidate` | `registered` | Candidate only | false | false | false |
| TL2 x86 | `bitnet_official_2b_tl2_x86_candidate` | `registered` | Candidate only | false | false | false |
| BF16 master to GPU int2/W2A8 | `bitnet_official_2b_bf16_gpu_int2_candidate` | `registered` | Candidate only | false | false | false |

## Required Not-Claims

- TL1, TL2, and BF16/GPU-int2 do not inherit I2_S/QK256 answer or backend proof.
- CUDA proof does not satisfy Apple, A770/OpenCL, CPU, TL1, TL2, or
  BF16/GPU-int2 proof.
- Dense regular-LLM CUDA proof does not satisfy BitNet packed-kernel proof.
- No-scale F32 diagnostic QK256 does not satisfy production I2_S proof.
- Upload-once weights and QK256 linears do not prove full device residency.
- Exact-profile server smoke does not prove broad production server readiness.

## Next Proof Families

1. Artifact, tokenizer, and prompt contracts for the official 2B family.
2. I2_S/QK256 and TL1/TL2 route contracts.
3. CPU, CUDA, Apple, and A770/OpenCL backend contracts.
4. Quality, performance, residency, server, and status-surface contracts.
5. Route-specific receipts before any promotion beyond the current I2_S/QK256
   bounded product CLI state.


## TL2 source-of-truth

- TL2 x86 remains `registered`/candidate until artifact, layout, scalar oracle, reference-good, and strict CPU receipts pass.
- ARM TL2 remains `unsupported_upstream` for tracked families.
- TL2 does not inherit I2_S/QK256 or TL1 proof.
