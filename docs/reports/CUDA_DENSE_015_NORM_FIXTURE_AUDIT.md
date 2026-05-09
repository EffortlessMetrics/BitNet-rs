# CUDA-DENSE-015 Dense GGUF Norm Fixture Audit

`CUDA-DENSE-015` extracts the first-layer dense GGUF RMSNorm tensors from the
verified Qwen2.5 0.5B Q8_0 artifact and computes deterministic CPU reference
outputs. This is the first concrete fixture for the top gap recorded by the
one-layer planner audit: `attention_norm` and `ffn_norm`.

This is not a CUDA RMSNorm implementation and not dense GGUF inference. The
receipt records `cuda_kernel_status=missing_cuda_kernel`, so the next proof
must add a CUDA RMSNorm parity fixture before one-layer strict CUDA execution
can advance.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-norm-fixture-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_norm_fixture_extraction` |
| `model.model_family` | `qwen` |
| `model.architecture` | `qwen2` |
| `descriptor_coverage.tensor_count` | `291` |
| `descriptor_coverage.quantization_families` | `f32`, `q8_0` |
| `norm_fixture_audit.roles_total` | `2` |
| `norm_fixture_audit.covered_roles` | `attention_norm`, `ffn_norm` |
| `norm_fixture_audit.cuda_kernel_status` | `missing_cuda_kernel` |
| `norm_fixture_audit.strict_cuda_ready` | `false` |
| `norm_fixture_audit.cpu_fallback_allowed` | `false` |
| `norm_fixture_audit.transfer_timing_status` | `not_measured_no_kernel` |

Extracted tensors:

| Role | Tensor | Hidden Dim | Tensor Type | Epsilon Source |
| --- | --- | ---: | --- | --- |
| `attention_norm` | `blk.0.attn_norm.weight` | `896` | `f32` | `qwen2.attention.layer_norm_rms_epsilon` |
| `ffn_norm` | `blk.0.ffn_norm.weight` | `896` | `f32` | `qwen2.attention.layer_norm_rms_epsilon` |

## May Claim

- Dense GGUF `attention_norm` and `ffn_norm` tensors can be selected from the
  verified Qwen2.5 0.5B Q8_0 GGUF artifact.
- The tensors can be materialized as F32 and evaluated by a deterministic CPU
  RMSNorm reference.
- The receipt records the current CUDA RMSNorm gap as `missing_cuda_kernel`.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA RMSNorm parity exists;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- speedup is accepted;
- full CUDA residency is proven;
- tokenizer, loader, transformer, server, or QK256 math changed.

## Validation

```powershell
cargo test --locked -p bitnet-models --lib --no-default-features dense_gguf_norm -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,full-cli dense_gguf_norm -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_norm -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- dense-gguf-norm-fixture --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-norm-fixture-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-norm-fixture-qwen25-q8.json
```

## Next Step

The next scoped lane should add a CUDA RMSNorm fixture/parity proof for these
same extracted norm fixtures. It should still avoid dense GGUF one-token or
decode claims until RMSNorm, RoPE, attention, and activation gaps are closed.
