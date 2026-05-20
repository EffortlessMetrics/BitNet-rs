# CUDA-DENSE-QWEN25-OPS-004 Logits Transfer Reduction Contract

Date: 2026-05-20
Work item: CUDA-DENSE-QWEN25-OPS-004
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0
Coverage row: `dense_qwen25_05b_q8_cuda`
Linked plan: `plans/native-rust-inference/dense-qwen25.md`
Linked spec: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`

## Summary

CUDA-DENSE-QWEN25-PERF-007 rejected Qwen2.5 speed and full-residency
promotion because the reviewed CUDA profiles remain slower than same-artifact
CPU means, pure H2D timing is unavailable, and logits transfer still downloads
full logits to the CPU sampler.

This slice tightens the receipt contract for the next optimization. A receipt
can no longer claim `device_to_host_bytes_reduced=true` only by lowering byte
counts. It must also prove that selection moved to a CUDA device-side sampler
path.

## Contract

A dense Qwen logits-transfer reduction receipt remains valid without a
reduction section for legacy receipts. When the section is present:

- non-reduced receipts must use `transfer_mode=full_logits_download_cpu_sampler`,
  `sampling_location=cpu`, full-logits D2H byte accounting, a non-empty
  `reduction_blocker`, selected-token equality, preserved top-k evidence, and
  unchanged quality receipts;
- reduced receipts must use `transfer_mode=device_top_k_cuda_sampler` or
  `transfer_mode=device_greedy_cuda_sampler`;
- reduced receipts must use `sampling_location=cuda_device`;
- reduced receipts must record `actual_device_to_host_bytes <
  full_logits_download_bytes`;
- `bytes_saved_vs_full_logits` must equal the measured byte reduction;
- `reduction_blocker` must be omitted or null after reduction is claimed;
- selected-token equality, top-k evidence, and quality receipts must remain
  preserved.

## Claim Boundary

May claim:

- the receipt validator now rejects fake reduced-D2H receipts that still name
  the CPU full-logits sampler;
- future reduced-D2H Qwen2.5 receipts must carry device-side sampler identity
  and measured byte reduction.

Must not claim:

- Qwen2.5 speedup;
- `benchmark_qualified=true`;
- full CUDA residency;
- pure H2D event copy timing;
- a runtime device top-k sampler exists;
- broad dense GGUF server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence.

## Validation

```powershell
rtk cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_qwen_short_decode
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk git diff --check
```
