# CUDA-MODEL-017 Qwen3 Repeated Comparator Capture Preflight

Status: source-capture preflight only

This report records the operator manifest for `CUDA-MODEL-017`, the Qwen3
0.6B Q8_0 repeated same-artifact CPU/CUDA comparator capture step on the
Windows 9950X3D + RTX 5070 Ti lane.

The generated manifest is:

```text
docs/reports/CUDA_MODEL_017_QWEN3_REPEATED_COMPARATOR_SOURCE_MANIFEST.json
```

## What This Proves

The manifest generator runs and emits the required source receipt plan for the
`qwen3_cuda_repeated_comparator` aggregate:

```powershell
cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- --manifest-out docs/reports/CUDA_MODEL_017_QWEN3_REPEATED_COMPARATOR_SOURCE_MANIFEST.json
```

The manifest is not a hardware receipt, does not prove CUDA execution, and does
not satisfy `CUDA-MODEL-017`.

## Required Source Receipts

`CUDA-MODEL-017` requires three source receipts for each profile:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
decode_128_from_warm_context
```

The manifest names 15 required source paths under:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/
```

Current repository state:

```text
required source receipts: 15
present source receipts: 0
missing source receipts: 15
```

## Required Source Fields

Every source receipt must preserve the exact Qwen3 model identity, strict CUDA
route, fallback rejection, quality/parity result, timing, transfer, launch,
VRAM, power, and thermal fields listed in the JSON manifest.

The aggregate must keep these claims false:

```text
speedup_claim
benchmark_qualified_speedup
full_cuda_residency_claimed
broad_dense_gguf_ready_claimed
qwen25_proof_inherited
bitnet_packed_i2s_qk256_proof
```

## Next Action

Run the five profiles on the Windows 9950X3D + RTX 5070 Ti machine, commit the
15 source receipts, then generate the aggregate:

```powershell
cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- `
  --one-token-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-one-token-cuda.json `
  --one-token-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-one-token-cuda.json `
  --one-token-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-one-token-cuda.json `
  --short-decode-8-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-short-decode-8-cuda.json `
  --short-decode-8-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-short-decode-8-cuda.json `
  --short-decode-8-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-short-decode-8-cuda.json `
  --short-decode-32-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-short-decode-32-cuda.json `
  --short-decode-32-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-short-decode-32-cuda.json `
  --short-decode-32-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-short-decode-32-cuda.json `
  --warm-session-3-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-warm-session-3-cuda.json `
  --warm-session-3-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-warm-session-3-cuda.json `
  --warm-session-3-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-warm-session-3-cuda.json `
  --decode-128-from-warm-context-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-decode-128-from-warm-context-cuda.json `
  --decode-128-from-warm-context-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-decode-128-from-warm-context-cuda.json `
  --decode-128-from-warm-context-run ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-decode-128-from-warm-context-cuda.json `
  --receipt-out ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-0_6b-repeated-comparator.json
```

`CUDA-MODEL-018` remains blocked until that aggregate exists and validates.
